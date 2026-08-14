#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "mmpack/detail/bits.hpp"
#include "mmpack/dictionary.hpp"
#include "mmpack/error.hpp"
#include "mmpack/format.hpp"
#include "mmpack/schema.hpp"
#include "mmpack/sink.hpp"

// The builder consumes records, measures them, then decides the record shape.
//
//   1. consume: intern text fields, deduplicate whole value tuples, stage
//      (key, tuple_id) pairs and check that keys arrive in order
//   2. measure: min/max per field over the distinct tuples
//   3. shape:   choose a width and bias per field, then decide inline vs
//      interned from an explicit cost model
//   4. write:   dictionaries, records, directory, schema, footer
//
// Every staged field value is one std::uint64_t regardless of kind, which is
// what keeps staging compact and serialization uniform: floats are stored as
// their bit pattern, text as a dictionary id, small byte fields packed. Writing
// a field is always store_uint(offset, width, value - bias).

namespace mmpack {

enum class interning_policy {
  automatic,  ///< decide from the cost model
  always,     ///< force whole-value interning
  never,      ///< force inline values (right for scan-heavy workloads)
};

enum class input_order {
  assume_sorted,   ///< keys must arrive strictly increasing; throw otherwise
  sort_if_needed,  ///< tolerate disorder and sort during finish()
};

struct build_options {
  /// Key split: address = key & ((1 << address_bits) - 1).
  unsigned address_bits = 32;
  interning_policy value_interning = interning_policy::automatic;
  input_order order = input_order::assume_sorted;
  /// Round each field offset up to its own width. Off by default.
  bool align_fields = false;
  /// Stop deduplicating tuples once this many records have been seen and the
  /// distinct ratio is still above `dedup_give_up_ratio`. Paying for a hash
  /// index that will not earn its keep is the thing being avoided.
  std::uint64_t dedup_sample = 1u << 20;
  double dedup_give_up_ratio = 0.9;
  /// Ceiling on dense directory slots, so one far-out partition index cannot
  /// make the builder materialize an enormous directory.
  std::uint64_t max_dense_slots = 1u << 20;
};

/// What the builder decided, and the numbers it decided from.
struct build_report {
  std::uint64_t records = 0;
  std::uint64_t distinct_values = 0;
  std::uint64_t partitions = 0;
  bool interned = false;
  bool dedup_abandoned = false;
  bool sorted_during_finish = false;
  unsigned address_width = 0;
  unsigned ref_width = 0;
  std::uint32_t value_stride = 0;
  std::uint32_t record_stride = 0;
  /// Record-and-dictionary bytes each layout would have cost, for auditing.
  std::uint64_t inline_estimate = 0;
  std::uint64_t interned_estimate = 0;
  std::uint64_t image_bytes = 0;
};

/// Where staged (key, tuple id) pairs live. An interface so a two-pass or
/// spilling implementation can replace it without touching the builder.
class record_staging {
 public:
  virtual ~record_staging() = default;
  virtual void push(std::uint64_t key, std::uint64_t tuple_id) = 0;
  virtual bool is_sorted() const noexcept = 0;
  virtual void sort() = 0;
  virtual std::uint64_t size() const noexcept = 0;
  virtual std::uint64_t key_at(std::uint64_t i) const noexcept = 0;
  virtual std::uint64_t tuple_at(std::uint64_t i) const noexcept = 0;
};

/// Straightforward in-memory staging: 16 bytes per record.
class memory_staging final : public record_staging {
 public:
  void push(std::uint64_t key, std::uint64_t tuple_id) override {
    if (!rows_.empty() && key <= rows_.back().key) sorted_ = false;
    rows_.push_back(row{key, tuple_id});
  }
  [[nodiscard]] bool is_sorted() const noexcept override { return sorted_; }
  void sort() override {
    std::sort(rows_.begin(), rows_.end(),
              [](const row& a, const row& b) { return a.key < b.key; });
    sorted_ = true;
  }
  [[nodiscard]] std::uint64_t size() const noexcept override { return rows_.size(); }
  [[nodiscard]] std::uint64_t key_at(std::uint64_t i) const noexcept override { return rows_[i].key; }
  [[nodiscard]] std::uint64_t tuple_at(std::uint64_t i) const noexcept override {
    return rows_[i].tuple;
  }

 private:
  struct row {
    std::uint64_t key;
    std::uint64_t tuple;
  };
  std::vector<row> rows_;
  bool sorted_ = true;
};

class table_builder;

/// Scratch handle for one record's field values. Fields left unset default to 0.
class record_ref {
 public:
  void set_uint(field_id id, std::uint64_t v);
  void set_sint(field_id id, std::int64_t v);
  void set_f32(field_id id, float v);
  void set_f64(field_id id, double v);
  void set_text(field_id id, std::string_view v);
  void set_bytes(field_id id, std::span<const std::byte> v);

 private:
  friend class table_builder;
  explicit record_ref(table_builder* owner, std::uint64_t key) : owner_(owner), key_(key) {}
  table_builder* owner_;
  std::uint64_t key_;
};

class table_builder {
 public:
  explicit table_builder(schema_builder schema, build_options options = {})
      : schema_(std::move(schema)),
        options_(options),
        staging_(std::make_unique<memory_staging>()),
        slots_(schema_.size(), 0),
        stats_(schema_.size()),
        index_(0, tuple_hash{&tuples_, static_cast<std::uint32_t>(schema_.size())},
               tuple_equal{&tuples_, static_cast<std::uint32_t>(schema_.size())}) {
    if (schema_.empty()) throw build_error("schema has no fields");
    if (options_.address_bits > 64) throw build_error("address_bits must be <= 64");
    for (const field_decl& f : schema_.fields()) {
      if (f.kind == field_kind::text) text_dicts_.emplace_back();
    }
  }

  table_builder(const table_builder&) = delete;
  table_builder& operator=(const table_builder&) = delete;

  /// Replace the staging strategy. Must be called before the first record.
  void set_staging(std::unique_ptr<record_staging> staging) {
    if (staging_ && staging_->size() != 0) throw build_error("set_staging() after the first record");
    staging_ = std::move(staging);
  }

  [[nodiscard]] const schema_builder& schema() const noexcept { return schema_; }

  /// Begin a record. Field values are written through the returned handle and
  /// the record is not staged until commit().
  [[nodiscard]] record_ref begin_record(std::uint64_t key) {
    std::fill(slots_.begin(), slots_.end(), std::uint64_t{0});
    return record_ref(this, key);
  }

  void commit(const record_ref& rec) {
    if (finished_) throw build_error("commit() after finish()");
    if (options_.order == input_order::assume_sorted && count_ != 0 && rec.key_ <= last_key_) {
      throw build_error("keys must arrive strictly increasing: " + std::to_string(rec.key_) +
                        " followed " + std::to_string(last_key_));
    }
    staging_->push(rec.key_, intern_tuple());
    last_key_ = rec.key_;
    ++count_;
  }

  /// Measure, choose the shape, and write the image. Returns what it decided.
  template <byte_sink Sink>
  build_report finish(Sink& sink) {
    if (finished_) throw build_error("finish() called twice");
    finished_ = true;

    build_report report;
    report.records = count_;
    report.distinct_values = tuple_count();
    report.dedup_abandoned = !dedup_active_;

    if (!staging_->is_sorted()) {
      if (options_.order == input_order::assume_sorted) {
        throw build_error("input was not key-sorted");  // unreachable: commit() throws first
      }
      staging_->sort();
      report.sorted_during_finish = true;
      check_sorted_unique();
    }

    choose_layout(report);
    write_image(sink, report);
    return report;
  }

 private:
  friend class record_ref;

  // --- staging ------------------------------------------------------------

  struct field_stats {
    std::uint64_t min = ~std::uint64_t{0};
    std::uint64_t max = 0;
    bool seen = false;
  };

  struct tuple_hash {
    const std::vector<std::uint64_t>* tuples;
    std::uint32_t width;
    [[nodiscard]] std::size_t operator()(std::uint64_t id) const noexcept {
      const std::uint64_t* p = tuples->data() + id * width;
      std::size_t h = 1469598103934665603ull;  // FNV-1a over the slot values
      for (std::uint32_t i = 0; i < width; ++i) {
        h ^= static_cast<std::size_t>(p[i]);
        h *= 1099511628211ull;
      }
      return h;
    }
  };

  struct tuple_equal {
    const std::vector<std::uint64_t>* tuples;
    std::uint32_t width;
    [[nodiscard]] bool operator()(std::uint64_t a, std::uint64_t b) const noexcept {
      return std::memcmp(tuples->data() + a * width, tuples->data() + b * width,
                         width * sizeof(std::uint64_t)) == 0;
    }
  };

  [[nodiscard]] std::uint64_t tuple_count() const noexcept {
    return slots_.empty() ? 0 : tuples_.size() / slots_.size();
  }

  /// Append the scratch slots as a tuple, collapsing onto an existing one when
  /// dedup is still active.
  std::uint64_t intern_tuple() {
    const std::uint32_t width = static_cast<std::uint32_t>(slots_.size());
    const std::uint64_t candidate = tuple_count();
    tuples_.insert(tuples_.end(), slots_.begin(), slots_.end());

    if (dedup_active_) {
      const auto [it, inserted] = index_.insert(candidate);
      if (!inserted) {
        tuples_.resize(tuples_.size() - width);  // collapse onto the existing tuple
        return *it;
      }
      maybe_give_up_on_dedup();
    }
    return candidate;
  }

  /// Dedup only pays when values actually repeat. If the distinct ratio is still
  /// high after a sample, drop the hash index and keep the memory.
  void maybe_give_up_on_dedup() {
    if (count_ + 1 < options_.dedup_sample) return;
    const double ratio = static_cast<double>(tuple_count()) / static_cast<double>(count_ + 1);
    if (ratio <= options_.dedup_give_up_ratio) return;
    dedup_active_ = false;
    index_.clear();
    index_.rehash(0);
  }

  void check_sorted_unique() const {
    for (std::uint64_t i = 1; i < staging_->size(); ++i) {
      if (staging_->key_at(i) <= staging_->key_at(i - 1)) {
        throw build_error("duplicate key after sorting: " + std::to_string(staging_->key_at(i)));
      }
    }
  }

  // --- measurement and shape ----------------------------------------------

  void observe(field_id id, std::uint64_t raw) {
    field_stats& s = stats_[id];
    if (!s.seen) {
      s.seen = true;
      s.min = raw;
      s.max = raw;
      return;
    }
    const auto kind = schema_[id].kind;
    if (kind == field_kind::sint_) {
      // Compare in signed space so the range, and therefore the width, is right.
      if (static_cast<std::int64_t>(raw) < static_cast<std::int64_t>(s.min)) s.min = raw;
      if (static_cast<std::int64_t>(raw) > static_cast<std::int64_t>(s.max)) s.max = raw;
    } else {
      if (raw < s.min) s.min = raw;
      if (raw > s.max) s.max = raw;
    }
  }

  void choose_layout(build_report& report) {
    const std::uint32_t n = static_cast<std::uint32_t>(schema_.size());
    descs_.assign(n, field_desc{});

    std::uint32_t offset = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      const field_decl& decl = schema_[i];
      field_stats& s = stats_[i];
      if (!s.seen) {
        s.min = 0;
        s.max = 0;
      }

      field_desc& d = descs_[i];
      d.kind = static_cast<std::uint8_t>(decl.kind);
      d.dict_index = 0;
      d.reserved = 0;
      d.bias = 0;

      switch (decl.kind) {
        case field_kind::uint_:
          d.bias = s.min;
          d.width = detail::width_for(s.max - s.min);
          break;
        case field_kind::sint_:
          // Storing (value - min) as unsigned makes negatives cost nothing
          // extra: only the span between min and max drives the width.
          d.bias = s.min;
          d.width = detail::width_for(static_cast<std::uint64_t>(s.max) -
                                      static_cast<std::uint64_t>(s.min));
          break;
        case field_kind::f32:
          d.width = 4;
          break;
        case field_kind::f64:
          d.width = 8;
          break;
        case field_kind::bytes:
          d.width = decl.fixed_width;
          break;
        case field_kind::text: {
          const std::uint64_t entries = text_dicts_[text_slot(i)].size();
          d.width = detail::width_for(entries == 0 ? 0 : entries - 1);
          d.dict_index = static_cast<std::uint16_t>(text_slot(i));
          break;
        }
      }

      if (options_.align_fields) offset = static_cast<std::uint32_t>(detail::align_up(offset, d.width));
      d.offset = offset;
      offset += d.width;
    }
    value_stride_ = offset;

    // Address width comes from the data too, not from address_bits.
    std::uint64_t max_address = 0;
    std::uint64_t max_partition = 0;
    for (std::uint64_t i = 0; i < staging_->size(); ++i) {
      const std::uint64_t key = staging_->key_at(i);
      max_address = std::max(max_address, address_of(key));
      max_partition = std::max(max_partition, partition_of(key));
    }
    address_width_ = detail::width_for(max_address);
    (void)max_partition;

    // The cost model. Dictionary bytes for text fields are the same either way,
    // so they cancel and only records plus the composite dictionary matter.
    const std::uint64_t distinct = tuple_count();
    ref_width_ = detail::width_for(distinct == 0 ? 0 : distinct - 1);
    const std::uint64_t inline_bytes = count_ * (address_width_ + value_stride_);
    const std::uint64_t interned_bytes = count_ * (address_width_ + ref_width_) +
                                         fixed_dictionary_size(distinct, value_stride_);

    switch (options_.value_interning) {
      case interning_policy::always: interned_ = true; break;
      case interning_policy::never: interned_ = false; break;
      case interning_policy::automatic: interned_ = interned_bytes < inline_bytes; break;
    }

    record_stride_ = address_width_ + (interned_ ? ref_width_ : value_stride_);

    report.interned = interned_;
    report.address_width = address_width_;
    report.ref_width = ref_width_;
    report.value_stride = value_stride_;
    report.record_stride = record_stride_;
    report.inline_estimate = inline_bytes;
    report.interned_estimate = interned_bytes;
  }

  [[nodiscard]] std::uint32_t text_slot(field_id id) const {
    std::uint32_t slot = 0;
    for (std::uint32_t i = 0; i < id; ++i) {
      if (schema_[i].kind == field_kind::text) ++slot;
    }
    return slot;
  }

  [[nodiscard]] std::uint64_t partition_of(std::uint64_t key) const noexcept {
    return options_.address_bits >= 64 ? 0 : key >> options_.address_bits;
  }
  [[nodiscard]] std::uint64_t address_of(std::uint64_t key) const noexcept {
    return options_.address_bits >= 64
               ? key
               : key & ((std::uint64_t{1} << options_.address_bits) - 1);
  }

  // --- writing -------------------------------------------------------------

  /// Serialize one tuple into `value_stride_` bytes.
  void encode_tuple(std::uint64_t tuple, std::byte* out) const {
    const std::uint32_t n = static_cast<std::uint32_t>(slots_.size());
    const std::uint64_t* values = tuples_.data() + tuple * n;
    std::memset(out, 0, value_stride_);
    for (std::uint32_t i = 0; i < n; ++i) {
      const field_desc& d = descs_[i];
      detail::store_uint(out + d.offset, d.width, values[i] - d.bias);
    }
  }

  template <byte_sink Sink>
  void write_image(Sink& sink, build_report& report) {
    std::uint64_t pos = 0;
    const auto emit = [&](const void* data, std::size_t n) {
      sink.write(data, n);
      pos += n;
    };
    const auto pad_to = [&](std::uint64_t alignment) {
      static constexpr std::byte zeros[64] = {};
      const std::uint64_t target = detail::align_up(pos, alignment);
      while (pos < target) {
        const std::uint64_t chunk = target - pos;
        emit(zeros, static_cast<std::size_t>(chunk < sizeof(zeros) ? chunk : sizeof(zeros)));
      }
    };

    const header head{{magic[0], magic[1], magic[2], magic[3], magic[4], magic[5], magic[6],
                       magic[7]},
                      format_version, 0};
    emit(&head, sizeof(head));

    // Dictionaries first: records reference them by index, and their offsets go
    // into the schema, which is written last.
    std::vector<dict_entry> dicts;
    for (const blob_dictionary_builder& dict : text_dicts_) {
      pad_to(8);
      const std::uint64_t at = pos;
      dict.write(sink);
      pos += dict.serialized_size();
      dicts.push_back(dict_entry{at, dict.serialized_size()});
    }

    std::uint32_t value_dict_index = 0;
    if (interned_) {
      pad_to(8);
      const std::uint64_t at = pos;
      const std::uint64_t distinct = tuple_count();
      write_fixed_dictionary_header(sink, distinct, value_stride_);
      pos += sizeof(dict_header);
      std::vector<std::byte> scratch(value_stride_);
      for (std::uint64_t t = 0; t < distinct; ++t) {
        encode_tuple(t, scratch.data());
        emit(scratch.data(), scratch.size());
      }
      value_dict_index = static_cast<std::uint32_t>(dicts.size());
      dicts.push_back(dict_entry{at, fixed_dictionary_size(distinct, value_stride_)});
    }

    // Records, grouped into partitions. Input is key-ordered, so partitions come
    // out non-decreasing and a directory slot opens whenever it changes.
    pad_to(8);
    std::vector<dir_entry> directory;
    std::vector<std::byte> record(record_stride_);
    for (std::uint64_t i = 0; i < staging_->size(); ++i) {
      const std::uint64_t key = staging_->key_at(i);
      const std::uint64_t partition = partition_of(key);
      if (directory.empty() || directory.back().partition != partition) {
        directory.push_back(dir_entry{partition, pos, 0, 0, 0});
      }
      std::memset(record.data(), 0, record.size());
      detail::store_uint(record.data(), address_width_, address_of(key));
      if (interned_) {
        detail::store_uint(record.data() + address_width_, ref_width_, staging_->tuple_at(i));
      } else {
        encode_tuple(staging_->tuple_at(i), record.data() + address_width_);
      }
      emit(record.data(), record.size());
      ++directory.back().count;
    }

    pad_to(8);
    const std::uint64_t dir_offset = pos;
    const bool dense = use_dense(directory);
    std::uint64_t dir_count = 0;
    if (dense) {
      std::uint64_t next = 0;
      for (const dir_entry& d : directory) {
        for (; next < d.partition; ++next) {
          const dir_entry hole{next, 0, 0, 0, 0};
          emit(&hole, sizeof(hole));
          ++dir_count;
        }
        emit(&d, sizeof(d));
        ++dir_count;
        next = d.partition + 1;
      }
    } else {
      for (const dir_entry& d : directory) {
        emit(&d, sizeof(d));
        ++dir_count;
      }
    }

    pad_to(8);
    const std::uint64_t schema_offset = pos;
    write_schema(emit, dicts, value_dict_index);
    const std::uint64_t schema_size = pos - schema_offset;

    footer foot{};
    std::memcpy(foot.magic, magic, sizeof(magic));
    foot.format_version = format_version;
    foot.flags = dense ? flags::dense_directory : 0u;
    foot.record_count = count_;
    foot.dir_offset = dir_offset;
    foot.dir_count = dir_count;
    foot.schema_offset = schema_offset;
    foot.schema_size = schema_size;
    foot.image_size = pos + sizeof(footer);
    emit(&foot, sizeof(foot));

    report.partitions = directory.size();
    report.image_bytes = pos;
  }

  template <class Emit>
  void write_schema(Emit&& emit, const std::vector<dict_entry>& dicts,
                    std::uint32_t value_dict_index) {
    std::string names;
    std::vector<field_desc> out = descs_;
    for (std::uint32_t i = 0; i < out.size(); ++i) {
      out[i].name_offset = static_cast<std::uint32_t>(names.size());
      names += schema_[i].name;
      names.push_back('\0');
    }

    schema_header head{};
    head.magic = schema_magic;
    head.field_count = static_cast<std::uint32_t>(out.size());
    head.dict_count = static_cast<std::uint32_t>(dicts.size());
    head.record_stride = record_stride_;
    head.value_stride = value_stride_;
    head.name_bytes = static_cast<std::uint32_t>(names.size());
    head.address_width = static_cast<std::uint8_t>(address_width_);
    head.address_bits = static_cast<std::uint8_t>(options_.address_bits);
    head.mode = static_cast<std::uint8_t>(interned_ ? value_mode::interned
                                                    : value_mode::inline_fields);
    head.ref_width = static_cast<std::uint8_t>(interned_ ? ref_width_ : 0);
    head.value_dict = interned_ ? value_dict_index : 0;

    emit(&head, sizeof(head));
    if (!out.empty()) emit(out.data(), out.size() * sizeof(field_desc));
    if (!dicts.empty()) emit(dicts.data(), dicts.size() * sizeof(dict_entry));
    if (!names.empty()) emit(names.data(), names.size());
  }

  [[nodiscard]] bool use_dense(const std::vector<dir_entry>& directory) const {
    if (directory.empty()) return false;
    const std::uint64_t span = directory.back().partition + 1;
    return span <= options_.max_dense_slots && span <= directory.size() * 4 + 16;
  }

  schema_builder schema_;
  build_options options_;
  std::unique_ptr<record_staging> staging_;

  std::vector<std::uint64_t> slots_;   ///< scratch for the record being built
  std::vector<std::uint64_t> tuples_;  ///< flat table of distinct tuples
  std::vector<field_stats> stats_;
  std::vector<blob_dictionary_builder> text_dicts_;
  std::unordered_set<std::uint64_t, tuple_hash, tuple_equal> index_;
  std::vector<field_desc> descs_;

  std::uint64_t count_ = 0;
  std::uint64_t last_key_ = 0;
  std::uint32_t value_stride_ = 0;
  std::uint32_t record_stride_ = 0;
  unsigned address_width_ = 1;
  unsigned ref_width_ = 1;
  bool interned_ = false;
  bool dedup_active_ = true;
  bool finished_ = false;
};

// --- record_ref, defined once table_builder is complete ---------------------

inline void record_ref::set_uint(field_id id, std::uint64_t v) {
  if (id >= owner_->slots_.size()) throw build_error("field id out of range");
  if (owner_->schema_[id].kind != field_kind::uint_) {
    throw build_error("field '" + owner_->schema_[id].name + "' is not a uint field");
  }
  owner_->slots_[id] = v;
  owner_->observe(id, v);
}

inline void record_ref::set_sint(field_id id, std::int64_t v) {
  if (id >= owner_->slots_.size()) throw build_error("field id out of range");
  if (owner_->schema_[id].kind != field_kind::sint_) {
    throw build_error("field '" + owner_->schema_[id].name + "' is not a sint field");
  }
  const auto raw = static_cast<std::uint64_t>(v);
  owner_->slots_[id] = raw;
  owner_->observe(id, raw);
}

inline void record_ref::set_f32(field_id id, float v) {
  if (id >= owner_->slots_.size()) throw build_error("field id out of range");
  if (owner_->schema_[id].kind != field_kind::f32) {
    throw build_error("field '" + owner_->schema_[id].name + "' is not an f32 field");
  }
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  owner_->slots_[id] = bits;
}

inline void record_ref::set_f64(field_id id, double v) {
  if (id >= owner_->slots_.size()) throw build_error("field id out of range");
  if (owner_->schema_[id].kind != field_kind::f64) {
    throw build_error("field '" + owner_->schema_[id].name + "' is not an f64 field");
  }
  std::uint64_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  owner_->slots_[id] = bits;
}

inline void record_ref::set_text(field_id id, std::string_view v) {
  if (id >= owner_->slots_.size()) throw build_error("field id out of range");
  if (owner_->schema_[id].kind != field_kind::text) {
    throw build_error("field '" + owner_->schema_[id].name + "' is not a text field");
  }
  owner_->slots_[id] = owner_->text_dicts_[owner_->text_slot(id)].intern(v);
}

inline void record_ref::set_bytes(field_id id, std::span<const std::byte> v) {
  if (id >= owner_->slots_.size()) throw build_error("field id out of range");
  const field_decl& decl = owner_->schema_[id];
  if (decl.kind != field_kind::bytes) {
    throw build_error("field '" + decl.name + "' is not a bytes field");
  }
  if (v.size() != decl.fixed_width) {
    throw build_error("field '" + decl.name + "' expects exactly " +
                      std::to_string(decl.fixed_width) + " bytes");
  }
  std::uint64_t packed = 0;
  std::memcpy(&packed, v.data(), v.size());  // width <= 8, checked at declaration
  owner_->slots_[id] = packed;
}

}  // namespace mmpack
