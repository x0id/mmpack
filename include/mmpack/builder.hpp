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
#include <unordered_map>
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
  /// Give each partition its own table of the global tuple ids it actually uses,
  /// so its records can carry a narrower local index. Only applies to interned
  /// images, and only where it pays for the table. This is stride reduction on
  /// the hot path, so it is on by default.
  bool partition_remap = true;
  /// Store only the stretch of the address field a partition actually uses,
  /// trimming bytes that are zero in every one of its records. Region-start
  /// tables trim heavily -- IPv6 /64 starts leave two thirds of the field zero --
  /// and like the remap this is pure stride reduction on the hot path, so it is
  /// on by default.
  bool partition_address_width = true;
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
  /// False when records were staged from caller-split parts, so the image
  /// defines no 64-bit key space.
  bool has_key_mapping = true;
  unsigned address_width = 0;
  unsigned ref_width = 0;
  std::uint32_t value_stride = 0;
  std::uint32_t record_stride = 0;
  /// Partitions given a local remap table, and the record bytes that saved.
  std::uint64_t remapped_partitions = 0;
  std::uint64_t remap_saved_bytes = 0;
  /// Address bytes actually stored across all records, against the
  /// records * address_width a single global width would have cost.
  std::uint64_t address_bytes_total = 0;
  std::uint64_t address_saved_bytes = 0;
  std::uint64_t narrowed_partitions = 0;
  /// Packed bytes per directory slot, once every field is narrowed to its range.
  std::uint32_t directory_stride = 0;
  /// Record-and-dictionary bytes each layout would have cost, for auditing.
  std::uint64_t inline_estimate = 0;
  std::uint64_t interned_estimate = 0;
  std::uint64_t image_bytes = 0;
};

/// Where staged records live between consumption and layout. Records are staged
/// as (partition, address) rather than as a packed key, because a caller-split
/// key need not fit in 64 bits at all. An interface so a two-pass or spilling
/// implementation can replace it without touching the builder.
class record_staging {
 public:
  virtual ~record_staging() = default;
  virtual void push(std::uint64_t partition, std::uint64_t address, std::uint64_t tuple_id) = 0;
  virtual bool is_sorted() const noexcept = 0;
  virtual void sort() = 0;
  virtual std::uint64_t size() const noexcept = 0;
  virtual std::uint64_t partition_at(std::uint64_t i) const noexcept = 0;
  virtual std::uint64_t address_at(std::uint64_t i) const noexcept = 0;
  virtual std::uint64_t tuple_at(std::uint64_t i) const noexcept = 0;

  /// Only implemented by staging that holds addresses too wide for a scalar.
  /// The narrow implementations keep the address as an integer and have no
  /// stable bytes to hand out, so they return an empty span; the builder asks
  /// for one form or the other according to the mode it is in.
  [[nodiscard]] virtual std::span<const std::byte> address_bytes_at(std::uint64_t) const noexcept {
    return {};
  }
};

/// Staging for images that do have a 64-bit key space: the two parts are packed
/// back into one word and taken apart again on read. 16 bytes per record, which
/// on a 290M-record build is 2.3 GB less than storing the parts separately.
class memory_staging final : public record_staging {
 public:
  explicit memory_staging(unsigned address_bits) noexcept : address_bits_(address_bits) {}

  void push(std::uint64_t partition, std::uint64_t address, std::uint64_t tuple_id) override {
    const std::uint64_t key = pack(partition, address);
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
  [[nodiscard]] std::uint64_t partition_at(std::uint64_t i) const noexcept override {
    return address_bits_ >= 64 ? 0 : rows_[i].key >> address_bits_;
  }
  [[nodiscard]] std::uint64_t address_at(std::uint64_t i) const noexcept override {
    return address_bits_ >= 64 ? rows_[i].key
                               : rows_[i].key & ((std::uint64_t{1} << address_bits_) - 1);
  }
  [[nodiscard]] std::uint64_t tuple_at(std::uint64_t i) const noexcept override {
    return rows_[i].tuple;
  }

 private:
  [[nodiscard]] std::uint64_t pack(std::uint64_t partition, std::uint64_t address) const noexcept {
    return address_bits_ >= 64 ? address : (partition << address_bits_) | address;
  }

  struct row {
    std::uint64_t key;
    std::uint64_t tuple;
  };
  std::vector<row> rows_;
  unsigned address_bits_;
  bool sorted_ = true;
};

/// Staging for addresses too wide for a scalar. The width is uniform, so the
/// bytes live in a flat arena indexed by record number and need no per-record
/// offset: 16 bytes plus the address width per record.
class blob_staging final : public record_staging {
 public:
  explicit blob_staging(unsigned address_width) : width_(address_width) {}

  void push(std::uint64_t, std::uint64_t, std::uint64_t) override {
    throw build_error("blob_staging needs the address bytes; use push_bytes()");
  }

  void push_bytes(std::uint64_t partition, std::span<const std::byte> address,
                  std::uint64_t tuple_id) {
    if (address.size() != width_) throw build_error("address width changed mid-build");
    if (!rows_.empty() && !before(rows_.size() - 1, partition, address)) sorted_ = false;
    rows_.push_back(row{partition, tuple_id});
    arena_.insert(arena_.end(), address.begin(), address.end());
  }

  [[nodiscard]] bool is_sorted() const noexcept override { return sorted_; }

  void sort() override {
    // Sort an index, then rebuild both arrays: the arena is fixed-stride, so
    // permuting it directly would need the same gather anyway.
    std::vector<std::uint64_t> order(rows_.size());
    for (std::uint64_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [this](std::uint64_t a, std::uint64_t b) {
      if (rows_[a].partition != rows_[b].partition) {
        return rows_[a].partition < rows_[b].partition;
      }
      return std::memcmp(slot(a), slot(b), width_) < 0;
    });

    std::vector<row> sorted_rows;
    std::vector<std::byte> sorted_arena;
    sorted_rows.reserve(rows_.size());
    sorted_arena.reserve(arena_.size());
    for (const std::uint64_t i : order) {
      sorted_rows.push_back(rows_[i]);
      sorted_arena.insert(sorted_arena.end(), slot(i), slot(i) + width_);
    }
    rows_.swap(sorted_rows);
    arena_.swap(sorted_arena);
    sorted_ = true;
  }

  [[nodiscard]] std::uint64_t size() const noexcept override { return rows_.size(); }
  [[nodiscard]] std::uint64_t partition_at(std::uint64_t i) const noexcept override {
    return rows_[i].partition;
  }
  /// Never meaningful here: the whole point is that the address does not fit.
  [[nodiscard]] std::uint64_t address_at(std::uint64_t) const noexcept override { return 0; }
  [[nodiscard]] std::span<const std::byte> address_bytes_at(
      std::uint64_t i) const noexcept override {
    return {slot(i), width_};
  }
  [[nodiscard]] std::uint64_t tuple_at(std::uint64_t i) const noexcept override {
    return rows_[i].tuple;
  }

 private:
  [[nodiscard]] const std::byte* slot(std::uint64_t i) const noexcept {
    return arena_.data() + i * width_;
  }
  [[nodiscard]] bool before(std::uint64_t last, std::uint64_t partition,
                            std::span<const std::byte> address) const noexcept {
    if (rows_[last].partition != partition) return rows_[last].partition < partition;
    return std::memcmp(slot(last), address.data(), width_) < 0;
  }

  struct row {
    std::uint64_t partition;
    std::uint64_t tuple;
  };
  std::vector<row> rows_;
  std::vector<std::byte> arena_;
  unsigned width_;
  bool sorted_ = true;
};

/// Staging for caller-split keys, where the parts need not share a 64-bit word.
/// 24 bytes per record, paid only by builds that actually need keys that wide.
class wide_staging final : public record_staging {
 public:
  void push(std::uint64_t partition, std::uint64_t address, std::uint64_t tuple_id) override {
    if (!rows_.empty() && !before(rows_.back().partition, rows_.back().address, partition, address)) {
      sorted_ = false;
    }
    rows_.push_back(row{partition, address, tuple_id});
  }
  [[nodiscard]] bool is_sorted() const noexcept override { return sorted_; }
  void sort() override {
    std::sort(rows_.begin(), rows_.end(), [](const row& a, const row& b) {
      return before(a.partition, a.address, b.partition, b.address);
    });
    sorted_ = true;
  }
  [[nodiscard]] std::uint64_t size() const noexcept override { return rows_.size(); }
  [[nodiscard]] std::uint64_t partition_at(std::uint64_t i) const noexcept override {
    return rows_[i].partition;
  }
  [[nodiscard]] std::uint64_t address_at(std::uint64_t i) const noexcept override {
    return rows_[i].address;
  }
  [[nodiscard]] std::uint64_t tuple_at(std::uint64_t i) const noexcept override {
    return rows_[i].tuple;
  }

 private:
  [[nodiscard]] static bool before(std::uint64_t pa, std::uint64_t aa, std::uint64_t pb,
                                   std::uint64_t ab) noexcept {
    return pa != pb ? pa < pb : aa < ab;
  }

  struct row {
    std::uint64_t partition;
    std::uint64_t address;
    std::uint64_t tuple;
  };
  std::vector<row> rows_;
  bool sorted_ = true;
};

class table_builder;

/// Scratch handle for one record's field values. Fields left unset default to 0.
/// Which key world a build is in, fixed by its first record.
enum class key_mode {
  undecided,
  keyed,          ///< begin_record(uint64), split by options.address_bits
  explicit_,      ///< begin_record_at(partition, uint64 address)
  explicit_wide,  ///< begin_record_at(partition, span) -- an address past 64 bits
};

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
  record_ref(table_builder* owner, std::uint64_t partition, std::uint64_t address)
      : owner_(owner), partition_(partition), address_(address) {}
  table_builder* owner_;
  std::uint64_t partition_;
  std::uint64_t address_;
};

class table_builder {
 public:
  explicit table_builder(schema_builder schema, build_options options = {})
      : schema_(std::move(schema)),
        options_(options),
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

  /// Begin a record keyed by a 64-bit value, split by options.address_bits.
  /// Field values are written through the returned handle and the record is not
  /// staged until commit().
  [[nodiscard]] record_ref begin_record(std::uint64_t key) {
    enter_mode(key_mode::keyed);
    std::fill(slots_.begin(), slots_.end(), std::uint64_t{0});
    return record_ref(this, partition_of(key), address_of(key));
  }

  /// Begin a record from parts the caller has already split out. This is the
  /// general entry point: the key itself can be any type and any width, because
  /// the library only ever sees the two ordering coordinates. The image records
  /// that it has no 64-bit key space, so nothing later pretends to rebuild one.
  [[nodiscard]] record_ref begin_record_at(std::uint64_t partition, std::uint64_t address) {
    enter_mode(key_mode::explicit_);
    std::fill(slots_.begin(), slots_.end(), std::uint64_t{0});
    return record_ref(this, partition, address);
  }

  /// Begin a record whose address is too wide for a scalar -- an IPv6 host part,
  /// a hash, anything. The bytes are opaque: they are stored as given and
  /// compared lexicographically, so the encoding must be order-preserving. Big
  /// endian usually is, and IPv6 network order already is.
  ///
  /// The width is fixed by the first such record, because a uniform stride is
  /// what makes the binary search possible.
  [[nodiscard]] record_ref begin_record_at(std::uint64_t partition,
                                           std::span<const std::byte> address) {
    if (address.empty()) throw build_error("a wide address must have at least one byte");
    enter_wide_mode(static_cast<unsigned>(address.size()));
    if (address.size() != wide_width_) {
      throw build_error("every address must be the same width: got " +
                        std::to_string(address.size()) + " after " +
                        std::to_string(wide_width_));
    }
    pending_address_.assign(address.begin(), address.end());
    std::fill(slots_.begin(), slots_.end(), std::uint64_t{0});
    return record_ref(this, partition, 0);
  }

  void commit(const record_ref& rec) {
    if (finished_) throw build_error("commit() after finish()");
    // Ordering is on (partition, address), which is what the image is sorted by.
    // For keyed builds that is the same comparison as on the key itself; for
    // wide addresses it is a byte comparison, matching how the reader searches.
    if (options_.order == input_order::assume_sorted && count_ != 0) {
      const bool increasing =
          last_partition_ < rec.partition_ ||
          (last_partition_ == rec.partition_ &&
           (blob_ != nullptr
                ? std::memcmp(last_wide_.data(), pending_address_.data(), wide_width_) < 0
                : last_address_ < rec.address_));
      if (!increasing) {
        throw build_error("keys must arrive strictly increasing at partition " +
                          std::to_string(rec.partition_));
      }
    }

    const std::uint64_t tuple = intern_tuple();
    if (blob_ != nullptr) {
      blob_->push_bytes(rec.partition_, pending_address_, tuple);
      last_wide_ = pending_address_;
    } else {
      staging_->push(rec.partition_, rec.address_, tuple);
    }
    last_partition_ = rec.partition_;
    last_address_ = rec.address_;
    ++count_;
  }

  /// Which key world this build is in. Undecided until the first record.
  [[nodiscard]] key_mode mode() const noexcept { return mode_; }

  /// Measure, choose the shape, and write the image. Returns what it decided.
  template <byte_sink Sink>
  build_report finish(Sink& sink) {
    if (finished_) throw build_error("finish() called twice");
    finished_ = true;

    // An empty build never entered a mode; treat it as keyed, since an empty
    // table with a key space is the less surprising of the two.
    if (!staging_) staging_ = make_staging(key_mode::keyed);

    build_report report;
    report.records = count_;
    report.distinct_values = tuple_count();
    report.dedup_abandoned = !dedup_active_;
    report.has_key_mapping = mode_ == key_mode::keyed || mode_ == key_mode::undecided;

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

  /// Fix the key world on the first record, and pick the staging that fits it.
  /// Keyed builds pack the parts back into one word; caller-split builds cannot,
  /// so they carry both.
  void enter_mode(key_mode wanted) {
    if (finished_) throw build_error("records added after finish()");
    if (mode_ == key_mode::undecided) {
      mode_ = wanted;
      if (!staging_) staging_ = make_staging(wanted);
      return;
    }
    if (mode_ != wanted) {
      throw build_error(
          "a build is either keyed or caller-split, not both: begin_record() and "
          "begin_record_at() cannot be mixed");
    }
  }

  /// As enter_mode, but also fixes the address width, which a wide build needs
  /// before it can allocate its arena.
  void enter_wide_mode(unsigned width) {
    if (finished_) throw build_error("records added after finish()");
    if (mode_ == key_mode::undecided) {
      mode_ = key_mode::explicit_wide;
      wide_width_ = width;
      if (!staging_) {
        auto owned = std::make_unique<blob_staging>(width);
        blob_ = owned.get();
        staging_ = std::move(owned);
      }
      return;
    }
    if (mode_ != key_mode::explicit_wide) {
      throw build_error(
          "a build is either keyed or caller-split, not both: begin_record() and "
          "begin_record_at() cannot be mixed");
    }
  }

  [[nodiscard]] std::unique_ptr<record_staging> make_staging(key_mode wanted) const {
    if (wanted == key_mode::explicit_) return std::make_unique<wide_staging>();
    return std::make_unique<memory_staging>(options_.address_bits);
  }

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
      const bool same_partition = staging_->partition_at(i - 1) == staging_->partition_at(i);
      const bool address_increases =
          blob_ != nullptr
              ? std::memcmp(staging_->address_bytes_at(i - 1).data(),
                            staging_->address_bytes_at(i).data(), wide_width_) < 0
              : staging_->address_at(i - 1) < staging_->address_at(i);
      const bool increasing =
          staging_->partition_at(i - 1) < staging_->partition_at(i) ||
          (same_partition && address_increases);
      if (!increasing) {
        throw build_error("duplicate key after sorting: (" +
                          std::to_string(staging_->partition_at(i)) + ", " +
                          std::to_string(staging_->address_at(i)) + ")");
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

    const std::uint64_t distinct = tuple_count();
    ref_width_ = detail::width_for(distinct == 0 ? 0 : distinct - 1);

    // One walk over the staged records: the address width comes from the data
    // rather than from address_bits, per-partition distinct counts give the
    // remap saving, and the OR of each partition's addresses gives the stretch
    // of the address field it actually needs. All three are wanted before the
    // layout can be priced.
    std::uint64_t max_address = 0;
    std::uint64_t remap_savings = 0;
    std::uint64_t address_bytes = 0;
    partition_skip_.clear();
    partition_width_.clear();
    {
      std::unordered_set<std::uint64_t> local;
      std::vector<std::byte> or_bytes(blob_ != nullptr ? wide_width_ : 0, std::byte{0});
      const std::uint64_t n = staging_->size();
      std::uint64_t i = 0;
      while (i < n) {
        const std::uint64_t partition = staging_->partition_at(i);
        local.clear();
        std::uint64_t or_bits = 0;
        std::fill(or_bytes.begin(), or_bytes.end(), std::byte{0});
        std::uint64_t j = i;
        while (j < n && staging_->partition_at(j) == partition) {
          if (blob_ != nullptr) {
            const auto a = staging_->address_bytes_at(j);
            for (unsigned b = 0; b < wide_width_; ++b) or_bytes[b] |= a[b];
          } else {
            const std::uint64_t a = staging_->address_at(j);
            max_address = std::max(max_address, a);
            or_bits |= a;
          }
          local.insert(staging_->tuple_at(j));
          ++j;
        }

        unsigned skip = 0;
        unsigned width = 0;
        if (blob_ != nullptr) {
          trim_bytes(or_bytes, skip, width);
        } else {
          trim_bits(or_bits, skip, width);
        }
        partition_skip_.push_back(static_cast<std::uint8_t>(skip));
        partition_width_.push_back(static_cast<std::uint8_t>(width));
        address_bytes += (j - i) * width;

        std::uint64_t saved = 0;
        (void)plan_remap(local.size(), j - i, saved);  // only the saving matters here
        remap_savings += saved;
        i = j;
      }
    }
    // A wide build already fixed its width; a narrow one takes the narrowest
    // that holds the largest address seen. Exact rather than snapped to
    // 1/2/4/8, because the reader loads the address through a mask where every
    // width costs the same.
    address_width_ = blob_ != nullptr ? wide_width_ : detail::exact_width_for(max_address);

    if (!options_.partition_address_width) {
      std::fill(partition_skip_.begin(), partition_skip_.end(), std::uint8_t{0});
      std::fill(partition_width_.begin(), partition_width_.end(),
                static_cast<std::uint8_t>(address_width_));
      address_bytes = count_ * address_width_;
    }
    for (std::size_t p = 0; p < partition_width_.size(); ++p) {
      if (partition_width_[p] != address_width_ || partition_skip_[p] != 0) {
        ++report.narrowed_partitions;
      }
    }
    report.address_bytes_total = address_bytes;
    report.address_saved_bytes = count_ * address_width_ - address_bytes;

    // The cost model. Dictionary bytes for text fields are the same either way,
    // so they cancel and only records plus the composite dictionary matter.
    // Remap savings belong on the interned side, or the model would price a
    // layout it is not actually going to write. The address term is the same on
    // both sides and so cancels for the decision, but it has to be right for the
    // estimates to mean anything.
    const std::uint64_t inline_bytes = address_bytes + count_ * value_stride_;
    std::uint64_t interned_bytes = address_bytes + count_ * ref_width_ +
                                   fixed_dictionary_size(distinct, value_stride_);
    interned_bytes -= std::min(interned_bytes, remap_savings);

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

  // The stretch of the address field a partition actually uses, taken from the
  // OR of every address in it: from the first nonzero byte to the last. Whatever
  // lies outside is zero in every record, so trimming it loses nothing and
  // zero-extension on read is exact.
  //
  // Which end pays off depends on the encoding, and both are worth having. A
  // little-endian integer puts a region start's low-order zeros at the *front*
  // of the field (IPv4 10.0.0.0 is 00 00 00 0A) and its small-magnitude zeros at
  // the back; a big-endian byte string is the other way round. Stating the rule
  // in byte positions rather than in significance covers both without caring.
  //
  /// All-zero means the partition holds one record whose address is zero, and
  /// nothing at all needs storing for it.
  static void trim_bits(std::uint64_t bits, unsigned& skip, unsigned& width) noexcept {
    if (bits == 0) {
      skip = 0;
      width = 0;
      return;
    }
    skip = 0;
    while ((bits & 0xff) == 0) {
      bits >>= 8;
      ++skip;
    }
    width = detail::exact_width_for(bits);
  }

  static void trim_bytes(const std::vector<std::byte>& or_bytes, unsigned& skip,
                         unsigned& width) noexcept {
    const auto n = static_cast<unsigned>(or_bytes.size());
    unsigned lo = 0;
    while (lo < n && or_bytes[lo] == std::byte{0}) ++lo;
    if (lo == n) {
      skip = 0;
      width = 0;
      return;
    }
    unsigned hi = n;
    while (hi > lo && or_bytes[hi - 1] == std::byte{0}) --hi;
    skip = lo;
    width = hi - lo;
  }

  /// Local reference width for a partition holding `distinct_here` distinct
  /// tuples across `count` records, or remap::none when a local table would not
  /// pay for itself. `savings` receives the record bytes it would save.
  ///
  /// Note the address width cancels out -- it is the same in both layouts -- so
  /// this can be priced before the address width is even known.
  [[nodiscard]] std::uint32_t plan_remap(std::uint64_t distinct_here, std::uint64_t count,
                                         std::uint64_t& savings) const {
    savings = 0;
    if (!options_.partition_remap || distinct_here == 0) return remap::none;
    const unsigned local = detail::width_for(distinct_here - 1);
    // A local index no narrower than the global one cannot repay its table.
    if (local >= ref_width_ || local > remap::max_local_width) return remap::none;
    const std::uint64_t table = distinct_here * sizeof(std::uint32_t);
    const std::uint64_t saved = count * (ref_width_ - local);
    if (saved <= table) return remap::none;
    savings = saved - table;
    return local;
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
    // Records, one partition at a time. Input is key-ordered, so a partition's
    // records are contiguous, which is what lets each one be priced and written
    // in a single pass without holding every remap table in memory at once.
    pad_to(8);
    std::vector<dir_entry> directory;
    std::vector<std::byte> record(record_stride_);
    std::vector<std::uint64_t> remap_table;                       // local -> global
    std::unordered_map<std::uint64_t, std::uint32_t> to_local;    // global -> local
    const std::uint64_t rows = staging_->size();
    std::uint64_t i = 0;
    std::size_t run = 0;  ///< indexes the per-partition ranges choose_layout derived
    while (i < rows) {
      const std::uint64_t partition = staging_->partition_at(i);
      std::uint64_t j = i;
      while (j < rows && staging_->partition_at(j) == partition) ++j;
      const std::uint64_t count = j - i;

      // This walk must visit partitions in exactly the order choose_layout did,
      // or the widths the layout was priced at would not be the ones written.
      if (run >= partition_width_.size()) throw build_error("partition ranges out of step");
      const unsigned addr_skip = partition_skip_[run];
      const unsigned addr_width = partition_width_[run];
      ++run;

      std::uint32_t schema_id = remap::none;
      remap_table.clear();
      to_local.clear();
      if (interned_) {
        // First-appearance order, so a rebuild of the same input is identical.
        for (std::uint64_t k = i; k < j; ++k) {
          const std::uint64_t global = staging_->tuple_at(k);
          if (to_local.emplace(global, static_cast<std::uint32_t>(remap_table.size())).second) {
            remap_table.push_back(global);
          }
        }
        std::uint64_t saved = 0;
        schema_id = plan_remap(remap_table.size(), count, saved);
        if (schema_id == remap::none) {
          remap_table.clear();
          to_local.clear();
        } else {
          ++report.remapped_partitions;
          report.remap_saved_bytes += saved;
        }
      }

      directory.push_back(dir_entry{partition, pos, count, schema_id,
                                    static_cast<std::uint32_t>(remap_table.size()),
                                    addr_width, addr_skip});

      for (const std::uint64_t global : remap_table) {
        const auto id = static_cast<std::uint32_t>(global);
        emit(&id, sizeof(id));
      }

      const unsigned width = schema_id != remap::none ? schema_id : ref_width_;
      const std::size_t stride = addr_width + (interned_ ? width : value_stride_);
      for (std::uint64_t k = i; k < j; ++k) {
        std::memset(record.data(), 0, stride);
        // Only the stretch this partition uses. For bytes that is a window into
        // the address; for an integer it is the value shifted down past the
        // low-order bytes that are zero throughout the partition.
        if (blob_ != nullptr) {
          const auto bytes = staging_->address_bytes_at(k);
          std::memcpy(record.data(), bytes.data() + addr_skip, addr_width);
        } else {
          const std::uint64_t address = staging_->address_at(k);
          detail::store_uint(record.data(), addr_width,
                             addr_skip >= 8 ? 0 : address >> (8 * addr_skip));
        }
        if (interned_) {
          const std::uint64_t global = staging_->tuple_at(k);
          const std::uint64_t ref = schema_id != remap::none ? to_local[global] : global;
          detail::store_uint(record.data() + addr_width, width, ref);
        } else {
          encode_tuple(staging_->tuple_at(k), record.data() + addr_width);
        }
        emit(record.data(), stride);
      }
      i = j;
    }
    if (run != partition_width_.size()) throw build_error("partition ranges out of step");

    pad_to(8);
    const std::uint64_t dir_offset = pos;
    const bool dense = use_dense(directory);

    // Narrow every directory field to what this image actually needs. A dense
    // directory drops the partition entirely: slot i describes partition i, so
    // storing it would only be a value to check against its own index.
    std::uint64_t max_partition = 0, max_offset = 0, max_count = 0, max_remap = 0, max_skip = 0;
    bool trimmed = false;
    for (const dir_entry& d : directory) {
      max_partition = std::max(max_partition, d.partition);
      max_offset = std::max(max_offset, d.offset);
      max_count = std::max(max_count, d.count);
      max_remap = std::max<std::uint64_t>(max_remap, d.remap_count);
      max_skip = std::max<std::uint64_t>(max_skip, d.address_skip);
      if (d.address_width != address_width_) trimmed = true;
    }
    // Both address fields are omitted unless some partition departs from the
    // global width, so an image with nothing to trim is exactly as wide as it
    // was before the feature existed.
    dir_ = make_dir_layout(dense ? 0u : detail::width_for(max_partition),
                           detail::width_for(max_offset), detail::width_for(max_count),
                           max_remap != 0 ? 1u : 0u,
                           max_remap != 0 ? detail::width_for(max_remap) : 0u,
                           trimmed ? 1u : 0u, max_skip != 0 ? 1u : 0u);

    std::vector<std::byte> slot(dir_.stride);
    std::uint64_t dir_count = 0;
    const auto emit_slot = [&](const dir_entry& d) {
      encode_dir_entry(slot.data(), dir_, d);
      emit(slot.data(), slot.size());
      ++dir_count;
    };
    if (dense) {
      std::uint64_t next = 0;
      for (const dir_entry& d : directory) {
        // A gap slot holds no records, but its address range still has to be a
        // legal one so the invariant holds for every slot without exception.
        for (; next < d.partition; ++next) {
          emit_slot(dir_entry{next, 0, 0, 0, 0, address_width_, 0});
        }
        emit_slot(d);
        next = d.partition + 1;
      }
    } else {
      for (const dir_entry& d : directory) emit_slot(d);
    }
    report.directory_stride = dir_.stride;

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
    // A caller-split build has no 64-bit key space to describe, and guessing a
    // shift from the observed addresses would produce keys that are ordered
    // correctly but are not the caller's.
    head.address_bits = (mode_ == key_mode::explicit_ || mode_ == key_mode::explicit_wide)
                            ? no_key_mapping
                            : static_cast<std::uint8_t>(options_.address_bits);
    head.mode = static_cast<std::uint8_t>(interned_ ? value_mode::interned
                                                    : value_mode::inline_fields);
    head.ref_width = static_cast<std::uint8_t>(interned_ ? ref_width_ : 0);
    head.value_dict = interned_ ? value_dict_index : 0;
    head.dir_partition_width = static_cast<std::uint8_t>(dir_.partition_width);
    head.dir_offset_width = static_cast<std::uint8_t>(dir_.offset_width);
    head.dir_count_width = static_cast<std::uint8_t>(dir_.count_width);
    head.dir_remap_width = static_cast<std::uint8_t>(dir_.remap_width);
    head.dir_addr_width = static_cast<std::uint8_t>(dir_.addr_width);
    head.dir_skip_width = static_cast<std::uint8_t>(dir_.skip_width);

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
  dir_layout dir_;  ///< settled once the directory's value ranges are known
  /// The address range each partition stores, in partition-run order. Derived in
  /// choose_layout and indexed again by the writer, so that the layout priced
  /// and the layout written cannot drift apart.
  std::vector<std::uint8_t> partition_skip_;
  std::vector<std::uint8_t> partition_width_;

  std::uint64_t count_ = 0;
  std::uint64_t last_partition_ = 0;
  std::uint64_t last_address_ = 0;
  key_mode mode_ = key_mode::undecided;
  blob_staging* blob_ = nullptr;  ///< non-null exactly in explicit_wide mode
  unsigned wide_width_ = 0;
  std::vector<std::byte> pending_address_;  ///< the open record's wide address
  std::vector<std::byte> last_wide_;        ///< for the ordering check
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
