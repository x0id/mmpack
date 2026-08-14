#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "mmpack/detail/bits.hpp"
#include "mmpack/dictionary.hpp"
#include "mmpack/error.hpp"
#include "mmpack/format.hpp"
#include "mmpack/schema.hpp"

namespace mmpack {

/// Read-only view over an mmpack image, typically a memory-mapped file.
///
/// The table never owns the mapped bytes and never copies them. It does keep a
/// small amount of open-time state -- the validated schema and one view per
/// dictionary -- so that a field read stays an offset add plus a width-dispatched
/// load rather than a re-parse. The caller keeps the mapping alive for as long as
/// the table and any iterators derived from it are used.
///
/// Ordering follows std::map: elements are sorted by (partition, address), which
/// is key order given the shift/mask split recorded in the schema.
class table {
 public:
  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = std::uint64_t;  // the key
    using difference_type = std::ptrdiff_t;

    const_iterator() = default;

    [[nodiscard]] std::uint64_t key() const { return owner_->key_at(slot_, index_); }
    [[nodiscard]] std::uint64_t partition() const { return owner_->slot_partition(slot_); }
    [[nodiscard]] std::uint64_t address() const { return owner_->address_at(slot_, index_); }
    [[nodiscard]] std::uint64_t operator*() const { return key(); }

    const_iterator& operator++() {
      ++index_;
      owner_->normalize(slot_, index_);
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++*this;
      return tmp;
    }
    const_iterator& operator--() {
      owner_->retreat(slot_, index_);
      return *this;
    }
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --*this;
      return tmp;
    }

    [[nodiscard]] friend bool operator==(const const_iterator& a, const const_iterator& b) {
      return a.slot_ == b.slot_ && a.index_ == b.index_;
    }
    [[nodiscard]] friend bool operator!=(const const_iterator& a, const const_iterator& b) {
      return !(a == b);
    }

   private:
    friend class table;
    const_iterator(const table* owner, std::uint64_t slot, std::uint64_t index)
        : owner_(owner), slot_(slot), index_(index) {}
    const table* owner_ = nullptr;
    std::uint64_t slot_ = 0;
    std::uint64_t index_ = 0;
  };

  using iterator = const_iterator;

  table() = default;

  /// Validate and attach to a mapped image. `length` must be the exact image
  /// size, because the footer is located from the end.
  ///
  /// A successful open proves the schema self-consistent, every directory slot
  /// in bounds, and every dictionary well formed. Only then are the accessors
  /// built, so no later field read can leave the region whatever the file
  /// contains.
  [[nodiscard]] static std::optional<table> try_open(const void* base, std::size_t length,
                                                     status* out = nullptr) {
    const auto fail = [&](status s) {
      if (out) *out = s;
      return std::optional<table>{};
    };
    if (out) *out = status::ok;

    if (base == nullptr || length < sizeof(header) + sizeof(footer)) return fail(status::too_small);
    const auto* bytes = static_cast<const std::byte*>(base);

    const header head = detail::load<header>(bytes);
    if (std::memcmp(head.magic, magic, sizeof(magic)) != 0) return fail(status::bad_magic);
    if (head.format_version != format_version) return fail(status::bad_version);

    // The header already proved this is an mmpack image, so a footer that does
    // not land on the magic means `length` is not the exact image size.
    const footer foot = detail::load<footer>(bytes + length - sizeof(footer));
    if (std::memcmp(foot.magic, magic, sizeof(magic)) != 0) return fail(status::bad_size);
    if (foot.format_version != format_version) return fail(status::bad_version);
    if (foot.image_size != length) return fail(status::bad_size);

    const std::uint64_t tail = length - sizeof(footer);

    // --- schema ---
    if (foot.schema_offset > tail || foot.schema_size > tail - foot.schema_offset) {
      return fail(status::bad_schema);
    }
    auto schema = schema_view::open(
        std::span<const std::byte>(bytes + foot.schema_offset, foot.schema_size));
    if (!schema) return fail(status::bad_schema);
    if (schema->record_stride() == 0) return fail(status::bad_schema);

    table t;
    t.base_ = bytes;
    t.size_ = length;
    t.schema_ = *schema;
    t.record_count_ = foot.record_count;
    t.dense_ = (foot.flags & flags::dense_directory) != 0;
    t.interned_ = schema->interned();
    t.record_stride_ = schema->record_stride();
    t.address_width_ = schema->address_width();
    t.ref_width_ = schema->ref_width();

    // --- directory ---
    if (foot.dir_offset % alignof(dir_entry) != 0 || foot.dir_offset < sizeof(header) ||
        foot.dir_offset > tail) {
      return fail(status::bad_directory);
    }
    if (foot.dir_count > (tail - foot.dir_offset) / sizeof(dir_entry)) {
      return fail(status::bad_directory);
    }
    t.directory_ = bytes + foot.dir_offset;
    t.dir_count_ = foot.dir_count;

    std::uint64_t total = 0;
    std::uint64_t previous = 0;
    std::uint64_t floor = sizeof(header);
    for (std::uint64_t i = 0; i < foot.dir_count; ++i) {
      const dir_entry d = t.slot(i);
      if (i > 0 && d.partition <= previous) return fail(status::bad_directory);
      if (t.dense_ && d.partition != i) return fail(status::bad_directory);
      if (d.count != 0) {
        if (d.offset < floor || d.offset > foot.dir_offset) return fail(status::bad_directory);
        if (d.count > (foot.dir_offset - d.offset) / t.record_stride_) {
          return fail(status::bad_directory);
        }
        floor = d.offset + d.count * t.record_stride_;
      }
      total += d.count;
      previous = d.partition;
    }
    if (total != foot.record_count) return fail(status::bad_directory);

    // --- dictionaries ---
    t.blob_dicts_.assign(schema->dict_count(), blob_dictionary_view{});
    for (std::uint32_t i = 0; i < schema->dict_count(); ++i) {
      const dict_entry e = schema->dictionary(i);
      if (e.offset < sizeof(header) || e.offset > tail || e.size > tail - e.offset) {
        return fail(status::bad_dictionary);
      }
      const std::span<const std::byte> block(bytes + e.offset, e.size);
      if (schema->interned() && i == schema->value_dict()) {
        auto fixed = fixed_dictionary_view::open(block);
        if (!fixed) return fail(status::bad_dictionary);
        // The composite dictionary's element size must be the value stride, or
        // every field offset would point somewhere else entirely.
        if (fixed->elem_size() != schema->value_stride()) return fail(status::bad_dictionary);
        t.value_dict_ = *fixed;
      } else {
        auto blob = blob_dictionary_view::open(block);
        if (!blob) return fail(status::bad_dictionary);
        t.blob_dicts_[i] = *blob;
      }
    }
    // Every text field must point at a blob dictionary, not the composite one.
    for (std::uint32_t i = 0; i < schema->field_count(); ++i) {
      const field_desc f = schema->field(i);
      if (static_cast<field_kind>(f.kind) != field_kind::text) continue;
      if (schema->interned() && f.dict_index == schema->value_dict()) {
        return fail(status::bad_dictionary);
      }
    }

    return std::optional<table>(std::move(t));
  }

  [[nodiscard]] static table open(const void* base, std::size_t length) {
    status s = status::ok;
    auto t = try_open(base, length, &s);
    if (!t) throw format_error(s);
    return std::move(*t);
  }

  [[nodiscard]] std::uint64_t size() const noexcept { return record_count_; }
  [[nodiscard]] bool empty() const noexcept { return record_count_ == 0; }
  [[nodiscard]] std::uint64_t image_size() const noexcept { return size_; }
  [[nodiscard]] std::uint64_t directory_slots() const noexcept { return dir_count_; }
  [[nodiscard]] bool has_dense_directory() const noexcept { return dense_; }
  [[nodiscard]] bool interned() const noexcept { return interned_; }
  [[nodiscard]] const schema_view& schema() const noexcept { return schema_; }

  [[nodiscard]] std::uint64_t partition_count() const noexcept {
    std::uint64_t n = 0;
    for (std::uint64_t i = 0; i < dir_count_; ++i) {
      if (slot_count(i) != 0) ++n;
    }
    return n;
  }

  [[nodiscard]] const_iterator begin() const { return make_iterator(0, 0); }
  [[nodiscard]] const_iterator end() const { return const_iterator(this, dir_count_, 0); }

  // --- lookup ---------------------------------------------------------------

  [[nodiscard]] const_iterator lower_bound(std::uint64_t key) const {
    return lower_bound_at(schema_.partition_of(key), schema_.address_of(key));
  }
  [[nodiscard]] const_iterator upper_bound(std::uint64_t key) const {
    return upper_bound_at(schema_.partition_of(key), schema_.address_of(key));
  }
  [[nodiscard]] const_iterator find(std::uint64_t key) const {
    return find_at(schema_.partition_of(key), schema_.address_of(key));
  }
  [[nodiscard]] bool contains(std::uint64_t key) const { return find(key) != end(); }

  /// First element at or after (partition, address), crossing into later
  /// partitions when the address runs past the end of its own.
  [[nodiscard]] const_iterator lower_bound_at(std::uint64_t partition,
                                              std::uint64_t address) const {
    std::uint64_t slot = slot_lower_bound(partition);
    std::uint64_t index = 0;
    if (slot < dir_count_ && slot_partition(slot) == partition) index = search_lower(slot, address);
    normalize(slot, index);
    return const_iterator(this, slot, index);
  }

  [[nodiscard]] const_iterator upper_bound_at(std::uint64_t partition,
                                              std::uint64_t address) const {
    std::uint64_t slot = slot_lower_bound(partition);
    std::uint64_t index = 0;
    if (slot < dir_count_ && slot_partition(slot) == partition) index = search_upper(slot, address);
    normalize(slot, index);
    return const_iterator(this, slot, index);
  }

  /// Greatest element with key <= target, or end() if every key is above it.
  ///
  /// This is the range-containment primitive: given boundary entries, floor()
  /// names the range covering a point. Doing it with upper_bound() plus a
  /// decrement works but puts a begin() check on every caller, and the check is
  /// easy to forget or get subtly wrong.
  [[nodiscard]] const_iterator floor(std::uint64_t key) const {
    return floor_at(schema_.partition_of(key), schema_.address_of(key));
  }

  /// Least element with key >= target, or end(). Identical to lower_bound();
  /// named for symmetry with floor() so containment code reads consistently.
  [[nodiscard]] const_iterator ceil(std::uint64_t key) const { return lower_bound(key); }

  /// As floor(), for callers that already hold the split key.
  [[nodiscard]] const_iterator floor_at(std::uint64_t partition, std::uint64_t address) const {
    const std::uint64_t slot = slot_floor(partition);
    if (slot >= dir_count_) return end();  // every partition is above the target

    if (slot_partition(slot) == partition) {
      const std::uint64_t count = slot_count(slot);
      if (count != 0) {
        const std::uint64_t index = search_floor(slot, address);
        if (index < count) return const_iterator(this, slot, index);
      }
      // The partition exists but holds nothing at or below `address`, so the
      // answer is the tail of an earlier partition.
      return floor_before(slot);
    }
    // slot_partition(slot) < partition: every element here precedes the target,
    // so this partition's own tail is the answer -- if it has one.
    return floor_at_or_before(slot);
  }

  [[nodiscard]] const_iterator ceil_at(std::uint64_t partition, std::uint64_t address) const {
    return lower_bound_at(partition, address);
  }

  [[nodiscard]] const_iterator find_at(std::uint64_t partition, std::uint64_t address) const {
    const std::uint64_t slot = slot_exact(partition);
    if (slot >= dir_count_) return end();
    const std::uint64_t count = slot_count(slot);
    if (count == 0) return end();
    const std::uint64_t index = search_lower(slot, address);
    if (index >= count || address_at(slot, index) != address) return end();
    return const_iterator(this, slot, index);
  }

  // --- fields ---------------------------------------------------------------

  [[nodiscard]] std::optional<field_id> field(std::string_view name) const {
    return schema_.find(name);
  }

  [[nodiscard]] std::optional<std::uint64_t> uint(const const_iterator& it, field_id id) const {
    return read_scalar(it, id, field_kind::uint_);
  }

  [[nodiscard]] std::optional<std::int64_t> sint(const const_iterator& it, field_id id) const {
    const auto raw = read_scalar(it, id, field_kind::sint_);
    if (!raw) return std::nullopt;
    return static_cast<std::int64_t>(*raw);
  }

  [[nodiscard]] std::optional<float> f32(const const_iterator& it, field_id id) const {
    const auto raw = read_scalar(it, id, field_kind::f32);
    if (!raw) return std::nullopt;
    float v;
    const auto bits = static_cast<std::uint32_t>(*raw);
    std::memcpy(&v, &bits, sizeof(v));
    return v;
  }

  [[nodiscard]] std::optional<double> f64(const const_iterator& it, field_id id) const {
    const auto raw = read_scalar(it, id, field_kind::f64);
    if (!raw) return std::nullopt;
    double v;
    const std::uint64_t bits = *raw;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
  }

  /// Raw bytes of a `bytes` field, pointing into the image.
  [[nodiscard]] std::optional<std::span<const std::byte>> bytes(const const_iterator& it,
                                                                field_id id) const {
    if (id >= schema_.field_count()) return std::nullopt;
    const field_desc d = schema_.field(id);
    if (static_cast<field_kind>(d.kind) != field_kind::bytes) return std::nullopt;
    const std::byte* base = value_base(it);
    if (base == nullptr) return std::nullopt;
    return std::span<const std::byte>(base + d.offset, d.width);
  }

  /// Text of an interned field, resolved through its dictionary.
  [[nodiscard]] std::optional<std::string_view> text(const const_iterator& it, field_id id) const {
    const auto raw = read_scalar(it, id, field_kind::text);
    if (!raw) return std::nullopt;
    const field_desc d = schema_.field(id);
    if (d.dict_index >= blob_dicts_.size()) return std::nullopt;
    return blob_dicts_[d.dict_index].text(*raw);
  }

 private:
  friend class const_iterator;

  [[nodiscard]] dir_entry slot(std::uint64_t i) const {
    return detail::load<dir_entry>(directory_ + i * sizeof(dir_entry));
  }
  [[nodiscard]] std::uint64_t slot_partition(std::uint64_t i) const {
    return detail::load<std::uint64_t>(directory_ + i * sizeof(dir_entry) +
                                       offsetof(dir_entry, partition));
  }
  [[nodiscard]] std::uint64_t slot_offset(std::uint64_t i) const {
    return detail::load<std::uint64_t>(directory_ + i * sizeof(dir_entry) +
                                       offsetof(dir_entry, offset));
  }
  [[nodiscard]] std::uint64_t slot_count(std::uint64_t i) const {
    return detail::load<std::uint64_t>(directory_ + i * sizeof(dir_entry) +
                                       offsetof(dir_entry, count));
  }

  [[nodiscard]] const std::byte* record(std::uint64_t s, std::uint64_t i) const {
    return base_ + slot_offset(s) + i * record_stride_;
  }
  [[nodiscard]] std::uint64_t address_at(std::uint64_t s, std::uint64_t i) const {
    return detail::load_uint(record(s, i), address_width_);
  }
  [[nodiscard]] std::uint64_t key_at(std::uint64_t s, std::uint64_t i) const {
    const std::uint64_t bits = schema_.address_bits();
    const std::uint64_t partition = slot_partition(s);
    return bits >= 64 ? address_at(s, i) : (partition << bits) | address_at(s, i);
  }

  /// Base of the value tuple for an element, or nullptr if the record's
  /// reference into the composite dictionary is out of range. Returning nullptr
  /// is what keeps a corrupt reference from becoming a wild read.
  [[nodiscard]] const std::byte* value_base(const const_iterator& it) const {
    if (it.slot_ >= dir_count_) return nullptr;
    const std::byte* r = record(it.slot_, it.index_);
    if (!interned_) return r + address_width_;
    const std::uint64_t ref = detail::load_uint(r + address_width_, ref_width_);
    return value_dict_.element(ref);
  }

  [[nodiscard]] std::optional<std::uint64_t> read_scalar(const const_iterator& it, field_id id,
                                                         field_kind expected) const {
    if (id >= schema_.field_count()) return std::nullopt;
    const field_desc d = schema_.field(id);
    if (static_cast<field_kind>(d.kind) != expected) return std::nullopt;
    const std::byte* base = value_base(it);
    if (base == nullptr) return std::nullopt;
    return detail::load_uint(base + d.offset, d.width) + d.bias;
  }

  void normalize(std::uint64_t& s, std::uint64_t& i) const {
    while (s < dir_count_ && i >= slot_count(s)) {
      ++s;
      i = 0;
    }
    if (s >= dir_count_) {
      s = dir_count_;
      i = 0;
    }
  }

  void retreat(std::uint64_t& s, std::uint64_t& i) const {
    if (i > 0) {
      --i;
      return;
    }
    while (s > 0) {
      --s;
      const std::uint64_t c = slot_count(s);
      if (c != 0) {
        i = c - 1;
        return;
      }
    }
  }

  [[nodiscard]] const_iterator make_iterator(std::uint64_t s, std::uint64_t i) const {
    normalize(s, i);
    return const_iterator(this, s, i);
  }

  [[nodiscard]] std::uint64_t slot_lower_bound(std::uint64_t p) const {
    if (dense_) return p < dir_count_ ? p : dir_count_;
    std::uint64_t lo = 0;
    std::uint64_t len = dir_count_;
    while (len > 0) {
      const std::uint64_t half = len / 2;
      const std::uint64_t mid = lo + half;
      if (slot_partition(mid) < p) {
        lo = mid + 1;
        len -= half + 1;
      } else {
        len = half;
      }
    }
    return lo;
  }

  [[nodiscard]] std::uint64_t slot_exact(std::uint64_t p) const {
    const std::uint64_t s = slot_lower_bound(p);
    if (s < dir_count_ && slot_partition(s) == p) return s;
    return dir_count_;
  }

  /// Last directory slot whose partition is <= p, or dir_count_ if every
  /// partition is above p. Derived by stepping back from "first slot above p"
  /// rather than computing p + 1, which would overflow at address_bits == 0.
  [[nodiscard]] std::uint64_t slot_floor(std::uint64_t p) const {
    if (dir_count_ == 0) return 0;  // equals dir_count_: the "none" sentinel
    if (dense_) return p < dir_count_ ? p : dir_count_ - 1;
    std::uint64_t lo = 0;
    std::uint64_t len = dir_count_;
    while (len > 0) {
      const std::uint64_t half = len / 2;
      const std::uint64_t mid = lo + half;
      if (slot_partition(mid) <= p) {
        lo = mid + 1;
        len -= half + 1;
      } else {
        len = half;
      }
    }
    return lo == 0 ? dir_count_ : lo - 1;
  }

  /// Last element of the greatest non-empty slot at or before `s`, or end().
  [[nodiscard]] const_iterator floor_at_or_before(std::uint64_t s) const {
    while (true) {
      const std::uint64_t count = slot_count(s);
      if (count != 0) return const_iterator(this, s, count - 1);
      if (s == 0) return end();
      --s;
    }
  }

  /// Same, but strictly before `s` -- used when slot `s` is the target
  /// partition and has already been shown to hold nothing low enough.
  [[nodiscard]] const_iterator floor_before(std::uint64_t s) const {
    if (s == 0) return end();
    return floor_at_or_before(s - 1);
  }

  [[nodiscard]] std::uint64_t search_lower(std::uint64_t s, std::uint64_t address) const {
    const std::byte* first = base_ + slot_offset(s);
    std::uint64_t lo = 0;
    std::uint64_t len = slot_count(s);
    while (len > 0) {
      const std::uint64_t half = len / 2;
      const std::uint64_t mid = lo + half;
      if (detail::load_uint(first + mid * record_stride_, address_width_) < address) {
        lo = mid + 1;
        len -= half + 1;
      } else {
        len = half;
      }
    }
    return lo;
  }

  /// Index of the largest address <= target within a slot, or slot_count(s) as
  /// a "none here" sentinel. Structurally this is search_upper with the result
  /// stepped back one, so the scan itself is unchanged -- the win over calling
  /// upper_bound() and decrementing is that no iterator is ever positioned past
  /// the partition and then walked back.
  [[nodiscard]] std::uint64_t search_floor(std::uint64_t s, std::uint64_t address) const {
    const std::uint64_t above = search_upper(s, address);
    return above == 0 ? slot_count(s) : above - 1;
  }

  [[nodiscard]] std::uint64_t search_upper(std::uint64_t s, std::uint64_t address) const {
    const std::byte* first = base_ + slot_offset(s);
    std::uint64_t lo = 0;
    std::uint64_t len = slot_count(s);
    while (len > 0) {
      const std::uint64_t half = len / 2;
      const std::uint64_t mid = lo + half;
      if (!(address < detail::load_uint(first + mid * record_stride_, address_width_))) {
        lo = mid + 1;
        len -= half + 1;
      } else {
        len = half;
      }
    }
    return lo;
  }

  const std::byte* base_ = nullptr;
  std::uint64_t size_ = 0;
  schema_view schema_;
  const std::byte* directory_ = nullptr;
  std::uint64_t dir_count_ = 0;
  std::uint64_t record_count_ = 0;
  std::vector<blob_dictionary_view> blob_dicts_;
  fixed_dictionary_view value_dict_;
  std::uint32_t record_stride_ = 0;
  unsigned address_width_ = 1;
  unsigned ref_width_ = 1;
  bool dense_ = false;
  bool interned_ = false;
};

}  // namespace mmpack
