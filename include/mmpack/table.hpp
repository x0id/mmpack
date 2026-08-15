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
  /// A positional cursor, not a dereferenceable iterator: an element is a
  /// partition plus an address that may be wider than any scalar, so there is
  /// nothing honest for operator* to return. Read the parts through the
  /// accessors instead.
  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;

    const_iterator() = default;

    [[nodiscard]] std::uint64_t partition() const { return owner_->slot_partition(slot_); }

    /// The address as an integer, when it fits one. Empty on images whose
    /// addresses are wider than 64 bits, where address_bytes() is the answer.
    [[nodiscard]] std::optional<std::uint64_t> address() const {
      if (!owner_->narrow_address()) return std::nullopt;
      return owner_->address_at(slot_, index_);
    }

    /// The bytes this element's partition actually stores -- the significant
    /// stretch of the address, which may be narrower than table::address_width()
    /// because the partition trimmed zero bytes off either end. They belong at
    /// offset address_skip() within the full field; everything else there is
    /// zero. Use address_into() to get the whole field back in one step.
    [[nodiscard]] std::span<const std::byte> address_bytes() const {
      return owner_->address_bytes_at(slot_, index_);
    }

    /// Where address_bytes() sits within the full address field.
    [[nodiscard]] unsigned address_skip() const { return owner_->address_skip_at(slot_); }

    /// Rebuild the full address field, zero-filling whatever was trimmed. `out`
    /// must be exactly table::address_width() bytes; returns false otherwise.
    [[nodiscard]] bool address_into(std::span<std::byte> out) const {
      return owner_->address_into_at(slot_, index_, out);
    }

    /// The 64-bit key, when the image defines one. Empty for images built from
    /// caller-split parts: the key type and its encoding are the caller's, and
    /// the library has no way to reproduce them. Reconstruct those through your
    /// own join -- see keyed_table in mmpack/keyed.hpp.
    [[nodiscard]] std::optional<std::uint64_t> key() const {
      if (!owner_->has_key_mapping()) return std::nullopt;
      return owner_->key_at(slot_, index_);
    }

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
    // A 64-bit key is a packing of the partition and the address, so a key
    // mapping cannot coexist with an address too wide to pack. Rejecting the
    // combination keeps key_at() off a path it could not compute.
    if (schema->has_key_mapping() && !schema->narrow_address()) return fail(status::bad_schema);

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
    // What follows the address in a record. Held separately because the address
    // part now varies per partition while this does not, so a partition's stride
    // is its own address width plus this.
    if (t.record_stride_ < t.address_width_) return fail(status::bad_schema);
    t.value_part_ = t.record_stride_ - t.address_width_;

    // --- directory ---
    t.dir_ = schema->directory_layout();
    if (t.dir_.stride == 0) return fail(status::bad_directory);
    // A dense directory is exactly the case where the partition is implicit, so
    // the two encodings must not disagree about which one this is.
    if ((t.dir_.partition_width == 0) != t.dense_) return fail(status::bad_directory);
    if (foot.dir_offset < sizeof(header) || foot.dir_offset > tail) {
      return fail(status::bad_directory);
    }
    if (foot.dir_count > (tail - foot.dir_offset) / t.dir_.stride) {
      return fail(status::bad_directory);
    }
    // Directory fields are read with a masked 8-byte load, so the slack past the
    // last slot has to be real. The footer alone guarantees it, but stating the
    // requirement here keeps it from quietly lapsing if the layout moves.
    if (foot.dir_offset + foot.dir_count * t.dir_.stride + sizeof(std::uint64_t) > length) {
      return fail(status::bad_directory);
    }
    t.directory_ = bytes + foot.dir_offset;
    t.dir_count_ = foot.dir_count;

    std::uint64_t total = 0;
    std::uint64_t previous = 0;
    std::uint64_t floor = sizeof(header);
    for (std::uint64_t i = 0; i < foot.dir_count; ++i) {
      const dir_entry d = t.slot(i);
      // Density no longer needs checking: a dense directory does not store the
      // partition at all, so slot i describes partition i by construction.
      if (i > 0 && d.partition <= previous) return fail(status::bad_directory);

      // The stretch of the address field this partition stores must lie inside
      // that field. Written as a subtraction against the width already shown to
      // be the smaller, so the sum cannot overflow.
      if (d.address_width > t.address_width_ ||
          d.address_skip > t.address_width_ - d.address_width) {
        return fail(status::bad_directory);
      }

      // A remapped partition carries its own reference width and lookup table,
      // so its geometry has to be validated before it can be trusted to size
      // the record block.
      std::uint64_t stride = d.address_width + t.value_part_;
      if (d.schema_id != remap::none) {
        if (d.schema_id > remap::max_local_width) return fail(status::bad_directory);
        if (!t.interned_) return fail(status::bad_directory);  // nothing to remap into
        if (d.remap_count == 0 || d.count == 0) return fail(status::bad_directory);
        stride = d.address_width + d.schema_id;
      } else if (d.remap_count != 0) {
        return fail(status::bad_directory);
      }
      // A zero stride would make every record start at the same byte, and the
      // record-count check below divide by zero. Reachable only from a corrupt
      // image: an address trimmed to nothing alongside an empty value tuple.
      if (stride == 0) return fail(status::bad_directory);

      if (d.count != 0) {
        if (d.offset < floor || d.offset > foot.dir_offset) return fail(status::bad_directory);
        const std::uint64_t available = foot.dir_offset - d.offset;
        const std::uint64_t table_bytes =
            static_cast<std::uint64_t>(d.remap_count) * sizeof(std::uint32_t);
        if (table_bytes > available) return fail(status::bad_directory);
        if (d.count > (available - table_bytes) / stride) return fail(status::bad_directory);
        floor = d.offset + table_bytes + d.count * stride;
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

  /// Whether this image defines a 64-bit key space. False when it was built from
  /// caller-split parts, in which case the key-taking lookups below have nothing
  /// to search and the *_at forms are the interface.
  [[nodiscard]] bool has_key_mapping() const noexcept { return schema_.has_key_mapping(); }

  /// Whether addresses fit an integer. When false they are opaque byte strings
  /// compared lexicographically, and the span-taking *_at forms are the way in.
  [[nodiscard]] bool narrow_address() const noexcept { return schema_.narrow_address(); }
  /// The full address field, which every query is given at. What an individual
  /// record stores may be narrower -- see iterator::address_bytes().
  [[nodiscard]] unsigned address_width() const noexcept { return address_width_; }
  /// Whether any partition stores less than the full field.
  [[nodiscard]] bool trims_addresses() const noexcept { return schema_.trims_addresses(); }

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

  // The key-taking forms are sugar over the *_at ones for images that define a
  // 64-bit key space. On an image built from caller-split parts there is no such
  // space, so rather than search a key the caller never wrote, they find nothing;
  // has_key_mapping() distinguishes that from a genuine miss.
  [[nodiscard]] const_iterator lower_bound(std::uint64_t key) const {
    if (!has_key_mapping()) return end();
    return lower_bound_at(schema_.partition_of(key), schema_.address_of(key));
  }
  [[nodiscard]] const_iterator upper_bound(std::uint64_t key) const {
    if (!has_key_mapping()) return end();
    return upper_bound_at(schema_.partition_of(key), schema_.address_of(key));
  }
  [[nodiscard]] const_iterator find(std::uint64_t key) const {
    if (!has_key_mapping()) return end();
    return find_at(schema_.partition_of(key), schema_.address_of(key));
  }
  [[nodiscard]] bool contains(std::uint64_t key) const { return find(key) != end(); }

  // Each *_at form comes in two shapes. The integer one serves images whose
  // address fits a scalar and finds nothing on wider ones, since the probe is
  // not in their address space at all. The byte-span one serves both, and
  // rejects a probe whose length is not this image's address width rather than
  // comparing against a different number of bytes than it was given.

  /// First element at or after (partition, address), crossing into later
  /// partitions when the address runs past the end of its own.
  [[nodiscard]] const_iterator lower_bound_at(std::uint64_t partition,
                                              std::uint64_t address) const {
    if (!narrow_address()) return end();
    return lower_bound_parts<true>(partition, address, nullptr);
  }
  [[nodiscard]] const_iterator lower_bound_at(std::uint64_t partition,
                                              std::span<const std::byte> address) const {
    if (address.size() != address_width_) return end();
    if (narrow_address()) {
      return lower_bound_parts<true>(partition, detail::load_uint(address.data(), address_width_),
                                     nullptr);
    }
    return lower_bound_parts<false>(partition, 0, address.data());
  }

  [[nodiscard]] const_iterator upper_bound_at(std::uint64_t partition,
                                              std::uint64_t address) const {
    if (!narrow_address()) return end();
    return upper_bound_parts<true>(partition, address, nullptr);
  }
  [[nodiscard]] const_iterator upper_bound_at(std::uint64_t partition,
                                              std::span<const std::byte> address) const {
    if (address.size() != address_width_) return end();
    if (narrow_address()) {
      return upper_bound_parts<true>(partition, detail::load_uint(address.data(), address_width_),
                                     nullptr);
    }
    return upper_bound_parts<false>(partition, 0, address.data());
  }

  /// Greatest element with key <= target, or end() if every key is above it.
  ///
  /// This is the range-containment primitive: given boundary entries, floor()
  /// names the range covering a point. Doing it with upper_bound() plus a
  /// decrement works but puts a begin() check on every caller, and the check is
  /// easy to forget or get subtly wrong.
  [[nodiscard]] const_iterator floor(std::uint64_t key) const {
    if (!has_key_mapping()) return end();
    return floor_at(schema_.partition_of(key), schema_.address_of(key));
  }

  /// Least element with key >= target, or end(). Identical to lower_bound();
  /// named for symmetry with floor() so containment code reads consistently.
  [[nodiscard]] const_iterator ceil(std::uint64_t key) const { return lower_bound(key); }

  /// As floor(), for callers that already hold the split key.
  [[nodiscard]] const_iterator floor_at(std::uint64_t partition, std::uint64_t address) const {
    if (!narrow_address()) return end();
    return floor_parts<true>(partition, address, nullptr);
  }
  [[nodiscard]] const_iterator floor_at(std::uint64_t partition,
                                        std::span<const std::byte> address) const {
    if (address.size() != address_width_) return end();
    if (narrow_address()) {
      return floor_parts<true>(partition, detail::load_uint(address.data(), address_width_), nullptr);
    }
    return floor_parts<false>(partition, 0, address.data());
  }

  [[nodiscard]] const_iterator ceil_at(std::uint64_t partition, std::uint64_t address) const {
    return lower_bound_at(partition, address);
  }
  [[nodiscard]] const_iterator ceil_at(std::uint64_t partition,
                                       std::span<const std::byte> address) const {
    return lower_bound_at(partition, address);
  }

  [[nodiscard]] const_iterator find_at(std::uint64_t partition, std::uint64_t address) const {
    if (!narrow_address()) return end();
    return find_parts<true>(partition, address, nullptr);
  }
  [[nodiscard]] const_iterator find_at(std::uint64_t partition,
                                       std::span<const std::byte> address) const {
    if (address.size() != address_width_) return end();
    if (narrow_address()) {
      return find_parts<true>(partition, detail::load_uint(address.data(), address_width_), nullptr);
    }
    return find_parts<false>(partition, 0, address.data());
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

  // Templated on the regime so that each public entry point resolves it once at
  // the boundary and everything below is monomorphic. Dispatching lower down
  // instead -- even with the branch outside the loop -- measured ~3% slower on
  // the narrow path, because it left a runtime choice between the entry and the
  // search that the compiler could no longer see through.
  template <bool Narrow>
  [[nodiscard]] const_iterator lower_bound_parts(std::uint64_t partition, std::uint64_t value,
                                                 const std::byte* bytes) const {
    std::uint64_t slot = slot_lower_bound(partition);
    std::uint64_t index = 0;
    if (slot < dir_count_ && slot_partition(slot) == partition) {
      const partition_layout p = layout_of(slot);
      // One call, not a choice between two: instantiating both loop bodies here
      // measured ~3% on the narrow path, the same code-size effect that made
      // templating the regime expensive in e9a2744. `tail` is folded into the
      // operand instead -- see narrow_probe and search_wide.
      if constexpr (Narrow) {
        const narrow_probe q = probe_narrow(p, value);
        index = search_lower_narrow(p, q.value + (q.tail ? 1 : 0));
      } else {
        const wide_probe q = probe_wide(p, bytes);
        index = q.head ? p.count : search_wide(p, q.bytes, q.tail ? 1 : 0);
      }
    }
    normalize(slot, index);
    return const_iterator(this, slot, index);
  }

  template <bool Narrow>
  [[nodiscard]] const_iterator upper_bound_parts(std::uint64_t partition, std::uint64_t value,
                                                 const std::byte* bytes) const {
    std::uint64_t slot = slot_lower_bound(partition);
    std::uint64_t index = 0;
    if (slot < dir_count_ && slot_partition(slot) == partition) {
      const partition_layout p = layout_of(slot);
      index = search_above<Narrow>(p, value, bytes);
    }
    normalize(slot, index);
    return const_iterator(this, slot, index);
  }

  template <bool Narrow>
  [[nodiscard]] const_iterator floor_parts(std::uint64_t partition, std::uint64_t value,
                                           const std::byte* bytes) const {
    const std::uint64_t slot = slot_floor(partition);
    if (slot >= dir_count_) return end();  // every partition is above the target

    if (slot_partition(slot) == partition) {
      const partition_layout p = layout_of(slot);
      if (p.count != 0) {
        const std::uint64_t above = search_above<Narrow>(p, value, bytes);
        if (above != 0) return const_iterator(this, slot, above - 1);
      }
      // The partition exists but holds nothing at or below the target, so the
      // answer is the tail of an earlier partition.
      return floor_before(slot);
    }
    // slot_partition(slot) < partition: every element here precedes the target,
    // so this partition's own tail is the answer -- if it has one.
    return floor_at_or_before(slot);
  }

  template <bool Narrow>
  [[nodiscard]] const_iterator find_parts(std::uint64_t partition, std::uint64_t value,
                                          const std::byte* bytes) const {
    const std::uint64_t slot = slot_exact(partition);
    if (slot >= dir_count_) return end();
    const partition_layout p = layout_of(slot);
    if (p.count == 0) return end();

    // Anything the query carries outside this partition's stored range cannot be
    // matched by a record, whose bytes there are zero by construction.
    std::uint64_t index = 0;
    if constexpr (Narrow) {
      const narrow_probe q = probe_narrow(p, value);
      if (q.tail) return end();
      index = search_lower_narrow(p, q.value);
      if (index >= p.count) return end();
      if (load_address(p.records + index * p.stride, p.address_width) != q.value) {
        return end();
      }
    } else {
      const wide_probe q = probe_wide(p, bytes);
      if (q.head || q.tail) return end();
      index = search_wide(p, q.bytes, 0);
      if (index >= p.count) return end();
      if (std::memcmp(p.records + index * p.stride, q.bytes, p.address_width) != 0) return end();
    }
    return const_iterator(this, slot, index);
  }

  [[nodiscard]] const std::byte* slot_bytes(std::uint64_t i) const {
    return directory_ + i * dir_.stride;
  }
  [[nodiscard]] dir_entry slot(std::uint64_t i) const {
    return decode_dir_entry(slot_bytes(i), dir_, i, address_width_);
  }
  // Directory fields are read as one wide load plus a mask rather than a switch
  // on the width. The 8-byte read is proven in bounds at open(): the directory
  // ends at or before image_size - sizeof(footer), so eight bytes past its last
  // byte still land inside the image.
  //
  /// A dense directory does not store the partition: slot i *is* partition i.
  [[nodiscard]] std::uint64_t slot_partition(std::uint64_t i) const {
    if (dir_.partition_width == 0) return i;
    return detail::load_uint_masked(slot_bytes(i) + dir_.partition_at, dir_.partition_mask);
  }
  [[nodiscard]] std::uint64_t slot_offset(std::uint64_t i) const {
    return detail::load_uint_masked(slot_bytes(i) + dir_.offset_at, dir_.offset_mask);
  }
  [[nodiscard]] std::uint64_t slot_count(std::uint64_t i) const {
    return detail::load_uint_masked(slot_bytes(i) + dir_.count_at, dir_.count_mask);
  }

  /// Everything about one partition's byte layout, read from the directory in a
  /// single 32-byte load. Stride varies per partition once remapping is in play,
  /// so it is resolved once here rather than re-derived at each access.
  struct partition_layout {
    const std::byte* records = nullptr;
    const std::byte* remap = nullptr;  ///< null unless the partition is remapped
    std::uint64_t count = 0;
    std::uint32_t remap_count = 0;
    unsigned stride = 0;
    unsigned ref_width = 0;  ///< local width when remapped, else the global one
    /// The stretch of the address field this partition stores: bytes
    /// [address_skip, address_skip + address_width) of a field address_width()
    /// bytes wide. Everything outside it is zero in every record here.
    unsigned address_width = 0;
    unsigned address_skip = 0;
    bool remapped = false;
  };

  [[nodiscard]] partition_layout layout_of(std::uint64_t s) const {
    const std::byte* p = slot_bytes(s);
    const std::uint64_t offset = detail::load_uint_masked(p + dir_.offset_at, dir_.offset_mask);
    const std::uint32_t schema_id =
        dir_.schema_width != 0
            ? static_cast<std::uint32_t>(
                  detail::load_uint_masked(p + dir_.schema_at, dir_.schema_mask))
            : 0;

    partition_layout out;
    out.count = detail::load_uint_masked(p + dir_.count_at, dir_.count_mask);
    out.remapped = schema_id != remap::none;
    out.remap_count =
        dir_.remap_width != 0
            ? static_cast<std::uint32_t>(
                  detail::load_uint_masked(p + dir_.remap_at, dir_.remap_mask))
            : 0;
    // Absent fields mean no partition trims anything, so the full field is the
    // answer and images without trimming cost nothing to read.
    out.address_width =
        dir_.addr_width != 0
            ? static_cast<unsigned>(detail::load_uint_masked(p + dir_.addr_at, dir_.addr_mask))
            : address_width_;
    out.address_skip =
        dir_.skip_width != 0
            ? static_cast<unsigned>(detail::load_uint_masked(p + dir_.skip_at, dir_.skip_mask))
            : 0;
    out.ref_width = out.remapped ? schema_id : ref_width_;
    out.stride = out.address_width + (out.remapped ? schema_id : value_part_);
    out.remap = out.remapped ? base_ + offset : nullptr;
    out.records =
        base_ + offset +
        (out.remapped ? static_cast<std::uint64_t>(out.remap_count) * sizeof(std::uint32_t) : 0);
    return out;
  }

  /// First index strictly above the target. `rec <= query` reduces to the same
  /// comparison whether or not the query has anything below the partition's
  /// stored range, so unlike lower_bound this never consults `tail`.
  template <bool Narrow>
  [[nodiscard]] std::uint64_t search_above(const partition_layout& p, std::uint64_t value,
                                           const std::byte* bytes) const {
    if constexpr (Narrow) {
      return search_upper_narrow(p, probe_narrow(p, value).value);
    } else {
      const wide_probe q = probe_wide(p, bytes);
      return q.head ? p.count : search_wide(p, q.bytes, 1);
    }
  }

  [[nodiscard]] const std::byte* record(std::uint64_t s, std::uint64_t i) const {
    const partition_layout p = layout_of(s);
    return p.records + i * p.stride;
  }
  /// The address as an integer. The stored bytes are the field's low `width`
  /// bytes starting `skip` in, so shifting them back up reproduces the value
  /// exactly -- the trimmed bytes were zero in every record here.
  ///
  /// The masked 8-byte load is in bounds for the same reason the directory's is:
  /// open() proves the records end at or before dir_offset and that dir_offset
  /// itself has eight readable bytes after it, so eight bytes from any record
  /// start still land inside the image.
  [[nodiscard]] std::uint64_t address_at(std::uint64_t s, std::uint64_t i) const {
    const partition_layout p = layout_of(s);
    const std::uint64_t stored =
        load_address(p.records + i * p.stride, p.address_width);
    return p.address_skip >= 8 ? 0 : stored << (8 * p.address_skip);
  }
  /// Only the bytes actually stored. Use address_skip_at() to place them, or
  /// address_into() to get the whole field back.
  [[nodiscard]] std::span<const std::byte> address_bytes_at(std::uint64_t s,
                                                            std::uint64_t i) const {
    const partition_layout p = layout_of(s);
    return {p.records + i * p.stride, p.address_width};
  }
  [[nodiscard]] unsigned address_skip_at(std::uint64_t s) const {
    return layout_of(s).address_skip;
  }
  /// Rebuild the full address field, zero-filling what this partition trimmed.
  /// `out` must be exactly address_width() bytes.
  [[nodiscard]] bool address_into_at(std::uint64_t s, std::uint64_t i,
                                     std::span<std::byte> out) const {
    if (out.size() != address_width_) return false;
    const partition_layout p = layout_of(s);
    std::memset(out.data(), 0, out.size());  // skip + width <= address_width_, proven at open
    std::memcpy(out.data() + p.address_skip, p.records + i * p.stride, p.address_width);
    return true;
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
    const partition_layout p = layout_of(it.slot_);
    const std::byte* r = p.records + it.index_ * p.stride;
    if (!interned_) return r + p.address_width;

    std::uint64_t ref = detail::load_uint(r + p.address_width, p.ref_width);
    if (p.remapped) {
      // The local index only means anything through this partition's table, and
      // a corrupt one would otherwise read straight into the records that
      // follow it.
      if (ref >= p.remap_count) return nullptr;
      ref = detail::load<std::uint32_t>(p.remap + ref * sizeof(std::uint32_t));
    }
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

  // A query is given at the full address width; a partition stores only part of
  // it. These reduce the query to that part once per lookup, and record what was
  // left over on each side.
  //
  //   tail -- the query has something *below* the stored range, so it sorts
  //           strictly after any record matching on the range itself
  //   head -- the query has something *above* it, so it sorts after every record
  //           in the partition outright
  //
  // Both are settled before the search and never consulted inside it. With tail
  // set, "rec < query" becomes "rec <= query on the stored range", which each
  // regime folds into its operand rather than into a second loop: the narrow
  // side searches for value + 1, the wide side raises search_wide's limit.
  // See search_above for why upper_bound needs neither.
  struct narrow_probe {
    std::uint64_t value = 0;  ///< the query, shifted down to the stored range
    bool tail = false;
  };
  struct wide_probe {
    const std::byte* bytes = nullptr;  ///< the query, advanced to the stored range
    bool tail = false;
    bool head = false;
  };

  /// No `head` here: the stored value is an integer bounded by the stored width,
  /// so a query above everything representable simply loses every comparison and
  /// the loop returns count on its own.
  ///
  /// `value + 1` is what lower_bound searches for when `tail` is set, and it
  /// cannot overflow: tail can only be true when the shift is nonzero, which
  /// already bounds value below 2^(64-shift).
  [[nodiscard]] static narrow_probe probe_narrow(const partition_layout& p,
                                                 std::uint64_t address) noexcept {
    narrow_probe out;
    const unsigned s = 8 * p.address_skip;
    out.value = s >= 64 ? 0 : address >> s;
    out.tail = s >= 64 ? address != 0 : (address & ((std::uint64_t{1} << s) - 1)) != 0;
    return out;
  }

  [[nodiscard]] wide_probe probe_wide(const partition_layout& p,
                                      const std::byte* address) const noexcept {
    wide_probe out;
    out.bytes = address + p.address_skip;
    for (unsigned i = 0; i < p.address_skip; ++i) {
      if (address[i] != std::byte{0}) {
        out.head = true;
        break;
      }
    }
    for (unsigned i = p.address_skip + p.address_width; i < address_width_; ++i) {
      if (address[i] != std::byte{0}) {
        out.tail = true;
        break;
      }
    }
    return out;
  }

  /// Load a stored address of runtime width.
  ///
  /// The natural sizes are read at their natural size and only the odd widths
  /// fall back to an 8-byte load and a mask -- not load_uint's memcpy, which
  /// with a runtime length is a call per probe.
  ///
  /// Reading *only* `width` bytes is what matters here, and it is worth more
  /// than removing the switch. Masking a uniform 8-byte load looked free, and is
  /// on the directory, which stays in cache; on records it measured 8-11% slower
  /// on a stride-6 image, because eight bytes from every record start straddle a
  /// cache line roughly seven times as often as two bytes do, and these records
  /// come from DRAM. The switch itself costs nothing: the width is loop-invariant
  /// and the branch predicts perfectly.
  ///
  /// The 8-byte read in the fallback is in bounds because open() proves records
  /// end at or before dir_offset and that dir_offset has eight readable bytes
  /// after it, so eight bytes from any record start still land inside the image.
  /// The mask is derived here rather than carried on partition_layout: only the
  /// fallback wants it, and keeping it in the struct made it live across the
  /// search loop for nothing.
  [[nodiscard]] static std::uint64_t load_address(const std::byte* p, unsigned width) noexcept {
    switch (width) {
      case 1: return detail::load<std::uint8_t>(p);
      case 2: return detail::load<std::uint16_t>(p);
      case 4: return detail::load<std::uint32_t>(p);
      case 8: return detail::load<std::uint64_t>(p);
      default:  // 0, 3, 5, 6, 7
        return detail::load_uint_masked(p, detail::mask_for_width(width));
    }
  }

  // The comparison regime is a property of the image, not of a probe, so it is
  // resolved once per lookup and never inside the loop. The narrow bodies are
  // kept as their own plain functions rather than an instantiation of a shared
  // template: that is what preserves their original codegen, and templating them
  // measured ~2.5% slower even though the branch was already outside the loop.
  [[nodiscard]] std::uint64_t search_lower_narrow(const partition_layout& p,
                                                  std::uint64_t address) const {
    std::uint64_t lo = 0;
    std::uint64_t len = p.count;
    while (len > 0) {
      const std::uint64_t half = len / 2;
      const std::uint64_t mid = lo + half;
      if (load_address(p.records + mid * p.stride, p.address_width) < address) {
        lo = mid + 1;
        len -= half + 1;
      } else {
        len = half;
      }
    }
    return lo;
  }

  /// Lexicographic on the raw bytes, which is why a wide address has to be given
  /// in an order-preserving encoding.
  ///
  /// `limit` selects the bound, which is why there is only one wide loop: a
  /// memcmp result is compared `< 0` for lower_bound and `< 1` -- that is,
  /// `<= 0` -- for upper_bound, and a query carrying something below the stored
  /// range needs the second even when asked for the first. It is loop-invariant,
  /// so this costs a register rather than an instantiation.
  [[nodiscard]] std::uint64_t search_wide(const partition_layout& p, const std::byte* address,
                                          int limit) const {
    std::uint64_t lo = 0;
    std::uint64_t len = p.count;
    while (len > 0) {
      const std::uint64_t half = len / 2;
      const std::uint64_t mid = lo + half;
      if (std::memcmp(p.records + mid * p.stride, address, p.address_width) < limit) {
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

  [[nodiscard]] std::uint64_t search_upper_narrow(const partition_layout& p,
                                                  std::uint64_t address) const {
    std::uint64_t lo = 0;
    std::uint64_t len = p.count;
    while (len > 0) {
      const std::uint64_t half = len / 2;
      const std::uint64_t mid = lo + half;
      if (!(address < load_address(p.records + mid * p.stride, p.address_width))) {
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
  dir_layout dir_;
  std::vector<blob_dictionary_view> blob_dicts_;
  fixed_dictionary_view value_dict_;
  std::uint32_t record_stride_ = 0;
  std::uint32_t value_part_ = 0;  ///< record_stride_ - address_width_
  unsigned address_width_ = 1;
  unsigned ref_width_ = 1;
  bool dense_ = false;
  bool interned_ = false;
};

}  // namespace mmpack
