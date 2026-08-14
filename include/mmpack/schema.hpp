#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "mmpack/detail/bits.hpp"
#include "mmpack/error.hpp"

// The schema is the whole point of this library: record shape is data, not a C++
// type, so it is computed from the input and written into the image.
//
// Serialized block, placed between the directory and the footer:
//
//   [schema_header 32B][field_desc * field_count][dict_entry * dict_count][name blob]
//
// Field descriptors always describe the layout *within the value tuple*, so both
// value modes share one description. Only the tuple base differs:
//
//   inline:   base = record + address_width
//   interned: ref  = load(record + address_width, ref_width)
//             base = dict[value_dict].offset + ref * value_stride
//
// There is no "packed" flag: the builder computes an explicit offset per field,
// so padding (or its absence) is already expressed by those offsets.

namespace mmpack {

using field_id = std::uint32_t;

enum class field_kind : std::uint8_t {
  uint_ = 0,   ///< unsigned integer, width and bias chosen from the value range
  sint_ = 1,   ///< signed integer, stored biased so the stored form is unsigned
  f32 = 2,     ///< IEEE binary32, always 4 bytes
  f64 = 3,     ///< IEEE binary64, always 8 bytes
  bytes = 4,   ///< fixed-size raw bytes, width declared by the caller
  text = 5,    ///< interned into a blob dictionary, stored as an index
};

inline constexpr std::uint8_t max_field_kind = 5;

[[nodiscard]] inline const char* to_string(field_kind k) noexcept {
  switch (k) {
    case field_kind::uint_: return "uint";
    case field_kind::sint_: return "sint";
    case field_kind::f32:   return "f32";
    case field_kind::f64:   return "f64";
    case field_kind::bytes: return "bytes";
    case field_kind::text:  return "text";
  }
  return "unknown";
}

enum class value_mode : std::uint8_t {
  inline_fields = 0,  ///< the value tuple sits in the record
  interned = 1,       ///< the record holds a reference into the value dictionary
};

inline constexpr std::uint32_t schema_magic = 0x4d534348u;  // 'MSCH'

struct schema_header {
  std::uint32_t magic;
  std::uint32_t field_count;
  std::uint32_t dict_count;
  std::uint32_t record_stride;   ///< bytes per record, address included
  std::uint32_t value_stride;    ///< bytes per value tuple
  std::uint32_t name_bytes;      ///< size of the trailing name blob
  std::uint8_t address_width;    ///< 1, 2, 4 or 8
  std::uint8_t address_bits;     ///< key split: address = key & ((1<<bits)-1)
  std::uint8_t mode;             ///< value_mode
  std::uint8_t ref_width;        ///< interned only: 1, 2, 4 or 8
  std::uint32_t value_dict;      ///< interned only: index into the dict table
};
static_assert(sizeof(schema_header) == 32, "schema_header must be padding-free");

struct field_desc {
  std::uint32_t offset;       ///< byte offset within the value tuple
  std::uint32_t name_offset;  ///< into the name blob, NUL terminated
  std::uint64_t bias;         ///< stored = actual - bias (wrapping); 0 if unused
  std::uint32_t width;        ///< bytes
  std::uint8_t kind;          ///< field_kind
  std::uint8_t reserved;
  std::uint16_t dict_index;   ///< text fields: index into the dict table
};
static_assert(sizeof(field_desc) == 24, "field_desc must be padding-free");

/// Where a dictionary lives in the image. Kept in the schema because dictionary
/// indices are a schema concept, which makes the schema alone enough to
/// interpret any record.
struct dict_entry {
  std::uint64_t offset;
  std::uint64_t size;
};
static_assert(sizeof(dict_entry) == 16, "dict_entry must be padding-free");

// --- build side -------------------------------------------------------------

struct field_decl {
  std::string name;
  field_kind kind;
  std::uint32_t fixed_width;  ///< bytes/f32/f64 only; 0 means "choose from data"
};

/// Declares what fields exist. Widths, biases and offsets are *not* set here --
/// they are computed once the data has been measured.
class schema_builder {
 public:
  field_id add_uint(std::string name) { return add(std::move(name), field_kind::uint_, 0); }
  field_id add_sint(std::string name) { return add(std::move(name), field_kind::sint_, 0); }
  field_id add_f32(std::string name) { return add(std::move(name), field_kind::f32, 4); }
  field_id add_f64(std::string name) { return add(std::move(name), field_kind::f64, 8); }
  field_id add_text(std::string name) { return add(std::move(name), field_kind::text, 0); }

  /// Small fixed-size raw field, at most 8 bytes (a MAC address, a flags word).
  /// Anything longer belongs in a text field, which handles arbitrary length and
  /// deduplicates as well.
  field_id add_bytes(std::string name, std::uint32_t width) {
    if (width == 0 || width > 8) {
      throw build_error("bytes field '" + name + "' must be 1..8 bytes wide; use a text field for more");
    }
    return add(std::move(name), field_kind::bytes, width);
  }

  [[nodiscard]] std::size_t size() const noexcept { return fields_.size(); }
  [[nodiscard]] bool empty() const noexcept { return fields_.empty(); }
  [[nodiscard]] const field_decl& operator[](field_id id) const { return fields_[id]; }
  [[nodiscard]] const std::vector<field_decl>& fields() const noexcept { return fields_; }

  [[nodiscard]] std::optional<field_id> find(std::string_view name) const {
    for (std::size_t i = 0; i < fields_.size(); ++i) {
      if (fields_[i].name == name) return static_cast<field_id>(i);
    }
    return std::nullopt;
  }

 private:
  field_id add(std::string name, field_kind kind, std::uint32_t width) {
    if (name.empty()) throw build_error("field name must not be empty");
    if (name.find('\0') != std::string::npos) {
      throw build_error("field name must not contain NUL: '" + name + "'");
    }
    if (find(name)) throw build_error("duplicate field name: '" + name + "'");
    fields_.push_back(field_decl{std::move(name), kind, width});
    return static_cast<field_id>(fields_.size() - 1);
  }

  std::vector<field_decl> fields_;
};

// --- read side --------------------------------------------------------------

/// Validated, non-owning view of a serialized schema block.
///
/// open() proves every descriptor self-consistent -- widths legal for the kind,
/// each field inside the value tuple, names NUL terminated inside the blob,
/// dictionary indices in range -- before any accessor is built. Everything
/// downstream may then trust the schema, which is what keeps field access down
/// to an offset add and a width-dispatched load.
class schema_view {
 public:
  schema_view() = default;

  [[nodiscard]] static std::optional<schema_view> open(std::span<const std::byte> block) noexcept {
    if (block.size() < sizeof(schema_header)) return std::nullopt;
    const schema_header head = detail::load<schema_header>(block.data());
    if (head.magic != schema_magic) return std::nullopt;

    const std::uint64_t fields_bytes =
        static_cast<std::uint64_t>(head.field_count) * sizeof(field_desc);
    const std::uint64_t dicts_bytes =
        static_cast<std::uint64_t>(head.dict_count) * sizeof(dict_entry);
    const std::uint64_t total =
        sizeof(schema_header) + fields_bytes + dicts_bytes + head.name_bytes;
    if (total > block.size()) return std::nullopt;

    if (!detail::is_valid_width(head.address_width)) return std::nullopt;
    if (head.address_bits > 64) return std::nullopt;
    if (head.mode > static_cast<std::uint8_t>(value_mode::interned)) return std::nullopt;
    if (head.value_stride == 0 && head.field_count != 0) return std::nullopt;

    // The record must be wide enough for the address plus whatever follows it.
    std::uint64_t minimum = head.address_width;
    if (head.mode == static_cast<std::uint8_t>(value_mode::interned)) {
      if (!detail::is_valid_width(head.ref_width)) return std::nullopt;
      if (head.value_dict >= head.dict_count) return std::nullopt;
      minimum += head.ref_width;
    } else {
      minimum += head.value_stride;
    }
    if (head.record_stride < minimum) return std::nullopt;

    schema_view view;
    view.head_ = head;
    view.fields_ = block.data() + sizeof(schema_header);
    view.dicts_ = view.fields_ + fields_bytes;
    view.names_ = view.dicts_ + dicts_bytes;

    for (std::uint32_t i = 0; i < head.field_count; ++i) {
      const field_desc f = view.field(i);
      if (f.kind > max_field_kind) return std::nullopt;
      const auto kind = static_cast<field_kind>(f.kind);

      switch (kind) {
        case field_kind::uint_:
        case field_kind::sint_:
        case field_kind::text:
          if (!detail::is_valid_width(f.width)) return std::nullopt;
          break;
        case field_kind::f32:
          if (f.width != 4) return std::nullopt;
          break;
        case field_kind::f64:
          if (f.width != 8) return std::nullopt;
          break;
        case field_kind::bytes:
          // Capped at 8 because a bytes field is read through the same
          // runtime-width path as the numeric ones; a wider declaration would
          // make that path write past a uint64.
          if (f.width == 0 || f.width > 8) return std::nullopt;
          break;
      }
      // Every field must lie wholly inside the value tuple, so an accessor can
      // never read past it however corrupt the image is.
      if (f.offset > head.value_stride || f.width > head.value_stride - f.offset) {
        return std::nullopt;
      }
      if (kind == field_kind::text && f.dict_index >= head.dict_count) return std::nullopt;

      // Names must be NUL terminated inside the blob.
      if (f.name_offset >= head.name_bytes) return std::nullopt;
      bool terminated = false;
      for (std::uint32_t j = f.name_offset; j < head.name_bytes; ++j) {
        if (view.names_[j] == std::byte{0}) {
          terminated = true;
          break;
        }
      }
      if (!terminated) return std::nullopt;
    }

    return view;
  }

  [[nodiscard]] std::uint32_t field_count() const noexcept { return head_.field_count; }
  [[nodiscard]] std::uint32_t dict_count() const noexcept { return head_.dict_count; }
  [[nodiscard]] std::uint32_t record_stride() const noexcept { return head_.record_stride; }
  [[nodiscard]] std::uint32_t value_stride() const noexcept { return head_.value_stride; }
  [[nodiscard]] unsigned address_width() const noexcept { return head_.address_width; }
  [[nodiscard]] unsigned address_bits() const noexcept { return head_.address_bits; }
  [[nodiscard]] unsigned ref_width() const noexcept { return head_.ref_width; }
  [[nodiscard]] std::uint32_t value_dict() const noexcept { return head_.value_dict; }
  [[nodiscard]] value_mode mode() const noexcept { return static_cast<value_mode>(head_.mode); }
  [[nodiscard]] bool interned() const noexcept { return mode() == value_mode::interned; }

  [[nodiscard]] field_desc field(field_id id) const noexcept {
    return detail::load<field_desc>(fields_ + static_cast<std::size_t>(id) * sizeof(field_desc));
  }

  [[nodiscard]] dict_entry dictionary(std::uint32_t index) const noexcept {
    return detail::load<dict_entry>(dicts_ + static_cast<std::size_t>(index) * sizeof(dict_entry));
  }

  [[nodiscard]] std::string_view name(field_id id) const noexcept {
    const auto* p = reinterpret_cast<const char*>(names_) + field(id).name_offset;
    return std::string_view(p);  // proven NUL terminated inside the blob by open()
  }

  /// Field id by name. Linear, but field counts are small and this is not a hot
  /// path -- keeping it allocation-free matters more.
  [[nodiscard]] std::optional<field_id> find(std::string_view wanted) const noexcept {
    for (std::uint32_t i = 0; i < head_.field_count; ++i) {
      if (name(i) == wanted) return i;
    }
    return std::nullopt;
  }

  /// Split a key the way the image was built.
  [[nodiscard]] std::uint64_t partition_of(std::uint64_t key) const noexcept {
    return head_.address_bits >= 64 ? 0 : key >> head_.address_bits;
  }
  [[nodiscard]] std::uint64_t address_of(std::uint64_t key) const noexcept {
    return head_.address_bits >= 64 ? key : key & ((std::uint64_t{1} << head_.address_bits) - 1);
  }

 private:
  schema_header head_{};
  const std::byte* fields_ = nullptr;
  const std::byte* dicts_ = nullptr;
  const std::byte* names_ = nullptr;
};

}  // namespace mmpack
