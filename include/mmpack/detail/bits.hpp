#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace mmpack::detail {

// Every field in this format sits at an offset computed at runtime, so typed
// loads would be UB on any image the library did not write itself. All access
// goes through memcpy, which compilers lower to a single instruction.
template <class T>
[[nodiscard]] inline T load(const std::byte* p) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

template <class T>
inline void store(std::byte* p, const T& v) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  std::memcpy(p, &v, sizeof(T));
}

// Runtime-width access reads and writes exactly `width` bytes, which only lines
// up with the native integer layout on a little-endian host. The format is
// native-endian by design, so rather than let a big-endian build silently
// produce wrong values for the odd widths, refuse to compile there.
static_assert(std::endian::native == std::endian::little,
              "mmpack images are native-endian and assume a little-endian host");

/// Load an unsigned integer of runtime width (1..8 bytes), zero extended.
/// This is the hot path for field access: the width comes from the schema, so it
/// is constant per field and the branch predicts perfectly. Numeric fields are
/// always 1, 2, 4 or 8; only `bytes` fields take the odd widths, and those are
/// never on the search path.
[[nodiscard]] inline std::uint64_t load_uint(const std::byte* p, unsigned width) noexcept {
  switch (width) {
    case 1: return load<std::uint8_t>(p);
    case 2: return load<std::uint16_t>(p);
    case 4: return load<std::uint32_t>(p);
    case 8: return load<std::uint64_t>(p);
    default: {
      std::uint64_t v = 0;
      std::memcpy(&v, p, width > 8 ? 8 : width);
      return v;
    }
  }
}

/// Store an unsigned integer in a runtime width. The caller must have proved the
/// value fits; width selection is what guarantees that.
inline void store_uint(std::byte* p, unsigned width, std::uint64_t v) noexcept {
  switch (width) {
    case 1: store<std::uint8_t>(p, static_cast<std::uint8_t>(v)); break;
    case 2: store<std::uint16_t>(p, static_cast<std::uint16_t>(v)); break;
    case 4: store<std::uint32_t>(p, static_cast<std::uint32_t>(v)); break;
    case 8: store<std::uint64_t>(p, v); break;
    default: std::memcpy(p, &v, width > 8 ? 8 : width); break;
  }
}

/// Narrowest of 1, 2, 4, 8 that can hold `range` distinct values 0..range-1.
[[nodiscard]] constexpr unsigned width_for(std::uint64_t max_value) noexcept {
  if (max_value <= 0xffull) return 1;
  if (max_value <= 0xffffull) return 2;
  if (max_value <= 0xffffffffull) return 4;
  return 8;
}

/// Mask selecting the low `width` bytes.
[[nodiscard]] constexpr std::uint64_t mask_for_width(unsigned width) noexcept {
  return width >= 8 ? ~std::uint64_t{0} : (std::uint64_t{1} << (8 * width)) - 1;
}

/// Load a narrow unsigned field as one wide load plus a mask, with no branch on
/// the width. Reads 8 bytes, so the caller must have proved that many are
/// readable -- see table's directory accessors, where open() has already bounded
/// the directory well clear of the image end.
[[nodiscard]] inline std::uint64_t load_uint_masked(const std::byte* p,
                                                    std::uint64_t mask) noexcept {
  return load<std::uint64_t>(p) & mask;
}

/// Largest value representable in `width` bytes.
[[nodiscard]] constexpr std::uint64_t max_for_width(unsigned width) noexcept {
  switch (width) {
    case 1: return 0xffull;
    case 2: return 0xffffull;
    case 4: return 0xffffffffull;
    default: return ~std::uint64_t{0};
  }
}

[[nodiscard]] constexpr bool is_valid_width(unsigned width) noexcept {
  return width == 1 || width == 2 || width == 4 || width == 8;
}

[[nodiscard]] constexpr std::uint64_t align_up(std::uint64_t v, std::uint64_t a) noexcept {
  return (v + (a - 1)) & ~(a - 1);
}

/// operator-> support for iterators whose operator* returns a prvalue.
template <class T>
struct arrow_proxy {
  T held;
  const T* operator->() const noexcept { return &held; }
};

}  // namespace mmpack::detail
