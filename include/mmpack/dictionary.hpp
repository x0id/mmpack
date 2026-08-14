#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "mmpack/detail/bits.hpp"
#include "mmpack/error.hpp"

// Dictionaries hold the distinct values that records refer to by index.
//
//   fixed: [dict_header 16B][ count * elem_size ]
//   blob:  [dict_header 16B][ (count+1) u32 offsets ][ blob bytes ]
//
// Ids are always an implied array index, never stored, so the serialized form is
// independent of how wide a reference the records happen to use. That is what
// lets the index width be chosen after the data has been measured.
//
// Two users: text fields intern into blob dictionaries, and whole-value
// interning stores the distinct value tuples in one fixed dictionary whose
// element size is the value stride.

namespace mmpack {

inline constexpr std::uint32_t dictionary_magic = 0x4d444943u;  // 'MDIC'

namespace dict_type {
inline constexpr std::uint32_t fixed = 0;
inline constexpr std::uint32_t blob = 1;
}  // namespace dict_type

struct dict_header {
  std::uint32_t magic;
  std::uint32_t type;
  std::uint32_t count;    ///< number of entries
  std::uint32_t payload;  ///< fixed: element size; blob: total blob bytes
};
static_assert(sizeof(dict_header) == 16, "dict_header must be padding-free");

// --- build side -------------------------------------------------------------

/// Interns variable-length byte strings and hands back dense ids.
///
/// Ids are `std::uint64_t` throughout, so no width has to be guessed while
/// interning; the record field width is chosen later from the final cardinality.
class blob_dictionary_builder {
 public:
  std::uint64_t intern(std::span<const std::byte> raw) {
    return intern(std::string_view(reinterpret_cast<const char*>(raw.data()), raw.size()));
  }

  std::uint64_t intern(std::string_view text) {
    if (const auto found = index_.find(text); found != index_.end()) return found->second;
    if (blob_size_ + text.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw build_error("dictionary blob exceeds 4 GiB");
    }
    const std::uint64_t id = order_.size();
    // unordered_map keeps element addresses stable across rehash, so the key
    // itself backs the ordered view without a second copy of the bytes.
    const auto [it, inserted] = index_.emplace(std::string(text), id);
    order_.push_back(&it->first);
    blob_size_ += static_cast<std::uint32_t>(text.size());
    return id;
  }

  [[nodiscard]] std::optional<std::uint64_t> find(std::string_view text) const {
    const auto found = index_.find(text);
    if (found == index_.end()) return std::nullopt;
    return found->second;
  }

  [[nodiscard]] std::string_view operator[](std::uint64_t id) const { return *order_[id]; }
  [[nodiscard]] std::uint64_t size() const noexcept { return order_.size(); }
  [[nodiscard]] bool empty() const noexcept { return order_.empty(); }
  [[nodiscard]] std::uint32_t blob_size() const noexcept { return blob_size_; }

  [[nodiscard]] std::uint64_t serialized_size() const noexcept {
    return sizeof(dict_header) + (order_.size() + 1) * sizeof(std::uint32_t) + blob_size_;
  }

  /// Append the dictionary to a sink at its current position.
  template <class Sink>
  void write(Sink& sink) const {
    const dict_header head{dictionary_magic, dict_type::blob,
                           static_cast<std::uint32_t>(order_.size()), blob_size_};
    sink.write(&head, sizeof(head));

    std::vector<std::uint32_t> offsets;
    offsets.reserve(order_.size() + 1);
    std::uint32_t running = 0;
    for (const std::string* entry : order_) {
      offsets.push_back(running);
      running += static_cast<std::uint32_t>(entry->size());
    }
    offsets.push_back(running);
    sink.write(offsets.data(), offsets.size() * sizeof(std::uint32_t));

    for (const std::string* entry : order_) sink.write(entry->data(), entry->size());
  }

 private:
  // Transparent hashing so lookups by string_view do not allocate.
  struct hash {
    using is_transparent = void;
    [[nodiscard]] std::size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
  };
  struct equal {
    using is_transparent = void;
    [[nodiscard]] bool operator()(std::string_view a, std::string_view b) const noexcept {
      return a == b;
    }
  };

  std::unordered_map<std::string, std::uint64_t, hash, equal> index_;
  std::vector<const std::string*> order_;
  std::uint32_t blob_size_ = 0;
};

/// Serialized size of a fixed dictionary, for laying out offsets before writing.
[[nodiscard]] inline std::uint64_t fixed_dictionary_size(std::uint64_t count,
                                                         std::uint32_t elem_size) noexcept {
  return sizeof(dict_header) + count * elem_size;
}

/// Write a fixed dictionary header. The caller then emits exactly
/// `count * elem_size` bytes of element data.
template <class Sink>
void write_fixed_dictionary_header(Sink& sink, std::uint64_t count, std::uint32_t elem_size) {
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw build_error("fixed dictionary exceeds 2^32 entries");
  }
  const dict_header head{dictionary_magic, dict_type::fixed, static_cast<std::uint32_t>(count),
                         elem_size};
  sink.write(&head, sizeof(head));
}

// --- read side --------------------------------------------------------------

/// Fixed-stride dictionary. Validation is O(1): the element size is uniform, so
/// only the total has to fit.
class fixed_dictionary_view {
 public:
  fixed_dictionary_view() = default;

  [[nodiscard]] static std::optional<fixed_dictionary_view> open(
      std::span<const std::byte> block) noexcept {
    if (block.size() < sizeof(dict_header)) return std::nullopt;
    const dict_header head = detail::load<dict_header>(block.data());
    if (head.magic != dictionary_magic || head.type != dict_type::fixed) return std::nullopt;

    const std::uint64_t total =
        sizeof(dict_header) + static_cast<std::uint64_t>(head.count) * head.payload;
    if (total > block.size()) return std::nullopt;

    fixed_dictionary_view view;
    view.elements_ = block.data() + sizeof(dict_header);
    view.count_ = head.count;
    view.elem_size_ = head.payload;
    return view;
  }

  [[nodiscard]] std::uint32_t size() const noexcept { return count_; }
  [[nodiscard]] std::uint32_t elem_size() const noexcept { return elem_size_; }
  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

  /// Base of an entry, or nullptr if the id is out of range. Returning nullptr
  /// rather than trusting the id is what makes a corrupt value reference safe.
  [[nodiscard]] const std::byte* element(std::uint64_t id) const noexcept {
    if (id >= count_) return nullptr;
    return elements_ + id * elem_size_;
  }

 private:
  const std::byte* elements_ = nullptr;
  std::uint32_t count_ = 0;
  std::uint32_t elem_size_ = 0;
};

/// Variable-length dictionary. open() proves the offset table monotonic and in
/// range, so lookups afterwards are a single id check.
class blob_dictionary_view {
 public:
  blob_dictionary_view() = default;

  [[nodiscard]] static std::optional<blob_dictionary_view> open(
      std::span<const std::byte> block) noexcept {
    if (block.size() < sizeof(dict_header)) return std::nullopt;
    const dict_header head = detail::load<dict_header>(block.data());
    if (head.magic != dictionary_magic || head.type != dict_type::blob) return std::nullopt;

    const std::uint64_t offsets_bytes =
        (static_cast<std::uint64_t>(head.count) + 1) * sizeof(std::uint32_t);
    const std::uint64_t total = sizeof(dict_header) + offsets_bytes + head.payload;
    if (total > block.size()) return std::nullopt;

    blob_dictionary_view view;
    view.offsets_ = block.data() + sizeof(dict_header);
    view.blob_ = view.offsets_ + offsets_bytes;
    view.count_ = head.count;

    std::uint32_t previous = 0;
    for (std::uint64_t i = 0; i <= head.count; ++i) {
      const auto value = detail::load<std::uint32_t>(view.offsets_ + i * sizeof(std::uint32_t));
      if (value < previous || value > head.payload) return std::nullopt;
      previous = value;
    }
    if (previous != head.payload) return std::nullopt;

    return view;
  }

  [[nodiscard]] std::uint32_t size() const noexcept { return count_; }
  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

  [[nodiscard]] std::optional<std::span<const std::byte>> bytes(std::uint64_t id) const noexcept {
    if (id >= count_) return std::nullopt;
    const auto from = detail::load<std::uint32_t>(offsets_ + id * sizeof(std::uint32_t));
    const auto to = detail::load<std::uint32_t>(offsets_ + (id + 1) * sizeof(std::uint32_t));
    return std::span<const std::byte>(blob_ + from, to - from);
  }

  [[nodiscard]] std::optional<std::string_view> text(std::uint64_t id) const noexcept {
    const auto raw = bytes(id);
    if (!raw) return std::nullopt;
    return std::string_view(reinterpret_cast<const char*>(raw->data()), raw->size());
  }

 private:
  const std::byte* offsets_ = nullptr;
  const std::byte* blob_ = nullptr;
  std::uint32_t count_ = 0;
};

}  // namespace mmpack
