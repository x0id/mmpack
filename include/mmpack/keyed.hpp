#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

#include "mmpack/builder.hpp"
#include "mmpack/table.hpp"

// Typed key ergonomics, layered over the (partition, address) core.
//
// mmpack stores two ordering coordinates and nothing else, so a key of any type
// and any width works as long as the caller can split it into a partition and an
// address. Doing that at every call site is repetitive and easy to get
// inconsistent, so these wrappers hold the split once.
//
//   struct split_v6 {
//     std::pair<std::uint64_t, std::uint64_t> operator()(const ipv6& k) const {
//       return {k.high, k.low};
//     }
//   };
//   struct join_v6 {
//     ipv6 operator()(std::uint64_t p, std::uint64_t a) const { return ipv6{p, a}; }
//   };
//
//   mmpack::keyed_builder<ipv6, split_v6> kb(tb, split_v6{});
//   auto rec = kb.begin_record(addr);   // splits, then begin_record_at
//   ...
//   mmpack::keyed_table<ipv6, split_v6, join_v6> kt(t, split_v6{}, join_v6{});
//   auto it = kt.floor(addr);
//   ipv6 back = kt.key(it);
//
// `Join` is optional, and key() exists only when it is supplied -- the same
// arrangement mmseek had, where join_key was optional and key() was gated on
// the joinable_traits concept. There it was a compile-time absence; here it is
// the same, because the gate lives in the wrapper rather than in the image.
//
// These are non-owning views over a table (or a reference to a builder) and add
// no runtime cost: every call forwards to the *_at form after an inlined split.

namespace mmpack {

/// Marker for "no join supplied", making key() unavailable.
struct no_join {};

template <class Key, class Split, class Join = no_join>
class keyed_table {
 public:
  using key_type = Key;
  using const_iterator = table::const_iterator;
  static constexpr bool joinable = !std::is_same_v<Join, no_join>;

  keyed_table(const table& source, Split split, Join join = {})
      : table_(&source), split_(std::move(split)), join_(std::move(join)) {}

  [[nodiscard]] const table& base() const noexcept { return *table_; }

  [[nodiscard]] const_iterator begin() const { return table_->begin(); }
  [[nodiscard]] const_iterator end() const { return table_->end(); }

  [[nodiscard]] const_iterator find(const Key& key) const {
    const auto parts = split_(key);
    return table_->find_at(parts.first, parts.second);
  }
  [[nodiscard]] const_iterator lower_bound(const Key& key) const {
    const auto parts = split_(key);
    return table_->lower_bound_at(parts.first, parts.second);
  }
  [[nodiscard]] const_iterator upper_bound(const Key& key) const {
    const auto parts = split_(key);
    return table_->upper_bound_at(parts.first, parts.second);
  }
  [[nodiscard]] const_iterator floor(const Key& key) const {
    const auto parts = split_(key);
    return table_->floor_at(parts.first, parts.second);
  }
  [[nodiscard]] const_iterator ceil(const Key& key) const {
    const auto parts = split_(key);
    return table_->ceil_at(parts.first, parts.second);
  }
  [[nodiscard]] bool contains(const Key& key) const { return find(key) != end(); }

  /// Rebuild the key an element was stored under. Present only when a join was
  /// supplied; without one there is nothing that knows the encoding.
  ///
  /// Templated on a defaulted parameter rather than constrained directly, so
  /// that its absence is a substitution failure a `requires` expression can
  /// detect, instead of a hard error.
  ///
  /// The join is offered whichever address form it accepts: an integer for
  /// images whose addresses fit one, the raw bytes otherwise. A join written for
  /// wide keys therefore works unchanged on a narrow image and vice versa,
  /// provided it takes the form that image actually has.
  ///
  /// The byte form is rebuilt at the full address width rather than passed
  /// through from the record, because a partition may store only the significant
  /// stretch of it. Handing a join the trimmed bytes would silently give it a
  /// different key than the one that was written. address_width() is a uint8 in
  /// the schema, so 255 bytes is a hard upper bound and this stays a stack copy;
  /// key reconstruction is not a hot path.
  template <class J = Join>
  [[nodiscard]] Key key(const const_iterator& it) const
    requires(!std::is_same_v<J, no_join>)
  {
    if constexpr (std::is_invocable_r_v<Key, const J&, std::uint64_t, std::uint64_t>) {
      return join_(it.partition(), it.address().value());
    } else {
      std::array<std::byte, 255> buffer;
      const std::span<std::byte> full(buffer.data(), table_->address_width());
      (void)it.address_into(full);
      return join_(it.partition(), std::span<const std::byte>(full));
    }
  }

 private:
  const table* table_;
  [[no_unique_address]] Split split_;
  [[no_unique_address]] Join join_;
};

// No deduction guide: Key never appears in the constructor's parameters, so it
// cannot be deduced. Spell it out -- keyed_table<ipv6, split_v6, join_v6>.

/// Feeds a table_builder from typed keys. The builder itself stays in
/// caller-split mode, so the resulting image reports no 64-bit key mapping.
template <class Key, class Split>
class keyed_builder {
 public:
  keyed_builder(table_builder& target, Split split)
      : builder_(&target), split_(std::move(split)) {}

  [[nodiscard]] table_builder& base() const noexcept { return *builder_; }

  [[nodiscard]] record_ref begin_record(const Key& key) {
    const auto parts = split_(key);
    return builder_->begin_record_at(parts.first, parts.second);
  }

  void commit(const record_ref& rec) { builder_->commit(rec); }

 private:
  table_builder* builder_;
  [[no_unique_address]] Split split_;
};

}  // namespace mmpack
