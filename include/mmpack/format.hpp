#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "mmpack/detail/bits.hpp"

// On-disk image layout (all integers native-endian; images are not portable
// across architectures of differing endianness -- the magic check catches it):
//
//   +----------------------------------------------------------------+
//   | header      16 bytes, written first so the file is sniffable    |
//   | dictionaries  field dictionaries + the composite value          |
//   |               dictionary, written before the records because    |
//   |               record contents reference them by index           |
//   | partition 0 records   count * record_stride, sorted by address  |
//   | partition 1 records                                             |
//   | ...                                                             |
//   | directory   dir_count * sizeof(dir_entry), sorted by partition  |
//   | schema      field descriptors + name blob                       |
//   | footer      64 bytes, always the last bytes of the image        |
//   +----------------------------------------------------------------+
//
// The directory, schema and footer live at the end so the writer never has to
// seek: partition offsets are only known after their records are emitted, and
// the schema is only final once the data has been measured. The reader finds
// the footer at base + length - sizeof(footer) and works backwards.
//
// A record is [address][value]. The address comes first at a fixed width so the
// binary search is uniform and does not have to consult the schema. The value is
// either the fields laid out inline, or one reference into the composite
// dictionary -- see schema.hpp.

namespace mmpack {

inline constexpr char magic[8] = {'M', 'M', 'P', 'A', 'C', 'K', '0', '1'};
/// 2 added per-partition value remap. A version-1 reader never consulted
/// dir_entry::schema_id, so handed a remapped image it would read records at the
/// global stride and return silent garbage rather than failing -- hence a bump
/// rather than a compatible extension.
/// 3 packed the directory at data-derived widths instead of a fixed 32-byte
/// struct.
/// 4 gave each partition its own address range within the address field. The
/// two new schema_header fields moved the descriptor blob that follows it, so a
/// version-3 reader would misparse the schema outright rather than misread
/// records -- but a bump is the honest way to say so either way.
inline constexpr std::uint32_t format_version = 4;

namespace flags {
/// Directory slot i describes partition i, including empty ones: O(1) select.
inline constexpr std::uint32_t dense_directory = 1u << 0;
}  // namespace flags

struct header {
  char magic[8];
  std::uint32_t format_version;
  std::uint32_t reserved;
};
static_assert(sizeof(header) == 16, "header must be padding-free");

/// Per-partition value remap.
///
/// Under whole-value interning every record holds a reference into the global
/// composite dictionary, so the reference is as wide as the *total* distinct
/// tuple count demands -- 4 bytes at 16M tuples. But an individual partition
/// usually touches only a handful of those tuples. Giving such a partition its
/// own table of the global ids it actually uses lets its records carry a 1- or
/// 2-byte local index instead, which is pure stride reduction on the hot path.
///
/// A remapped partition's bytes are laid out as
///
///     [remap table: remap_count * 4 bytes of global tuple ids][records]
///
/// with `offset` pointing at the table, so records start at
/// `offset + remap_count * 4` and run at `address_width + schema_id` stride.
namespace remap {
inline constexpr std::uint32_t none = 0;  ///< records hold a global reference
/// Any other value of schema_id is the local reference width in bytes. Only 1
/// and 2 occur: a 4-byte local index is never narrower than the global one, so
/// remapping could not pay for its own table.
inline constexpr std::uint32_t max_local_width = 2;
}  // namespace remap

/// One directory slot, in its decoded form. This is **not** the on-disk layout:
/// the directory is stored field-by-field at widths chosen from the data, the
/// same way records are, because a full 32-byte struct spends most of its bytes
/// on leading zeros. See dir_layout.
struct dir_entry {
  std::uint64_t partition;     ///< partition index
  std::uint64_t offset;        ///< byte offset of the partition's bytes, from image base
  std::uint64_t count;         ///< number of records
  std::uint32_t schema_id;     ///< 0 = global reference; else local reference width
  std::uint32_t remap_count;   ///< entries in the remap table; 0 when schema_id == 0
  /// The stretch of the address field this partition actually stores. Every
  /// address here is zero outside [address_skip, address_skip + address_width),
  /// so those bytes are trimmed and re-supplied as zeros on read. See the
  /// trimming rule in table.hpp.
  std::uint32_t address_width;
  std::uint32_t address_skip;
};

/// How a directory slot is packed on disk.
///
/// Every field is narrowed to what the image actually needs: `offset` to the
/// image size, `count` to the largest partition, `schema_id` to a single byte,
/// and `remap_count` to the largest remap table. `partition` disappears entirely
/// from a dense directory, where slot i describes partition i by construction --
/// storing it would only be a value that has to be checked against its own
/// index.
///
/// Typical geo-IP shape: 0 + 4 + 2 + 1 + 2 = 9 bytes against the 32 a struct
/// would take, so seven slots share a cache line instead of two.
struct dir_layout {
  unsigned partition_width = 0;  ///< 0 = implicit, i.e. a dense directory
  unsigned offset_width = 0;
  unsigned count_width = 0;
  unsigned schema_width = 0;  ///< 0 or 1; 0 when no partition is remapped
  unsigned remap_width = 0;   ///< 0 when no partition is remapped
  unsigned addr_width = 0;    ///< 0 when every partition stores the full address
  unsigned skip_width = 0;    ///< 0 when no partition trims a prefix

  unsigned partition_at = 0;
  unsigned offset_at = 0;
  unsigned count_at = 0;
  unsigned schema_at = 0;
  unsigned remap_at = 0;
  unsigned addr_at = 0;
  unsigned skip_at = 0;
  unsigned stride = 0;

  // Masks for reading a field as one wide load rather than a width switch.
  // Field widths are fixed for the whole image, so the dispatch a generic
  // load_uint would do is pure overhead on the lookup path.
  std::uint64_t partition_mask = 0;
  std::uint64_t offset_mask = 0;
  std::uint64_t count_mask = 0;
  std::uint64_t schema_mask = 0;
  std::uint64_t remap_mask = 0;
  std::uint64_t addr_mask = 0;
  std::uint64_t skip_mask = 0;
};

[[nodiscard]] inline dir_layout make_dir_layout(unsigned partition_width, unsigned offset_width,
                                                unsigned count_width, unsigned schema_width,
                                                unsigned remap_width, unsigned addr_width,
                                                unsigned skip_width) noexcept {
  dir_layout out;
  out.partition_width = partition_width;
  out.offset_width = offset_width;
  out.count_width = count_width;
  out.schema_width = schema_width;
  out.remap_width = remap_width;
  out.addr_width = addr_width;
  out.skip_width = skip_width;

  unsigned at = 0;
  out.partition_at = at;
  at += partition_width;
  out.offset_at = at;
  at += offset_width;
  out.count_at = at;
  at += count_width;
  out.schema_at = at;
  at += schema_width;
  out.remap_at = at;
  at += remap_width;
  out.addr_at = at;
  at += addr_width;
  out.skip_at = at;
  at += skip_width;
  out.stride = at;

  out.partition_mask = detail::mask_for_width(partition_width);
  out.offset_mask = detail::mask_for_width(offset_width);
  out.count_mask = detail::mask_for_width(count_width);
  out.schema_mask = detail::mask_for_width(schema_width);
  out.remap_mask = detail::mask_for_width(remap_width);
  out.addr_mask = detail::mask_for_width(addr_width);
  out.skip_mask = detail::mask_for_width(skip_width);
  return out;
}

inline void encode_dir_entry(std::byte* out, const dir_layout& layout, const dir_entry& d) {
  std::memset(out, 0, layout.stride);
  if (layout.partition_width) {
    detail::store_uint(out + layout.partition_at, layout.partition_width, d.partition);
  }
  detail::store_uint(out + layout.offset_at, layout.offset_width, d.offset);
  detail::store_uint(out + layout.count_at, layout.count_width, d.count);
  if (layout.schema_width) {
    detail::store_uint(out + layout.schema_at, layout.schema_width, d.schema_id);
  }
  if (layout.remap_width) {
    detail::store_uint(out + layout.remap_at, layout.remap_width, d.remap_count);
  }
  if (layout.addr_width) {
    detail::store_uint(out + layout.addr_at, layout.addr_width, d.address_width);
  }
  if (layout.skip_width) {
    detail::store_uint(out + layout.skip_at, layout.skip_width, d.address_skip);
  }
}

/// `index` supplies the partition when the directory is dense and therefore
/// does not store it. `full_address_width` supplies the address width when no
/// partition trims anything and the field is therefore absent.
[[nodiscard]] inline dir_entry decode_dir_entry(const std::byte* in, const dir_layout& layout,
                                                std::uint64_t index,
                                                unsigned full_address_width = 0) noexcept {
  dir_entry d{};
  d.partition = layout.partition_width
                    ? detail::load_uint(in + layout.partition_at, layout.partition_width)
                    : index;
  d.offset = detail::load_uint(in + layout.offset_at, layout.offset_width);
  d.count = detail::load_uint(in + layout.count_at, layout.count_width);
  d.schema_id =
      layout.schema_width
          ? static_cast<std::uint32_t>(detail::load_uint(in + layout.schema_at, layout.schema_width))
          : 0;
  d.remap_count =
      layout.remap_width
          ? static_cast<std::uint32_t>(detail::load_uint(in + layout.remap_at, layout.remap_width))
          : 0;
  d.address_width =
      layout.addr_width
          ? static_cast<std::uint32_t>(detail::load_uint(in + layout.addr_at, layout.addr_width))
          : full_address_width;
  d.address_skip =
      layout.skip_width
          ? static_cast<std::uint32_t>(detail::load_uint(in + layout.skip_at, layout.skip_width))
          : 0;
  return d;
}

struct footer {
  char magic[8];
  std::uint32_t format_version;
  std::uint32_t flags;
  std::uint64_t record_count;    ///< total records across all partitions
  std::uint64_t dir_offset;      ///< byte offset of the directory
  std::uint64_t dir_count;       ///< number of directory slots
  std::uint64_t schema_offset;   ///< byte offset of the schema block
  std::uint64_t schema_size;     ///< schema block bytes
  std::uint64_t image_size;      ///< total image bytes, including this footer
};
static_assert(sizeof(footer) == 64, "footer must be padding-free");

}  // namespace mmpack
