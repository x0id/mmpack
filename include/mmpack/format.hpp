#pragma once

#include <cstddef>
#include <cstdint>

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
inline constexpr std::uint32_t format_version = 2;

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

struct dir_entry {
  std::uint64_t partition;     ///< partition index
  std::uint64_t offset;        ///< byte offset of the partition's bytes, from image base
  std::uint64_t count;         ///< number of records
  std::uint32_t schema_id;     ///< 0 = global reference; else local reference width
  std::uint32_t remap_count;   ///< entries in the remap table; 0 when schema_id == 0
};
static_assert(sizeof(dir_entry) == 32, "dir_entry must be padding-free");

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
