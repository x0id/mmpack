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
inline constexpr std::uint32_t format_version = 1;

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

struct dir_entry {
  std::uint64_t partition;   ///< partition index
  std::uint64_t offset;      ///< byte offset of first record, from image base
  std::uint64_t count;       ///< number of records
  std::uint32_t schema_id;   ///< reserved for per-partition shapes; 0 = global
  std::uint32_t reserved;
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
