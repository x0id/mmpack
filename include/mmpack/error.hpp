#pragma once

#include <stdexcept>
#include <string>

namespace mmpack {

/// Why a mapped region was rejected by table::try_open().
enum class status : int {
  ok = 0,
  too_small,        ///< region cannot even hold a header and a footer
  bad_magic,        ///< not an mmpack image
  bad_version,      ///< written by a different format version
  bad_size,         ///< region length is not the exact image size, or the footer is corrupt
  bad_directory,    ///< directory is out of bounds, unsorted, or overlapping
  bad_schema,       ///< schema is out of bounds or internally inconsistent
  bad_dictionary,   ///< a dictionary is out of bounds or internally inconsistent
  bad_user_region,  ///< user region escapes the image or overlaps the records
};

[[nodiscard]] inline const char* to_string(status s) noexcept {
  switch (s) {
    case status::ok:              return "ok";
    case status::too_small:       return "region too small";
    case status::bad_magic:       return "bad magic";
    case status::bad_version:     return "unsupported format version";
    case status::bad_size:        return "image size does not match region length";
    case status::bad_directory:   return "corrupt partition directory";
    case status::bad_schema:      return "corrupt or inconsistent schema";
    case status::bad_dictionary:  return "corrupt dictionary";
    case status::bad_user_region: return "corrupt user region";
  }
  return "unknown";
}

/// Thrown by table::open(). Use try_open() for a non-throwing check.
class format_error : public std::runtime_error {
 public:
  explicit format_error(status s)
      : std::runtime_error(std::string("mmpack: ") + to_string(s)), status_(s) {}

  [[nodiscard]] status code() const noexcept { return status_; }

 private:
  status status_;
};

/// Thrown by the builder: bad schema construction, out-of-order input, or a
/// value that does not fit its declared field.
class build_error : public std::logic_error {
 public:
  explicit build_error(const std::string& what) : std::logic_error("mmpack: " + what) {}
};

}  // namespace mmpack
