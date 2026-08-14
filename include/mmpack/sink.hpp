#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace mmpack {

/// Everything the builder needs from an output: append bytes, in order, once.
/// No tell, no seek -- the format is designed so none is ever required.
template <class S>
concept byte_sink = requires(S& s, const void* data, std::size_t n) {
  { s.write(data, n) };
};

/// Collects the image in memory. Handy for tests and small images.
class vector_sink {
 public:
  void write(const void* data, std::size_t n) {
    const auto* p = static_cast<const std::byte*>(data);
    bytes_.insert(bytes_.end(), p, p + n);
  }

  [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
  [[nodiscard]] const std::byte* data() const noexcept { return bytes_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_.size(); }
  void clear() noexcept { bytes_.clear(); }

 private:
  std::vector<std::byte> bytes_;
};

/// Counts bytes without storing them. Run a build against this first to learn
/// the exact image size, then allocate and build again for real.
class counting_sink {
 public:
  void write(const void*, std::size_t n) noexcept { size_ += n; }
  [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
  void clear() noexcept { size_ = 0; }

 private:
  std::uint64_t size_ = 0;
};

/// Appends to a stdio stream. The stream stays owned by the caller.
class stdio_sink {
 public:
  explicit stdio_sink(std::FILE* file) noexcept : file_(file) {}

  void write(const void* data, std::size_t n) {
    if (n == 0) return;
    if (std::fwrite(data, 1, n, file_) != n) throw std::runtime_error("mmpack: short write");
  }

 private:
  std::FILE* file_;
};

/// Appends to a std::ostream. The stream stays owned by the caller.
class ostream_sink {
 public:
  explicit ostream_sink(std::ostream& out) noexcept : out_(&out) {}

  void write(const void* data, std::size_t n) {
    out_->write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!*out_) throw std::runtime_error("mmpack: short write");
  }

 private:
  std::ostream* out_;
};

/// Writes straight into a caller-provided buffer, e.g. a writable mapping sized
/// by a prior counting_sink pass. Throws if the build overruns the buffer.
class buffer_sink {
 public:
  buffer_sink(void* base, std::size_t capacity) noexcept
      : base_(static_cast<std::byte*>(base)), capacity_(capacity) {}

  void write(const void* data, std::size_t n) {
    if (n > capacity_ - offset_) throw std::runtime_error("mmpack: buffer overrun");
    std::memcpy(base_ + offset_, data, n);
    offset_ += n;
  }

  [[nodiscard]] std::size_t size() const noexcept { return offset_; }

 private:
  std::byte* base_;
  std::size_t capacity_;
  std::size_t offset_ = 0;
};

}  // namespace mmpack
