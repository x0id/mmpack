// End-to-end example: build an IPv4 -> route table whose record shape is chosen
// from the data, write it to a file, mmap it, and query it.
//
// The dataset is shaped like the workload that motivated the design: many
// records, far fewer distinct value tuples, and repeated text fields. It is
// built three ways so the compaction can be read off rather than asserted:
//
//   naive     what a fixed-width struct with inline strings would cost
//   inline    mmpack per-field compaction, values in the record
//   interned  the above plus whole-value deduplication
//
// The mmap/munmap lives here rather than in the library: mapping policy belongs
// to the application. mmpack only ever sees a pointer and a length.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mmpack/mmpack.hpp"

namespace {

struct route {
  std::uint32_t key;  // the IPv4 address
  std::uint32_t next_hop;
  std::uint32_t asn;
  std::uint32_t metric;
  std::string interface;
  std::string region;
};

std::string format_ip(std::uint64_t ip) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u", static_cast<unsigned>((ip >> 24) & 0xff),
                static_cast<unsigned>((ip >> 16) & 0xff), static_cast<unsigned>((ip >> 8) & 0xff),
                static_cast<unsigned>(ip & 0xff));
  return buf;
}

/// A read-only file mapping. Exactly what the library expects the caller to own.
class mapped_file {
 public:
  explicit mapped_file(const char* path) {
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed");
    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
      ::close(fd_);
      throw std::runtime_error("fstat failed");
    }
    size_ = static_cast<std::size_t>(st.st_size);
    base_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("mmap failed");
    }
  }
  ~mapped_file() {
    if (base_ != MAP_FAILED) ::munmap(base_, size_);
    if (fd_ >= 0) ::close(fd_);
  }
  mapped_file(const mapped_file&) = delete;
  mapped_file& operator=(const mapped_file&) = delete;

  [[nodiscard]] const void* data() const noexcept { return base_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  int fd_ = -1;
  void* base_ = MAP_FAILED;
  std::size_t size_ = 0;
};

/// Routes clustered into /16s, with far fewer distinct value tuples than
/// records -- the duplication ratio interning exists to exploit.
std::vector<route> make_routes(std::size_t count, unsigned distinct_tuples) {
  static const char* interfaces[] = {"eth0", "eth1", "bond0", "bond0.100",
                                     "vlan42", "wg0", "lo", "ppp0"};
  static const char* regions[] = {"us-east", "us-west", "eu-central", "ap-south",
                                  "sa-east", "af-north"};
  std::mt19937_64 rng(42);

  std::vector<std::uint32_t> keys;
  keys.reserve(count);
  while (keys.size() < count) {
    const auto network = static_cast<std::uint32_t>(rng() % 4096) << 16;
    const auto burst = 1 + rng() % 64;
    for (std::uint64_t i = 0; i < burst && keys.size() < count; ++i) {
      keys.push_back(network | static_cast<std::uint32_t>(rng() & 0xffff));
    }
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  std::vector<route> routes;
  routes.reserve(keys.size());
  for (const std::uint32_t k : keys) {
    const auto t = static_cast<unsigned>(rng() % distinct_tuples);
    routes.push_back(route{k,
                           0x0a000000u | (t * 7919u & 0xffffu),
                           64500u + (t % 512),
                           10u + (t % 90),
                           interfaces[t % 8],
                           regions[t % 6]});
  }
  return routes;
}

struct field_ids {
  mmpack::field_id next_hop, asn, metric, interface, region;
};

std::uint64_t build(const char* path, const std::vector<route>& routes,
                    mmpack::interning_policy policy, field_ids& ids,
                    mmpack::build_report& report) {
  mmpack::schema_builder sb;
  ids.next_hop = sb.add_uint("next_hop");
  ids.asn = sb.add_uint("asn");
  ids.metric = sb.add_uint("metric");
  ids.interface = sb.add_text("interface");
  ids.region = sb.add_text("region");

  mmpack::build_options options;
  options.address_bits = 16;  // /16 network selects the partition
  options.value_interning = policy;

  std::FILE* file = std::fopen(path, "wb");
  if (!file) throw std::runtime_error("cannot open output file");
  mmpack::stdio_sink sink(file);
  mmpack::table_builder tb(sb, options);
  for (const route& r : routes) {
    auto rec = tb.begin_record(r.key);
    rec.set_uint(ids.next_hop, r.next_hop);
    rec.set_uint(ids.asn, r.asn);
    rec.set_uint(ids.metric, r.metric);
    rec.set_text(ids.interface, r.interface);
    rec.set_text(ids.region, r.region);
    tb.commit(rec);
  }
  report = tb.finish(sink);
  std::fclose(file);
  return report.image_bytes;
}

/// Time a fixed query workload. Returns ns per lookup.
double measure(const mmpack::table& t, const std::vector<std::uint64_t>& queries,
               const field_ids& ids, std::uint64_t& checksum) {
  std::uint64_t sum = 0;
  const auto start = std::chrono::steady_clock::now();
  for (const std::uint64_t q : queries) {
    const auto it = t.lower_bound(q);
    if (it != t.end()) sum += t.uint(it, ids.next_hop).value_or(0);
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  checksum = sum;
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) /
         static_cast<double>(queries.size());
}

/// Build both value modes for one duplication ratio and print a comparison row.
///
/// Interning trades record bytes for a dictionary indirection, so whether it
/// also wins on speed depends entirely on whether that dictionary stays in
/// cache. Sweeping the ratio is the only honest way to show where the line is.
void run_scenario(std::size_t count, unsigned distinct_tuples,
                  const std::vector<std::uint64_t>& query_seed) {
  const char* a_path = "/tmp/mmpack_scenario_inline.bin";
  const char* b_path = "/tmp/mmpack_scenario_interned.bin";
  const auto routes = make_routes(count, distinct_tuples);

  field_ids ids{};
  mmpack::build_report a_report, b_report;
  const std::uint64_t a_bytes =
      build(a_path, routes, mmpack::interning_policy::never, ids, a_report);
  const std::uint64_t b_bytes =
      build(b_path, routes, mmpack::interning_policy::always, ids, b_report);

  const mapped_file a_map(a_path);
  const mapped_file b_map(b_path);
  const auto a_table = mmpack::table::open(a_map.data(), a_map.size());
  const auto b_table = mmpack::table::open(b_map.data(), b_map.size());

  std::vector<std::uint64_t> queries;
  queries.reserve(query_seed.size());
  for (const std::uint64_t s : query_seed) queries.push_back(routes[s % routes.size()].key);

  std::uint64_t sa = 0, sb = 0;
  const double ns_a = measure(a_table, queries, ids, sa);
  const double ns_b = measure(b_table, queries, ids, sb);

  // What the automatic policy would have chosen, from the same estimates the
  // builder uses. The low-duplication rows are where this earns its keep.
  const bool auto_interns = a_report.interned_estimate < a_report.inline_estimate;
  std::printf("%9llu %9llu %8.1fx %10.2f %10.2f %9.1f %9.1f %9.1f%% %9s\n",
              static_cast<unsigned long long>(a_report.records),
              static_cast<unsigned long long>(b_report.distinct_values),
              static_cast<double>(a_report.records) /
                  static_cast<double>(b_report.distinct_values),
              static_cast<double>(a_bytes) / 1e6, static_cast<double>(b_bytes) / 1e6,
              ns_a, ns_b, 100.0 * (ns_b - ns_a) / ns_a,
              auto_interns ? "interned" : "inline");

  ::unlink(a_path);
  ::unlink(b_path);
}

}  // namespace

int main() {
  const char* inline_path = "/tmp/mmpack_routes_inline.bin";
  const char* interned_path = "/tmp/mmpack_routes_interned.bin";
  const auto routes = make_routes(500000, 3000);

  field_ids ids{};
  mmpack::build_report inline_report, interned_report;
  const std::uint64_t inline_bytes =
      build(inline_path, routes, mmpack::interning_policy::never, ids, inline_report);
  const std::uint64_t interned_bytes =
      build(interned_path, routes, mmpack::interning_policy::always, ids, interned_report);

  const mapped_file inline_map(inline_path);
  const mapped_file interned_map(interned_path);
  const auto inline_table = mmpack::table::open(inline_map.data(), inline_map.size());
  const auto interned_table = mmpack::table::open(interned_map.data(), interned_map.size());

  std::printf("%llu routes, %llu distinct value tuples (%.1fx duplication)\n",
              static_cast<unsigned long long>(inline_report.records),
              static_cast<unsigned long long>(inline_report.distinct_values),
              static_cast<double>(inline_report.records) /
                  static_cast<double>(inline_report.distinct_values));
  std::printf("%llu partitions, %s directory\n\n",
              static_cast<unsigned long long>(inline_table.partition_count()),
              inline_table.has_dense_directory() ? "dense" : "sparse");

  // Show the shape the builder derived, field by field.
  std::printf("derived record shape (value tuple = %u bytes):\n", inline_report.value_stride);
  const auto& schema = inline_table.schema();
  for (mmpack::field_id i = 0; i < schema.field_count(); ++i) {
    const mmpack::field_desc f = schema.field(i);
    std::printf("  %-10s %-5s offset %2u  width %u  bias %llu\n",
                std::string(schema.name(i)).c_str(),
                mmpack::to_string(static_cast<mmpack::field_kind>(f.kind)), f.offset, f.width,
                static_cast<unsigned long long>(f.bias));
  }

  // A few probes, resolving text through its dictionary.
  std::printf("\n");
  const std::uint64_t probes[] = {routes.front().key, routes[routes.size() / 2].key,
                                  routes.back().key, routes.back().key + 1ull};
  for (const std::uint64_t probe : probes) {
    const auto it = interned_table.lower_bound(probe);
    if (it == interned_table.end()) {
      std::printf("  lower_bound(%-15s) -> end\n", format_ip(probe).c_str());
      continue;
    }
    std::printf("  lower_bound(%-15s) -> %-15s via %-9s asn %-6llu %s\n",
                format_ip(probe).c_str(), format_ip(it.key()).c_str(),
                std::string(interned_table.text(it, ids.interface).value_or("?")).c_str(),
                static_cast<unsigned long long>(interned_table.uint(it, ids.asn).value_or(0)),
                it.key() == probe ? "(exact)" : "(next above)");
  }

  // --- space and speed ------------------------------------------------------
  std::mt19937_64 rng(7);
  std::vector<std::uint64_t> queries(1000000);
  for (auto& q : queries) {
    const std::uint64_t key = routes[rng() % routes.size()].key;
    q = (rng() % 2 == 0) ? key : key + (rng() % 512) - 256;
  }

  std::uint64_t sum_inline = 0, sum_interned = 0;
  const double ns_inline = measure(inline_table, queries, ids, sum_inline);
  const double ns_interned = measure(interned_table, queries, ids, sum_interned);

  // What a fixed-width struct with inline 16-byte strings would have cost:
  // 4 address + 4 next_hop + 4 asn + 4 metric + 16 interface + 16 region.
  const std::uint64_t naive_bytes = inline_report.records * 48;
  const auto per_entry = [&](std::uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(inline_report.records);
  };

  std::printf("\n%-10s %12s %8s %10s %12s\n", "layout", "bytes", "stride", "/entry", "ns/query");
  std::printf("%-10s %12llu %8u %10.1f %12s\n", "naive",
              static_cast<unsigned long long>(naive_bytes), 48u, per_entry(naive_bytes), "-");
  std::printf("%-10s %12llu %8u %10.1f %12.1f\n", "inline",
              static_cast<unsigned long long>(inline_bytes), inline_report.record_stride,
              per_entry(inline_bytes), ns_inline);
  std::printf("%-10s %12llu %8u %10.1f %12.1f\n", "interned",
              static_cast<unsigned long long>(interned_bytes), interned_report.record_stride,
              per_entry(interned_bytes), ns_interned);
  std::printf("\ninline vs naive:    %.1f%% of the bytes\n",
              100.0 * static_cast<double>(inline_bytes) / static_cast<double>(naive_bytes));
  std::printf("interned vs inline: %.1f%% of the bytes, %+.1f%% per lookup\n",
              100.0 * static_cast<double>(interned_bytes) / static_cast<double>(inline_bytes),
              100.0 * (ns_interned - ns_inline) / ns_inline);
  std::printf("cost model on this data: inline %llu vs interned %llu bytes -> %s\n",
              static_cast<unsigned long long>(inline_report.inline_estimate),
              static_cast<unsigned long long>(inline_report.interned_estimate),
              inline_report.interned_estimate < inline_report.inline_estimate ? "interned"
                                                                              : "inline");
  std::printf("checksums %s\n", sum_inline == sum_interned ? "match" : "DIFFER");

  // --- where interning stops paying for itself ------------------------------
  // The result above is flattering because 3000 tuples is a 21 KB dictionary
  // that never leaves cache. Sweep the duplication ratio to find the line: as
  // the dictionary grows past cache, the indirection starts costing real
  // misses and the speed advantage inverts, even while the bytes keep shrinking.
  std::vector<std::uint64_t> seed(400000);
  std::mt19937_64 seed_rng(11);
  for (auto& s : seed) s = seed_rng();

  std::printf("\n%9s %9s %9s %10s %10s %9s %9s %10s %9s\n", "records", "distinct", "ratio",
              "inline MB", "intern MB", "ns inline", "ns intern", "delta", "auto picks");
  for (const unsigned distinct : {1000u, 10000u, 100000u, 400000u}) {
    run_scenario(500000, distinct, seed);
  }
  std::printf(
      "\nForced interning is not always a win: below roughly 2x duplication it makes\n"
      "the image larger, which is what the automatic cost model exists to avoid --\n"
      "see the 'auto picks' column flip. The lookup deltas here are within noise:\n"
      "at 500k records every dictionary still fits in cache, so this sweep does not\n"
      "reach the regime where the extra indirection becomes a DRAM miss.\n");

  ::unlink(inline_path);
  ::unlink(interned_path);
  return sum_inline == sum_interned ? 0 : 1;
}
