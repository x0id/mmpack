// Corruption harness: flip bytes in a valid image and confirm that whatever
// survives try_open() can still be searched, iterated and read field by field
// without leaving the region. Build with -fsanitize=address,undefined -- ASan is
// the actual oracle here, the checks below only keep the work from being
// optimized away.
//
//   c++ -std=c++20 -Iinclude -fsanitize=address,undefined tests/fuzz_open.cpp
//
// The schema is itself parsed from untrusted bytes, so a bad field offset, width or value
// reference would turn every record access into an out-of-bounds read. Corruption is
// therefore aimed at the schema and directory specifically, not just sprayed at random.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mmpack/mmpack.hpp"

namespace {

struct fields {
  mmpack::field_id country, region, population, temp, mac, ratio;
};

/// Fold a float into the checksum without a value conversion.
template <class T>
std::uint64_t bits_of(T v) noexcept {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(v));
  return bits;
}

std::vector<std::byte> valid_image(std::mt19937_64& rng, fields& ids,
                                   mmpack::vector_sink& sink) {
  static const char* countries[] = {"US", "FR", "DE", "JP", "BR"};

  mmpack::schema_builder sb;
  ids.country = sb.add_text("country");
  ids.region = sb.add_text("region");
  ids.population = sb.add_uint("population");
  ids.temp = sb.add_sint("temp_c");
  ids.mac = sb.add_bytes("mac", 6);  // an odd width, on purpose
  ids.ratio = sb.add_f64("ratio");

  mmpack::build_options options;
  options.address_bits = 4 + rng() % 8;
  // Cover both value modes, since interned adds a whole indirection to attack.
  options.value_interning = (rng() % 2) ? mmpack::interning_policy::always
                                        : mmpack::interning_policy::never;

  mmpack::table_builder tb(sb, options);
  const int n = 1 + static_cast<int>(rng() % 400);
  std::uint64_t key = 0;
  for (int i = 0; i < n; ++i) {
    key += 1 + rng() % 40;
    const unsigned t = static_cast<unsigned>(rng() % 12);
    auto rec = tb.begin_record(key);
    rec.set_text(ids.country, countries[t % 5]);
    rec.set_text(ids.region, "region-" + std::to_string(t % 4));
    rec.set_uint(ids.population, 1000 + t);
    rec.set_sint(ids.temp, -50 + static_cast<std::int64_t>(t));
    const std::byte mac[6] = {std::byte{1}, std::byte{2}, std::byte{3},
                              std::byte{4}, std::byte{5}, static_cast<std::byte>(t)};
    rec.set_bytes(ids.mac, std::span<const std::byte>(mac, 6));
    rec.set_f64(ids.ratio, static_cast<double>(t) / 3.0);
    tb.commit(rec);
  }
  tb.finish(sink);
  return sink.bytes();
}

/// Touch every access path the library exposes, so ASan sees any bad read.
std::uint64_t exercise(const mmpack::table& t, const fields& ids) {
  std::uint64_t sink = 0;

  for (auto it = t.begin(); it != t.end(); ++it) {
    sink += it.key() + it.partition() + it.address();
    if (auto v = t.uint(it, ids.population)) sink += *v;
    if (auto v = t.sint(it, ids.temp)) sink += static_cast<std::uint64_t>(*v);
    // Bit-cast rather than value-cast: corrupted bytes decode to arbitrary
    // doubles, and converting those to an integer would be UB in the harness.
    if (auto v = t.f64(it, ids.ratio)) sink += bits_of(*v);
    if (auto v = t.text(it, ids.country)) sink += v->size();
    if (auto v = t.text(it, ids.region)) sink += v->size();
    if (auto v = t.bytes(it, ids.mac)) {
      for (const std::byte b : *v) sink += static_cast<std::uint64_t>(b);
    }
  }

  auto it = t.end();
  while (it != t.begin()) {
    --it;
    sink += it.address();
  }

  // Field ids that may not exist in a corrupted schema, plus wild ones.
  for (mmpack::field_id id : {0u, 1u, 5u, 6u, 99u, 0xffffffffu}) {
    if (auto v = t.uint(t.begin(), id)) sink += *v;
    if (auto v = t.text(t.begin(), id)) sink += v->size();
    if (auto v = t.bytes(t.begin(), id)) sink += v->size();
    if (auto v = t.f32(t.begin(), id)) sink += bits_of(*v);
  }
  for (const char* name : {"country", "population", "mac", "missing"}) {
    if (auto id = t.field(name)) sink += *id;
  }

  // Probe the key space, including keys well past anything stored.
  std::uint64_t probe = 0;
  for (int i = 0; i < 200; ++i) {
    probe = probe * 6364136223846793005ull + 1442695040888963407ull;
    for (const std::uint64_t k : {probe, probe >> 32, std::uint64_t{0}, ~std::uint64_t{0}}) {
      if (auto lb = t.lower_bound(k); lb != t.end()) sink += lb.key();
      if (auto ub = t.upper_bound(k); ub != t.end()) sink += ub.address();
      if (auto f = t.find(k); f != t.end()) sink += f.key();
      sink += t.contains(k) ? 1 : 0;
      // floor() walks directory slots backwards, which no other entry point
      // does, so a corrupt directory has to be aimed at it directly.
      if (auto fl = t.floor(k); fl != t.end()) {
        sink += fl.key();
        if (auto v = t.uint(fl, ids.population)) sink += *v;
      }
      if (auto ce = t.ceil(k); ce != t.end()) sink += ce.address();
      if (auto fl = t.floor_at(k >> 8, k & 0xff); fl != t.end()) sink += fl.address();
      if (auto ce = t.ceil_at(k >> 8, k & 0xff); ce != t.end()) sink += ce.address();
    }
  }
  return sink;
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 20000;
  std::mt19937_64 rng(12345);

  std::uint64_t opened = 0, rejected = 0, sink = 0;
  fields ids{};
  mmpack::vector_sink holder;
  auto image = valid_image(rng, ids, holder);

  for (int round = 0; round < rounds; ++round) {
    if (round % 400 == 0) {
      holder.clear();
      image = valid_image(rng, ids, holder);
    }

    auto corrupt = image;
    const auto foot = mmpack::detail::load<mmpack::footer>(
        corrupt.data() + corrupt.size() - sizeof(mmpack::footer));

    const int flips = 1 + static_cast<int>(rng() % 4);
    for (int i = 0; i < flips; ++i) {
      std::size_t offset;
      const int where = static_cast<int>(rng() % 4);
      if (where == 0 && foot.schema_size > 0) {
        // The schema: field offsets, widths, kinds, dictionary pointers.
        offset = static_cast<std::size_t>(foot.schema_offset + rng() % foot.schema_size);
      } else if (where == 1) {
        // The tail: directory and footer.
        offset = corrupt.size() - 1 - static_cast<std::size_t>(rng() % 128);
      } else if (where == 2 && foot.dir_offset > 0) {
        // The records, which is where value references live.
        offset = static_cast<std::size_t>(rng() % foot.dir_offset);
      } else {
        offset = static_cast<std::size_t>(rng() % corrupt.size());
      }
      corrupt[offset] = static_cast<std::byte>(rng() & 0xff);
    }

    if (auto t = mmpack::table::try_open(corrupt.data(), corrupt.size())) {
      ++opened;
      sink += exercise(*t, ids);
    } else {
      ++rejected;
    }
  }

  std::printf("fuzz_open: %d rounds, %llu opened, %llu rejected (checksum %llu)\n", rounds,
              static_cast<unsigned long long>(opened),
              static_cast<unsigned long long>(rejected),
              static_cast<unsigned long long>(sink));
  if (opened == 0) {
    std::printf("  WARNING: nothing survived corruption; the fuzzer proved nothing\n");
    return 1;
  }
  return 0;
}
