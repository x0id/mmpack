// Self-contained test suite: no external framework, just CHECK macros.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "mmpack/mmpack.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(expr)                                                         \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(expr)) {                                                          \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);         \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

#define CHECK_THROWS(expr, exception_type)                                  \
  do {                                                                      \
    ++g_checks;                                                             \
    bool caught = false;                                                    \
    try {                                                                   \
      expr;                                                                 \
    } catch (const exception_type&) {                                       \
      caught = true;                                                        \
    } catch (...) {                                                         \
    }                                                                       \
    if (!caught) {                                                          \
      std::printf("  FAIL %s:%d: expected %s from %s\n", __FILE__,          \
                  __LINE__, #exception_type, #expr);                        \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

void run(const char* name, void (*fn)()) {
  const int before = g_failures;
  std::printf("[ run  ] %s\n", name);
  fn();
  if (g_failures == before) std::printf("[  ok  ] %s\n", name);
}

// --- fixtures ---------------------------------------------------------------

struct row {
  std::uint64_t key;
  std::string country;
  std::string region;
  std::uint64_t population;
  std::int64_t temp;
};

/// Rows with heavy tuple duplication, which is what interning exists for.
std::vector<row> make_rows(std::size_t count, unsigned distinct_tuples, std::uint64_t seed = 7) {
  static const char* countries[] = {"US", "FR", "DE", "JP", "BR"};
  static const char* regions[] = {"north", "south", "east", "west", "central"};
  std::mt19937_64 rng(seed);
  std::vector<row> rows;
  rows.reserve(count);
  std::uint64_t key = 0;
  for (std::size_t i = 0; i < count; ++i) {
    key += 1 + rng() % 5;  // strictly increasing
    const unsigned t = static_cast<unsigned>(rng() % distinct_tuples);
    rows.push_back(row{key, countries[t % 5], regions[(t / 5) % 5], 1000 + (t % 200),
                       -20 + static_cast<std::int64_t>(t % 50)});
  }
  return rows;
}

struct built {
  mmpack::vector_sink sink;
  mmpack::build_report report;
  mmpack::field_id country{}, region{}, population{}, temp{};
};

std::unique_ptr<built> build(const std::vector<row>& rows, mmpack::build_options options = {}) {
  auto out = std::make_unique<built>();
  mmpack::schema_builder sb;
  out->country = sb.add_text("country");
  out->region = sb.add_text("region");
  out->population = sb.add_uint("population");
  out->temp = sb.add_sint("temp_c");

  mmpack::table_builder tb(sb, options);
  for (const row& r : rows) {
    auto rec = tb.begin_record(r.key);
    rec.set_text(out->country, r.country);
    rec.set_text(out->region, r.region);
    rec.set_uint(out->population, r.population);
    rec.set_sint(out->temp, r.temp);
    tb.commit(rec);
  }
  out->report = tb.finish(out->sink);
  return out;
}

/// Reading and rewriting directory slots of a built image. The directory is
/// packed at data-derived widths, so tests cannot treat it as an array of
/// structs; they have to go through the same layout the reader uses.
struct dir_access {
  mmpack::footer foot{};
  mmpack::dir_layout layout{};
  unsigned address_width = 0;  ///< supplies the slot width when no slot stores one
};

dir_access open_directory(const std::vector<std::byte>& image) {
  const auto t = mmpack::table::open(image.data(), image.size());
  dir_access out;
  out.foot =
      mmpack::detail::load<mmpack::footer>(image.data() + image.size() - sizeof(mmpack::footer));
  out.layout = t.schema().directory_layout();
  out.address_width = t.address_width();
  return out;
}

mmpack::dir_entry get_slot(const std::vector<std::byte>& image, const dir_access& a,
                           std::uint64_t i) {
  return mmpack::decode_dir_entry(image.data() + a.foot.dir_offset + i * a.layout.stride, a.layout,
                                  i, a.address_width);
}

void put_slot(std::vector<std::byte>& image, const dir_access& a, std::uint64_t i,
              const mmpack::dir_entry& d) {
  mmpack::encode_dir_entry(image.data() + a.foot.dir_offset + i * a.layout.stride, a.layout, d);
}

/// Every row resolves back to exactly what went in.
void verify_contents(const mmpack::table& t, const built& b, const std::vector<row>& rows) {
  for (const row& r : rows) {
    const auto it = t.find(r.key);
    CHECK(it != t.end());
    if (it == t.end()) continue;
    CHECK(it.key() == r.key);
    CHECK(t.text(it, b.country).value_or("") == r.country);
    CHECK(t.text(it, b.region).value_or("") == r.region);
    CHECK(t.uint(it, b.population).value_or(0) == r.population);
    CHECK(t.sint(it, b.temp).value_or(0) == r.temp);
  }
}

// --- tests ------------------------------------------------------------------

void test_roundtrip_both_value_modes() {
  const auto rows = make_rows(2000, 25);

  mmpack::build_options inline_opts;
  inline_opts.address_bits = 8;
  inline_opts.value_interning = mmpack::interning_policy::never;

  mmpack::build_options interned_opts;
  interned_opts.address_bits = 8;
  interned_opts.value_interning = mmpack::interning_policy::always;

  const auto a = build(rows, inline_opts);
  const auto b = build(rows, interned_opts);

  const auto ta = mmpack::table::open(a->sink.data(), a->sink.size());
  const auto tb = mmpack::table::open(b->sink.data(), b->sink.size());

  CHECK(!ta.interned());
  CHECK(tb.interned());
  CHECK(ta.size() == rows.size());
  CHECK(tb.size() == rows.size());

  verify_contents(ta, *a, rows);
  verify_contents(tb, *b, rows);

  // The two layouts must agree on every query, key for key.
  auto ia = ta.begin();
  auto ib = tb.begin();
  for (; ia != ta.end() && ib != tb.end(); ++ia, ++ib) {
    CHECK(ia.key() == ib.key());
    CHECK(ta.text(ia, a->country) == tb.text(ib, b->country));
    CHECK(ta.uint(ia, a->population) == tb.uint(ib, b->population));
    CHECK(ta.sint(ia, a->temp) == tb.sint(ib, b->temp));
  }
  CHECK(ia == ta.end());
  CHECK(ib == tb.end());

  // With 25 distinct tuples across 2000 records, interning must be smaller.
  CHECK(b->sink.size() < a->sink.size());
}

void test_cost_model_decides() {
  // Heavy duplication: interning should win.
  {
    const auto rows = make_rows(5000, 20);
    const auto b = build(rows);
    CHECK(b->report.interned);
    CHECK(b->report.distinct_values <= 20);
    CHECK(b->report.interned_estimate < b->report.inline_estimate);
  }
  // Every record distinct: interning cannot pay, because the dictionary is as
  // large as the records it replaces.
  {
    std::vector<row> rows;
    for (std::uint64_t i = 0; i < 3000; ++i) {
      rows.push_back(row{i, "US", "north", 1000 + i, static_cast<std::int64_t>(i)});
    }
    const auto b = build(rows);
    CHECK(!b->report.interned);
    CHECK(b->report.distinct_values == rows.size());
    CHECK(b->report.inline_estimate <= b->report.interned_estimate);
  }
}

void test_width_and_bias_selection() {
  const auto widths = [](std::uint64_t lo, std::uint64_t hi) {
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::table_builder tb(sb, {});
    for (std::uint64_t i = 0; i < 2; ++i) {
      auto rec = tb.begin_record(i);
      rec.set_uint(f, i == 0 ? lo : hi);
      tb.commit(rec);
    }
    mmpack::vector_sink sink;
    return tb.finish(sink).value_stride;
  };

  // Width follows the *range*, not the magnitude: that is the bias at work.
  CHECK(widths(0, 255) == 1);
  CHECK(widths(0, 256) == 2);
  CHECK(widths(0, 65535) == 2);
  CHECK(widths(0, 65536) == 4);
  CHECK(widths(1000000, 1000255) == 1);      // huge values, tiny range
  CHECK(widths(1000000, 1000256) == 2);
  CHECK(widths(0, 0xffffffffull) == 4);
  CHECK(widths(0, 0x100000000ull) == 8);

  // Signed fields are biased the same way, so negatives cost nothing extra.
  mmpack::schema_builder sb;
  const auto f = sb.add_sint("v");
  mmpack::table_builder tb(sb, {});
  const std::int64_t values[] = {-1000000, -999999, -999900};
  for (std::uint64_t i = 0; i < 3; ++i) {
    auto rec = tb.begin_record(i);
    rec.set_sint(f, values[i]);
    tb.commit(rec);
  }
  mmpack::vector_sink sink;
  const auto rep = tb.finish(sink);
  CHECK(rep.value_stride == 1);  // range is 100

  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  for (std::uint64_t i = 0; i < 3; ++i) {
    CHECK(t.sint(t.find(i), id).value() == values[i]);
  }
}

void test_text_dictionary_widths() {
  // Cardinality drives the index width, and it is chosen after interning.
  const auto stride_for = [](int distinct) {
    mmpack::schema_builder sb;
    const auto f = sb.add_text("s");
    mmpack::table_builder tb(sb, {});
    for (int i = 0; i < distinct; ++i) {
      auto rec = tb.begin_record(static_cast<std::uint64_t>(i));
      rec.set_text(f, "value-" + std::to_string(i));
      tb.commit(rec);
    }
    mmpack::vector_sink sink;
    return tb.finish(sink).value_stride;
  };
  CHECK(stride_for(256) == 1);
  CHECK(stride_for(257) == 2);

  // Repeated text is stored once.
  mmpack::schema_builder sb;
  const auto f = sb.add_text("s");
  mmpack::table_builder tb(sb, {});
  for (std::uint64_t i = 0; i < 1000; ++i) {
    auto rec = tb.begin_record(i);
    rec.set_text(f, i % 2 ? "even-longer-repeated-string" : "short");
    tb.commit(rec);
  }
  mmpack::vector_sink sink;
  const auto rep = tb.finish(sink);
  CHECK(rep.value_stride == 1);
  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("s").value();
  for (std::uint64_t i = 0; i < 1000; ++i) {
    CHECK(t.text(t.find(i), id).value() == (i % 2 ? "even-longer-repeated-string" : "short"));
  }
  // 1000 records with two distinct strings must not cost 1000 copies. Stored
  // inline the text alone would be 16000 bytes; here the records are the floor
  // at 1000 * 3, and the two strings are held once.
  CHECK(sink.size() < 3500);
  CHECK(sink.size() >= 1000 * rep.record_stride);
}

void test_lookup_matches_std_map() {
  std::mt19937_64 rng(20260813);
  std::map<std::uint64_t, std::uint64_t> oracle;
  std::uint64_t key = 0;

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::build_options options;
  options.address_bits = 8;  // lots of small partitions, with gaps
  mmpack::table_builder tb(sb, options);
  for (int i = 0; i < 4000; ++i) {
    key += 1 + rng() % 400;  // leaves gaps within and between partitions
    const std::uint64_t v = rng() % 1000;
    auto rec = tb.begin_record(key);
    rec.set_uint(f, v);
    tb.commit(rec);
    oracle[key] = v;
  }
  mmpack::vector_sink sink;
  tb.finish(sink);
  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  CHECK(t.size() == oracle.size());

  const auto same = [&](mmpack::table::const_iterator got,
                        std::map<std::uint64_t, std::uint64_t>::const_iterator want) {
    if (want == oracle.end()) return got == t.end();
    return got != t.end() && got.key() == want->first && t.uint(got, id).value() == want->second;
  };

  std::vector<std::uint64_t> probes;
  for (const auto& [k, v] : oracle) {
    probes.push_back(k);
    probes.push_back(k - 1);
    probes.push_back(k + 1);
  }
  probes.push_back(0);
  probes.push_back(~std::uint64_t{0});

  for (const std::uint64_t probe : probes) {
    CHECK(same(t.lower_bound(probe), oracle.lower_bound(probe)));
    CHECK(same(t.upper_bound(probe), oracle.upper_bound(probe)));
    CHECK(same(t.find(probe), oracle.find(probe)));
    CHECK(t.contains(probe) == oracle.contains(probe));
  }

  // Forward and backward iteration match the oracle exactly.
  auto want = oracle.begin();
  for (auto it = t.begin(); it != t.end(); ++it, ++want) CHECK(it.key() == want->first);

  auto rwant = oracle.rbegin();
  auto it = t.end();
  while (it != t.begin()) {
    --it;
    CHECK(it.key() == rwant->first);
    ++rwant;
  }
  CHECK(rwant == oracle.rend());
}

void test_floor_and_ceil_match_oracle() {
  // floor(k) is the range-containment primitive; the oracle is prev(upper_bound).
  std::mt19937_64 rng(4242);
  std::map<std::uint64_t, std::uint64_t> oracle;
  std::uint64_t key = 0;

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::build_options options;
  options.address_bits = 8;  // many small partitions, with gaps between them
  mmpack::table_builder tb(sb, options);
  for (int i = 0; i < 3000; ++i) {
    key += 1 + rng() % 500;  // gaps within and across partitions
    const std::uint64_t v = rng() % 1000;
    auto rec = tb.begin_record(key);
    rec.set_uint(f, v);
    tb.commit(rec);
    oracle[key] = v;
  }
  mmpack::vector_sink sink;
  tb.finish(sink);
  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();

  std::vector<std::uint64_t> probes;
  for (const auto& [k, v] : oracle) {
    probes.push_back(k);
    probes.push_back(k - 1);
    probes.push_back(k + 1);
  }
  probes.push_back(0);
  probes.push_back(1);
  probes.push_back(~std::uint64_t{0});

  for (const std::uint64_t probe : probes) {
    // floor
    const auto got = t.floor(probe);
    const auto above = oracle.upper_bound(probe);
    if (above == oracle.begin()) {
      CHECK(got == t.end());
    } else {
      const auto want = std::prev(above);
      CHECK(got != t.end());
      if (got != t.end()) {
        CHECK(got.key() == want->first);
        CHECK(t.uint(got, id).value() == want->second);
        CHECK(got.key() <= probe);  // the defining property
      }
    }

    // ceil is lower_bound, and must agree with it exactly
    CHECK(t.ceil(probe) == t.lower_bound(probe));
    const auto c = t.ceil(probe);
    const auto want_c = oracle.lower_bound(probe);
    if (want_c == oracle.end()) {
      CHECK(c == t.end());
    } else {
      CHECK(c != t.end());
      if (c != t.end()) {
        CHECK(c.key() == want_c->first);
        CHECK(c.key() >= probe);
      }
    }

    // The pre-split forms must agree with the key forms.
    const std::uint64_t p = t.schema().partition_of(probe);
    const std::uint64_t a = t.schema().address_of(probe);
    CHECK(t.floor_at(p, a) == got);
    CHECK(t.ceil_at(p, a) == c);
  }
}

void test_floor_edge_cases() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");

  {  // empty table
    mmpack::table_builder tb(sb, {});
    mmpack::vector_sink sink;
    tb.finish(sink);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    CHECK(t.floor(0) == t.end());
    CHECK(t.floor(~std::uint64_t{0}) == t.end());
    CHECK(t.ceil(0) == t.end());
  }

  // Partitions 1 and 7 populated, everything between them empty. Exercises the
  // cross-partition retreat, which is the only path that leaves the slot the
  // search landed on.
  mmpack::build_options options;
  options.address_bits = 8;
  mmpack::table_builder tb(sb, options);
  for (std::uint64_t key : {0x0110ull, 0x0120ull, 0x0705ull}) {
    auto rec = tb.begin_record(key);
    rec.set_uint(f, key);
    tb.commit(rec);
  }
  mmpack::vector_sink sink;
  tb.finish(sink);
  const auto t = mmpack::table::open(sink.data(), sink.size());

  CHECK(t.floor(0x0110).key() == 0x0110);  // exact hit on the first key
  CHECK(t.floor(0x0111).key() == 0x0110);  // between keys in one partition
  CHECK(t.floor(0x0120).key() == 0x0120);  // exact hit
  CHECK(t.floor(0x0121).key() == 0x0120);  // past the last key of partition 1
  CHECK(t.floor(0x0400).key() == 0x0120);  // partition that does not exist
  CHECK(t.floor(0x0700).key() == 0x0120);  // partition 7 exists, but nothing <= 0x00
  CHECK(t.floor(0x0705).key() == 0x0705);  // exact hit in the last partition
  CHECK(t.floor(0x9999).key() == 0x0705);  // past everything -> the last element
  CHECK(t.floor(0x010f) == t.end());       // before the first element
  CHECK(t.floor(0x0000) == t.end());
  CHECK(t.floor(0x0100) == t.end());       // same partition, below the first address

  // ceil mirrors it.
  CHECK(t.ceil(0x0000).key() == 0x0110);
  CHECK(t.ceil(0x0121).key() == 0x0705);
  CHECK(t.ceil(0x0705).key() == 0x0705);
  CHECK(t.ceil(0x0706) == t.end());

  // A floor result is a normal, usable iterator: dereferenceable and movable.
  auto it = t.floor(0x0400);
  CHECK(t.uint(it, f).value() == 0x0120);
  ++it;
  CHECK(it.key() == 0x0705);
  --it;
  CHECK(it.key() == 0x0120);
}

void test_floor_sparse_directory() {
  // The sparse path uses a different slot search, so it needs its own coverage.
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::build_options options;
  options.address_bits = 8;
  mmpack::table_builder tb(sb, options);
  const std::vector<std::uint64_t> keys = {(5ull << 8) | 3, (1ull << 30) | 9,
                                           (1ull << 40) | 2};
  for (std::uint64_t k : keys) {
    auto rec = tb.begin_record(k);
    rec.set_uint(f, k);
    tb.commit(rec);
  }
  mmpack::vector_sink sink;
  tb.finish(sink);
  const auto t = mmpack::table::open(sink.data(), sink.size());
  CHECK(!t.has_dense_directory());

  CHECK(t.floor((5ull << 8) | 2) == t.end());       // below everything
  CHECK(t.floor((5ull << 8) | 3).key() == keys[0]);  // exact
  CHECK(t.floor((5ull << 8) | 4).key() == keys[0]);  // past the partition's end
  CHECK(t.floor((9ull << 8)).key() == keys[0]);      // partition between two present ones
  CHECK(t.floor((1ull << 30) | 9).key() == keys[1]);
  CHECK(t.floor((1ull << 40) | 1).key() == keys[1]);  // last partition, below its first
  CHECK(t.floor(~std::uint64_t{0}).key() == keys[2]);  // above everything
}

void test_cross_partition_fallthrough() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::build_options options;
  options.address_bits = 8;
  mmpack::table_builder tb(sb, options);
  for (std::uint64_t key : {0x0110ull, 0x0120ull, 0x0705ull}) {
    auto rec = tb.begin_record(key);
    rec.set_uint(f, key);
    tb.commit(rec);
  }
  mmpack::vector_sink sink;
  tb.finish(sink);
  const auto t = mmpack::table::open(sink.data(), sink.size());

  CHECK(t.lower_bound(0x0121).key() == 0x0705);  // past the end of partition 1
  CHECK(t.lower_bound(0x0400).key() == 0x0705);  // partition that does not exist
  CHECK(t.lower_bound(0x0000).key() == 0x0110);
  CHECK(t.lower_bound(0x0706) == t.end());
  CHECK(t.upper_bound(0x0705) == t.end());
  CHECK(t.upper_bound(0x0120).key() == 0x0705);
  CHECK(t.find(0x0121) == t.end());  // find must not fall through
  CHECK(t.find(0x0400) == t.end());
}

void test_input_ordering() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");

  // Out-of-order input fails fast, and the message names both keys.
  {
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record(10);
    a.set_uint(f, 1);
    tb.commit(a);
    auto b = tb.begin_record(5);
    b.set_uint(f, 2);
    CHECK_THROWS(tb.commit(b), mmpack::build_error);
  }
  // Duplicate keys are caught by the same adjacency check.
  {
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record(10);
    a.set_uint(f, 1);
    tb.commit(a);
    auto b = tb.begin_record(10);
    b.set_uint(f, 2);
    CHECK_THROWS(tb.commit(b), mmpack::build_error);
  }
  // Opt-in recovery sorts instead of throwing.
  {
    mmpack::build_options options;
    options.order = mmpack::input_order::sort_if_needed;
    mmpack::table_builder tb(sb, options);
    for (std::uint64_t key : {50ull, 10ull, 30ull, 20ull}) {
      auto rec = tb.begin_record(key);
      rec.set_uint(f, key * 2);
      tb.commit(rec);
    }
    mmpack::vector_sink sink;
    const auto rep = tb.finish(sink);
    CHECK(rep.sorted_during_finish);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    const auto id = t.field("v").value();
    std::vector<std::uint64_t> seen;
    for (auto it = t.begin(); it != t.end(); ++it) {
      seen.push_back(it.key().value());
      CHECK(t.uint(it, id).value() == it.key().value() * 2);
    }
    CHECK((seen == std::vector<std::uint64_t>{10, 20, 30, 50}));
  }
  // Duplicates still rejected on the sorting path.
  {
    mmpack::build_options options;
    options.order = mmpack::input_order::sort_if_needed;
    mmpack::table_builder tb(sb, options);
    for (std::uint64_t key : {50ull, 10ull, 50ull}) {
      auto rec = tb.begin_record(key);
      rec.set_uint(f, 1);
      tb.commit(rec);
    }
    mmpack::vector_sink sink;
    CHECK_THROWS(tb.finish(sink), mmpack::build_error);
  }
}

void test_all_field_kinds() {
  mmpack::schema_builder sb;
  const auto u = sb.add_uint("u");
  const auto s = sb.add_sint("s");
  const auto a = sb.add_f32("a");
  const auto d = sb.add_f64("d");
  const auto m = sb.add_bytes("mac", 6);
  const auto x = sb.add_text("x");

  const std::byte mac[6] = {std::byte{0xde}, std::byte{0xad}, std::byte{0xbe},
                            std::byte{0xef}, std::byte{0x00}, std::byte{0x11}};

  mmpack::table_builder tb(sb, {});
  auto rec = tb.begin_record(42);
  rec.set_uint(u, 12345);
  rec.set_sint(s, -9876);
  rec.set_f32(a, 1.5f);
  rec.set_f64(d, -2.25);
  rec.set_bytes(m, std::span<const std::byte>(mac, 6));
  rec.set_text(x, "hello");
  tb.commit(rec);

  mmpack::vector_sink sink;
  tb.finish(sink);
  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto it = t.find(42);
  CHECK(it != t.end());
  CHECK(t.uint(it, u).value() == 12345);
  CHECK(t.sint(it, s).value() == -9876);
  CHECK(t.f32(it, a).value() == 1.5f);
  CHECK(t.f64(it, d).value() == -2.25);
  CHECK(t.text(it, x).value() == "hello");
  const auto raw = t.bytes(it, m);
  CHECK(raw.has_value());
  if (raw) {
    CHECK(raw->size() == 6);
    CHECK(std::memcmp(raw->data(), mac, 6) == 0);
  }

  // Reading a field as the wrong kind is a nullopt, never a reinterpretation.
  CHECK(!t.uint(it, s).has_value());
  CHECK(!t.sint(it, u).has_value());
  CHECK(!t.text(it, u).has_value());
  CHECK(!t.f32(it, d).has_value());
  CHECK(!t.f64(it, a).has_value());
  CHECK(!t.bytes(it, u).has_value());
  CHECK(!t.uint(it, 999).has_value());  // field id out of range

  // Names round-trip, unknown names are absent.
  CHECK(t.field("mac").value() == m);
  CHECK(!t.field("nope").has_value());
  CHECK(t.schema().name(u) == "u");
  CHECK(t.schema().field_count() == 6);
}

void test_schema_build_errors() {
  mmpack::schema_builder sb;
  sb.add_uint("a");
  CHECK_THROWS(sb.add_uint("a"), mmpack::build_error);       // duplicate name
  CHECK_THROWS(sb.add_uint(""), mmpack::build_error);        // empty name
  CHECK_THROWS(sb.add_bytes("b", 0), mmpack::build_error);   // zero width
  CHECK_THROWS(sb.add_bytes("b", 9), mmpack::build_error);   // wider than 8

  mmpack::schema_builder empty;
  CHECK_THROWS(mmpack::table_builder(empty, {}), mmpack::build_error);

  // Setting a field through the wrong setter is caught at build time.
  mmpack::schema_builder s2;
  const auto u = s2.add_uint("u");
  const auto x = s2.add_text("x");
  mmpack::table_builder tb(s2, {});
  auto rec = tb.begin_record(1);
  CHECK_THROWS(rec.set_text(u, "no"), mmpack::build_error);
  CHECK_THROWS(rec.set_uint(x, 1), mmpack::build_error);
  CHECK_THROWS(rec.set_uint(99, 1), mmpack::build_error);

  // A bytes field demands exactly its declared width.
  mmpack::schema_builder s3;
  const auto b = s3.add_bytes("b", 4);
  mmpack::table_builder tb3(s3, {});
  auto r3 = tb3.begin_record(1);
  const std::byte three[3] = {};
  CHECK_THROWS(r3.set_bytes(b, std::span<const std::byte>(three, 3)), mmpack::build_error);
}

void test_empty_and_single() {
  mmpack::schema_builder sb;
  sb.add_uint("v");
  {
    mmpack::table_builder tb(sb, {});
    mmpack::vector_sink sink;
    const auto rep = tb.finish(sink);
    CHECK(rep.records == 0);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    CHECK(t.empty());
    CHECK(t.begin() == t.end());
    CHECK(t.lower_bound(0) == t.end());
    CHECK(t.find(0) == t.end());
    CHECK(t.partition_count() == 0);
  }
  {
    mmpack::table_builder tb(sb, {});
    auto rec = tb.begin_record(77);
    rec.set_uint(0, 5);
    tb.commit(rec);
    mmpack::vector_sink sink;
    tb.finish(sink);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    CHECK(t.size() == 1);
    CHECK(t.begin().key() == 77);
    auto it = t.end();
    --it;
    CHECK(it == t.begin());
    CHECK(t.lower_bound(78) == t.end());
  }
}

void test_dense_and_sparse_directories() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  const auto build_with = [&](const std::vector<std::uint64_t>& keys, unsigned bits) {
    auto sink = std::make_unique<mmpack::vector_sink>();
    mmpack::build_options options;
    options.address_bits = bits;
    mmpack::table_builder tb(sb, options);
    for (std::uint64_t k : keys) {
      auto rec = tb.begin_record(k);
      rec.set_uint(f, k);
      tb.commit(rec);
    }
    tb.finish(*sink);
    return sink;
  };

  {  // tight partition indices -> dense
    std::vector<std::uint64_t> keys;
    for (std::uint64_t p = 0; p < 8; ++p) keys.push_back((p << 8) | 1);
    const auto sink = build_with(keys, 8);
    const auto t = mmpack::table::open(sink->data(), sink->size());
    CHECK(t.has_dense_directory());
    CHECK(t.partition_count() == 8);
    for (std::uint64_t k : keys) CHECK(t.find(k) != t.end());
  }
  {  // one far-flung partition -> sparse, and no giant directory
    const std::vector<std::uint64_t> keys = {1ull, (1ull << 40) | 1};
    const auto sink = build_with(keys, 8);
    const auto t = mmpack::table::open(sink->data(), sink->size());
    CHECK(!t.has_dense_directory());
    CHECK(t.directory_slots() == 2);
    CHECK(sink->size() < 4096);
    CHECK(t.find((1ull << 40) | 1) != t.end());
    CHECK(t.lower_bound(5).key() == ((1ull << 40) | 1));
  }
}

void test_align_fields_option() {
  const auto stride = [](bool align) {
    mmpack::schema_builder sb;
    sb.add_bytes("a", 1);
    sb.add_uint("b");
    mmpack::build_options options;
    options.align_fields = align;
    mmpack::table_builder tb(sb, options);
    for (std::uint64_t i = 0; i < 2; ++i) {
      auto rec = tb.begin_record(i);
      const std::byte one[1] = {std::byte{7}};
      rec.set_bytes(0, std::span<const std::byte>(one, 1));
      rec.set_uint(1, i == 0 ? 0 : 100000);  // forces a 4-byte field
      tb.commit(rec);
    }
    mmpack::vector_sink sink;
    return tb.finish(sink).value_stride;
  };
  CHECK(stride(false) == 5);  // 1 + 4, packed
  CHECK(stride(true) == 8);   // 1, pad to 4, then 4
}

void test_dedup_give_up() {
  // All-distinct data past the sample point: the builder should abandon dedup
  // rather than keep paying for a hash index that collapses nothing.
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::build_options options;
  options.dedup_sample = 100;
  options.dedup_give_up_ratio = 0.9;
  mmpack::table_builder tb(sb, options);
  for (std::uint64_t i = 0; i < 500; ++i) {
    auto rec = tb.begin_record(i);
    rec.set_uint(f, i);  // every tuple distinct
    tb.commit(rec);
  }
  mmpack::vector_sink sink;
  const auto rep = tb.finish(sink);
  CHECK(rep.dedup_abandoned);
  CHECK(!rep.interned);

  // Correctness must be unaffected by the give-up.
  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  CHECK(t.size() == 500);
  for (std::uint64_t i = 0; i < 500; ++i) CHECK(t.uint(t.find(i), id).value() == i);

  // Duplicated data keeps dedup on.
  mmpack::table_builder tb2(sb, options);
  for (std::uint64_t i = 0; i < 500; ++i) {
    auto rec = tb2.begin_record(i);
    rec.set_uint(f, i % 3);
    tb2.commit(rec);
  }
  mmpack::vector_sink sink2;
  const auto rep2 = tb2.finish(sink2);
  CHECK(!rep2.dedup_abandoned);
  CHECK(rep2.distinct_values == 3);
}

/// Many distinct tuples globally (forcing a 2-byte global reference) but only a
/// handful in each partition (which fit a 1-byte local index) -- the shape
/// per-partition remap exists for.
std::pair<std::unique_ptr<mmpack::vector_sink>, mmpack::build_report> build_remap_case(
    bool remap, std::uint64_t partitions = 400, std::uint64_t per_partition = 100,
    std::uint64_t distinct_each = 10) {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  auto sink = std::make_unique<mmpack::vector_sink>();
  mmpack::build_options o;
  o.address_bits = 8;
  o.value_interning = mmpack::interning_policy::always;
  o.partition_remap = remap;
  mmpack::table_builder tb(sb, o);
  for (std::uint64_t p = 0; p < partitions; ++p) {
    for (std::uint64_t a = 0; a < per_partition; ++a) {
      auto rec = tb.begin_record((p << 8) | a);
      rec.set_uint(f, p * distinct_each + (a % distinct_each));
      tb.commit(rec);
    }
  }
  const auto report = tb.finish(*sink);
  return {std::move(sink), report};
}

void test_partition_remap() {
  auto [plain_sink, plain] = build_remap_case(false);
  auto [remap_sink, remapped] = build_remap_case(true);

  // The global reference must genuinely be wider than the local one, or the
  // test would prove nothing.
  CHECK(plain.ref_width == 2);
  CHECK(plain.remapped_partitions == 0);
  CHECK(remapped.remapped_partitions == 400);
  CHECK(remapped.remap_saved_bytes > 0);
  CHECK(remap_sink->size() < plain_sink->size());

  const auto a = mmpack::table::open(plain_sink->data(), plain_sink->size());
  const auto b = mmpack::table::open(remap_sink->data(), remap_sink->size());
  CHECK(a.size() == b.size());
  const auto fa = a.field("v").value();
  const auto fb = b.field("v").value();

  // Every record must resolve to the same value through both layouts.
  for (std::uint64_t p = 0; p < 400; ++p) {
    for (std::uint64_t addr = 0; addr < 100; ++addr) {
      const std::uint64_t key = (p << 8) | addr;
      const auto ia = a.find(key);
      const auto ib = b.find(key);
      CHECK(ia != a.end());
      CHECK(ib != b.end());
      if (ia == a.end() || ib == b.end()) continue;
      const std::uint64_t want = p * 10 + (addr % 10);
      CHECK(a.uint(ia, fa).value() == want);
      CHECK(b.uint(ib, fb).value() == want);
    }
  }

  // Iteration, and every search entry point, must agree key for key.
  auto ia = a.begin();
  auto ib = b.begin();
  for (; ia != a.end() && ib != b.end(); ++ia, ++ib) {
    CHECK(ia.key() == ib.key());
    CHECK(a.uint(ia, fa) == b.uint(ib, fb));
  }
  CHECK(ia == a.end());
  CHECK(ib == b.end());

  std::mt19937_64 rng(31337);
  for (int i = 0; i < 4000; ++i) {
    const std::uint64_t probe = rng() % (400ull << 8);
    const auto la = a.lower_bound(probe);
    const auto lb = b.lower_bound(probe);
    CHECK((la == a.end()) == (lb == b.end()));
    if (la != a.end() && lb != b.end()) CHECK(la.key() == lb.key());

    const auto flr_a = a.floor(probe);
    const auto flr_b = b.floor(probe);
    CHECK((flr_a == a.end()) == (flr_b == b.end()));
    if (flr_a != a.end() && flr_b != b.end()) {
      CHECK(flr_a.key() == flr_b.key());
      CHECK(a.uint(flr_a, fa) == b.uint(flr_b, fb));
    }
  }
}

void test_partition_remap_is_selective() {
  // A partition whose tuples are nearly all distinct cannot repay a remap table,
  // so the builder must leave it on the global reference. Mixed images are the
  // interesting case because the reader has to switch stride per partition.
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::build_options o;
  o.address_bits = 8;
  o.value_interning = mmpack::interning_policy::always;
  mmpack::table_builder tb(sb, o);

  std::map<std::uint64_t, std::uint64_t> oracle;
  for (std::uint64_t p = 0; p < 300; ++p) {
    for (std::uint64_t a = 0; a < 200; ++a) {
      // Even partitions repeat a few values; odd ones are all distinct.
      const std::uint64_t v = (p % 2 == 0) ? p * 1000 + (a % 4) : p * 1000 + a;
      const std::uint64_t key = (p << 8) | a;
      auto rec = tb.begin_record(key);
      rec.set_uint(f, v);
      tb.commit(rec);
      oracle[key] = v;
    }
  }
  const auto report = tb.finish(sink);

  // Some remapped, some not -- that is the mixed image we want to read back.
  CHECK(report.remapped_partitions > 0);
  CHECK(report.remapped_partitions < 300);
  CHECK(report.ref_width >= 2);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  CHECK(t.size() == oracle.size());
  for (const auto& [k, v] : oracle) {
    const auto it = t.find(k);
    CHECK(it != t.end());
    if (it != t.end()) CHECK(t.uint(it, id).value() == v);
  }
  auto want = oracle.begin();
  for (auto it = t.begin(); it != t.end(); ++it, ++want) {
    CHECK(it.key() == want->first);
    CHECK(t.uint(it, id).value() == want->second);
  }
}

// --- arbitrary keys ---------------------------------------------------------

/// A key too wide to pack into 64 bits, split into the two ordering coordinates
/// the image is actually built from.
struct ipv6 {
  std::uint64_t high;
  std::uint64_t low;
  bool operator==(const ipv6&) const = default;
  bool operator<(const ipv6& o) const {
    return high != o.high ? high < o.high : low < o.low;
  }
};

struct split_v6 {
  std::pair<std::uint64_t, std::uint64_t> operator()(const ipv6& k) const {
    return {k.high, k.low};
  }
};
struct join_v6 {
  ipv6 operator()(std::uint64_t p, std::uint64_t a) const { return ipv6{p, a}; }
};

void test_explicit_parts_wide_keys() {
  // 128-bit keys, split 64/64. Neither half fits alongside the other, so there
  // is no 64-bit key space at all -- which is the whole point of this mode.
  std::mt19937_64 rng(90210);
  std::map<ipv6, std::uint64_t> oracle;
  for (std::uint64_t p = 0; p < 50; ++p) {
    for (int i = 0; i < 20; ++i) {
      const std::uint64_t high = (p << 48) | 0x2001'0db8'0000'0000ull;
      const std::uint64_t low = rng();
      oracle[ipv6{high, low}] = rng() % 1000;
    }
  }

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (const auto& [k, v] : oracle) {  // std::map iterates in key order
    auto rec = tb.begin_record_at(k.high, k.low);
    rec.set_uint(f, v);
    tb.commit(rec);
  }
  const auto report = tb.finish(sink);
  CHECK(!report.has_key_mapping);
  CHECK(tb.mode() == mmpack::key_mode::explicit_);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  CHECK(!t.has_key_mapping());
  CHECK(t.size() == oracle.size());
  const auto id = t.field("v").value();

  // Every record resolves through the pre-split entry points.
  for (const auto& [k, v] : oracle) {
    const auto it = t.find_at(k.high, k.low);
    CHECK(it != t.end());
    if (it == t.end()) continue;
    CHECK(it.partition() == k.high);
    CHECK(it.address() == k.low);
    CHECK(t.uint(it, id).value() == v);
    CHECK(!it.key().has_value());  // no 64-bit key exists for this image
  }

  // Iteration is in (partition, address) order, which is the key order.
  auto want = oracle.begin();
  for (auto it = t.begin(); it != t.end(); ++it, ++want) {
    CHECK(it.partition() == want->first.high);
    CHECK(it.address() == want->first.low);
  }

  // floor_at against a std::map oracle, including probes between real keys.
  for (int i = 0; i < 2000; ++i) {
    const ipv6 probe{(rng() % 52) << 48 | 0x2001'0db8'0000'0000ull, rng()};
    const auto got = t.floor_at(probe.high, probe.low);
    const auto above = oracle.upper_bound(probe);
    if (above == oracle.begin()) {
      CHECK(got == t.end());
    } else {
      const auto expect = std::prev(above);
      CHECK(got != t.end());
      if (got != t.end()) {
        CHECK(got.partition() == expect->first.high);
        CHECK(got.address() == expect->first.low);
      }
    }
  }

  // The 64-bit key API finds nothing rather than searching a space that does
  // not exist. has_key_mapping() is what tells those two apart.
  CHECK(t.find(0) == t.end());
  CHECK(t.lower_bound(0) == t.end());
  CHECK(t.upper_bound(0) == t.end());
  CHECK(t.floor(~std::uint64_t{0}) == t.end());
  CHECK(t.ceil(0) == t.end());
  CHECK(!t.contains(0));
}

void test_keyed_wrapper() {
  std::vector<ipv6> keys;
  for (std::uint64_t p = 1; p <= 30; ++p) {
    for (std::uint64_t a = 0; a < 10; ++a) keys.push_back(ipv6{p * 1000, a * 7 + 1});
  }

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  mmpack::keyed_builder<ipv6, split_v6> kb(tb, split_v6{});
  for (std::size_t i = 0; i < keys.size(); ++i) {
    auto rec = kb.begin_record(keys[i]);
    rec.set_uint(f, i);
    kb.commit(rec);
  }
  tb.finish(sink);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  const mmpack::keyed_table<ipv6, split_v6, join_v6> kt(t, split_v6{}, join_v6{});

  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto it = kt.find(keys[i]);
    CHECK(it != kt.end());
    if (it == kt.end()) continue;
    CHECK(t.uint(it, id).value() == i);
    CHECK(kt.key(it) == keys[i]);  // the join reconstructs the original key
    CHECK(kt.contains(keys[i]));
  }

  // Bounds work through the wrapper too, on a key that is not stored.
  const ipv6 gap{15 * 1000, 4};  // between 1 and 8
  CHECK(kt.contains(gap) == false);
  const auto fl = kt.floor(gap);
  CHECK(fl != kt.end());
  if (fl != kt.end()) CHECK(kt.key(fl) == (ipv6{15 * 1000, 1}));
  const auto ce = kt.ceil(gap);
  CHECK(ce != kt.end());
  if (ce != kt.end()) CHECK(kt.key(ce) == (ipv6{15 * 1000, 8}));

  // Past everything.
  CHECK(kt.floor(ipv6{0, 0}) == kt.end());
  CHECK(kt.ceil(ipv6{~std::uint64_t{0}, 0}) == kt.end());

  // Without a join the wrapper still searches, but key() is not available --
  // the same arrangement mmseek had, where join_key was optional and key() was
  // gated on it.
  const mmpack::keyed_table<ipv6, split_v6> plain(t, split_v6{});
  CHECK(plain.find(keys[3]) != plain.end());
  static_assert(mmpack::keyed_table<ipv6, split_v6, join_v6>::joinable);
  static_assert(!mmpack::keyed_table<ipv6, split_v6>::joinable);
  static_assert(requires(const mmpack::keyed_table<ipv6, split_v6, join_v6>& k,
                         mmpack::table::const_iterator i) { k.key(i); });
  // The negative direction is not asserted here: Apple Clang 14 reports the
  // unsatisfied constraint as a hard error inside a requires-expression rather
  // than as a substitution failure. The guarantee still holds -- calling key()
  // without a join does not compile -- and `joinable` above records the intent.
}

void test_keyed_wrapper_wide() {
  // The wrapper should carry a wide split with no changes of its own: the span
  // overloads are picked by ordinary resolution.
  struct v6 {
    std::array<std::byte, 16> bytes{};
    bool operator<(const v6& o) const { return bytes < o.bytes; }
    bool operator==(const v6& o) const = default;
  };
  struct split_wide {
    std::pair<std::uint64_t, std::span<const std::byte>> operator()(const v6& k) const {
      // Partition on the first two bytes, address on the remaining fourteen.
      const std::uint64_t p = (static_cast<std::uint64_t>(k.bytes[0]) << 8) |
                              static_cast<std::uint64_t>(k.bytes[1]);
      return {p, std::span<const std::byte>(k.bytes.data() + 2, 14)};
    }
  };
  struct join_wide {
    v6 operator()(std::uint64_t p, std::span<const std::byte> a) const {
      v6 out{};
      out.bytes[0] = static_cast<std::byte>((p >> 8) & 0xff);
      out.bytes[1] = static_cast<std::byte>(p & 0xff);
      std::memcpy(out.bytes.data() + 2, a.data(), a.size());
      return out;
    }
  };

  std::mt19937_64 rng(777);
  std::vector<v6> keys;
  for (int i = 0; i < 400; ++i) {
    v6 k{};
    for (std::size_t b = 0; b < k.bytes.size(); ++b) {
      k.bytes[b] = static_cast<std::byte>(rng() & 0xff);
    }
    k.bytes[0] = static_cast<std::byte>(rng() % 8);  // a handful of partitions
    keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  mmpack::schema_builder sb2;
  const auto vf = sb2.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb2, {});
  mmpack::keyed_builder<v6, split_wide> kb(tb, split_wide{});
  for (std::size_t i = 0; i < keys.size(); ++i) {
    auto rec = kb.begin_record(keys[i]);
    rec.set_uint(vf, i);
    kb.commit(rec);
  }
  const auto report = tb.finish(sink);
  CHECK(report.address_width == 14);
  CHECK(!report.has_key_mapping);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  const mmpack::keyed_table<v6, split_wide, join_wide> kt(t, split_wide{}, join_wide{});

  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto it = kt.find(keys[i]);
    CHECK(it != kt.end());
    if (it == kt.end()) continue;
    CHECK(t.uint(it, id).value() == i);
    CHECK(kt.key(it) == keys[i]);  // the join round-trips a 128-bit key
  }
  // floor through the wrapper on a key that is not stored.
  v6 probe = keys[100];
  probe.bytes[15] = std::byte{0x00};
  const auto fl = kt.floor(probe);
  CHECK(fl != kt.end());
  if (fl != kt.end()) CHECK(!(probe < kt.key(fl)));
}

void test_key_modes() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");

  {  // mixing the two entry points in one build is refused
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record(10);
    a.set_uint(f, 1);
    tb.commit(a);
    CHECK_THROWS(tb.begin_record_at(1, 2), mmpack::build_error);
  }
  {  // and the other way round
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(1, 2);
    a.set_uint(f, 1);
    tb.commit(a);
    CHECK_THROWS(tb.begin_record(10), mmpack::build_error);
  }
  {  // ordering is on (partition, address): a partition going backwards throws
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(5, 1);
    a.set_uint(f, 1);
    tb.commit(a);
    auto b = tb.begin_record_at(4, 9);
    b.set_uint(f, 2);
    CHECK_THROWS(tb.commit(b), mmpack::build_error);
  }
  {  // as does an address going backwards inside one partition
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(5, 10);
    a.set_uint(f, 1);
    tb.commit(a);
    auto b = tb.begin_record_at(5, 3);
    b.set_uint(f, 2);
    CHECK_THROWS(tb.commit(b), mmpack::build_error);
  }
  {  // and an exact duplicate
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(5, 10);
    a.set_uint(f, 1);
    tb.commit(a);
    auto b = tb.begin_record_at(5, 10);
    b.set_uint(f, 2);
    CHECK_THROWS(tb.commit(b), mmpack::build_error);
  }
  {  // sort_if_needed recovers unordered caller-split input
    mmpack::build_options o;
    o.order = mmpack::input_order::sort_if_needed;
    mmpack::table_builder tb(sb, o);
    for (const auto& [p, a] : std::vector<std::pair<std::uint64_t, std::uint64_t>>{
             {9, 1}, {2, 5}, {9, 0}, {2, 1}}) {
      auto rec = tb.begin_record_at(p, a);
      rec.set_uint(f, p * 100 + a);
      tb.commit(rec);
    }
    mmpack::vector_sink sink;
    const auto rep = tb.finish(sink);
    CHECK(rep.sorted_during_finish);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    const auto id = t.field("v").value();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> seen;
    for (auto it = t.begin(); it != t.end(); ++it) {
      seen.push_back({it.partition(), it.address().value()});
      CHECK(t.uint(it, id).value() == it.partition() * 100 + it.address().value());
    }
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> want{{2, 1}, {2, 5}, {9, 0}, {9, 1}};
    CHECK(seen == want);
  }
  {  // an empty build has a key mapping: an empty table with a key space is the
     // less surprising of the two
    mmpack::table_builder tb(sb, {});
    mmpack::vector_sink sink;
    const auto rep = tb.finish(sink);
    CHECK(rep.has_key_mapping);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    CHECK(t.has_key_mapping());
    CHECK(t.begin() == t.end());
  }
}

void test_keyed_and_explicit_agree() {
  // The same data through both entry points must produce the same image, which
  // pins that the 64-bit key API really is only sugar over the parts.
  const unsigned bits = 8;
  const auto build_keyed = [&](mmpack::vector_sink& sink) {
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::build_options o;
    o.address_bits = bits;
    mmpack::table_builder tb(sb, o);
    for (std::uint64_t p = 0; p < 30; ++p) {
      for (std::uint64_t a = 0; a < 9; ++a) {
        auto rec = tb.begin_record((p << bits) | (a * 5));
        rec.set_uint(f, p * 31 + a);
        tb.commit(rec);
      }
    }
    return tb.finish(sink);
  };
  const auto build_explicit = [&](mmpack::vector_sink& sink) {
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::build_options o;
    o.address_bits = bits;
    mmpack::table_builder tb(sb, o);
    for (std::uint64_t p = 0; p < 30; ++p) {
      for (std::uint64_t a = 0; a < 9; ++a) {
        auto rec = tb.begin_record_at(p, a * 5);
        rec.set_uint(f, p * 31 + a);
        tb.commit(rec);
      }
    }
    return tb.finish(sink);
  };

  mmpack::vector_sink keyed, split;
  const auto ka = build_keyed(keyed);
  const auto kb = build_explicit(split);

  // Identical but for the one byte that records whether a key space exists.
  CHECK(keyed.size() == split.size());
  CHECK(ka.has_key_mapping);
  CHECK(!kb.has_key_mapping);
  CHECK(ka.record_stride == kb.record_stride);
  CHECK(ka.directory_stride == kb.directory_stride);

  const auto ta = mmpack::table::open(keyed.data(), keyed.size());
  const auto tb2 = mmpack::table::open(split.data(), split.size());
  const auto fa = ta.field("v").value();
  const auto fb = tb2.field("v").value();
  CHECK(ta.size() == tb2.size());

  auto ia = ta.begin();
  auto ib = tb2.begin();
  for (; ia != ta.end() && ib != tb2.end(); ++ia, ++ib) {
    CHECK(ia.partition() == ib.partition());
    CHECK(ia.address() == ib.address());
    CHECK(ta.uint(ia, fa) == tb2.uint(ib, fb));
    CHECK(ia.key().has_value());
    CHECK(!ib.key().has_value());
    CHECK(ia.key().value() == ((ia.partition() << bits) | ia.address().value()));
  }
  CHECK(ia == ta.end());
  CHECK(ib == tb2.end());
}

void test_key_mapping_sentinel_rejected() {
  // A shift is 0..64, so the sentinel must not be mistakable for one, and an
  // out-of-range shift must still be refused.
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (std::uint64_t i = 0; i < 20; ++i) {
    auto rec = tb.begin_record(i);
    rec.set_uint(f, i);
    tb.commit(rec);
  }
  tb.finish(sink);
  const auto image = sink.bytes();
  mmpack::status s = mmpack::status::ok;

  const auto patch_bits = [&](std::uint8_t bits) {
    auto corrupt = image;
    const auto foot =
        mmpack::detail::load<mmpack::footer>(corrupt.data() + corrupt.size() - sizeof(mmpack::footer));
    auto head = mmpack::detail::load<mmpack::schema_header>(corrupt.data() + foot.schema_offset);
    head.address_bits = bits;
    mmpack::detail::store(corrupt.data() + foot.schema_offset, head);
    return corrupt;
  };

  {  // the sentinel is accepted, and turns the image into a no-key-space one
    auto corrupt = patch_bits(mmpack::no_key_mapping);
    const auto t = mmpack::table::open(corrupt.data(), corrupt.size());
    CHECK(!t.has_key_mapping());
    CHECK(t.find(3) == t.end());          // no key space to search
    CHECK(t.find_at(0, 3) != t.end());    // the parts still work
    CHECK(!t.begin().key().has_value());
  }
  {  // a shift beyond 64 that is not the sentinel is still rejected
    auto corrupt = patch_bits(65);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // as is one just below the sentinel
    auto corrupt = patch_bits(0xfe);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
}

// --- addresses wider than 64 bits -------------------------------------------

using addr14 = std::array<std::byte, 14>;

/// Big endian, so lexicographic byte order is numeric order -- the encoding
/// contract wide addresses carry.
addr14 make_addr(std::uint64_t hi, std::uint64_t lo) {
  addr14 out{};
  for (int i = 0; i < 6; ++i) out[i] = static_cast<std::byte>((hi >> (8 * (5 - i))) & 0xff);
  for (int i = 0; i < 8; ++i) out[6 + i] = static_cast<std::byte>((lo >> (8 * (7 - i))) & 0xff);
  return out;
}

std::span<const std::byte> as_span(const addr14& a) { return {a.data(), a.size()}; }

void test_wide_addresses() {
  // A /16 partition with a 14-byte host part: the address alone is 112 bits, so
  // this is exactly the shape that did not fit before.
  std::mt19937_64 rng(6060);
  std::map<std::pair<std::uint64_t, addr14>, std::uint64_t> oracle;
  for (std::uint64_t p = 0; p < 40; ++p) {
    for (int i = 0; i < 25; ++i) oracle[{p, make_addr(rng(), rng())}] = rng() % 10000;
  }

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (const auto& [k, v] : oracle) {  // std::map orders by (partition, bytes)
    auto rec = tb.begin_record_at(k.first, as_span(k.second));
    rec.set_uint(f, v);
    tb.commit(rec);
  }
  const auto report = tb.finish(sink);
  CHECK(!report.has_key_mapping);
  CHECK(report.address_width == 14);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  CHECK(!t.narrow_address());
  CHECK(t.address_width() == 14);
  CHECK(t.size() == oracle.size());
  const auto id = t.field("v").value();

  for (const auto& [k, v] : oracle) {
    const auto it = t.find_at(k.first, as_span(k.second));
    CHECK(it != t.end());
    if (it == t.end()) continue;
    CHECK(it.partition() == k.first);
    CHECK(!it.address().has_value());  // does not fit an integer
    const auto bytes = it.address_bytes();
    CHECK(bytes.size() == 14);
    CHECK(std::memcmp(bytes.data(), k.second.data(), 14) == 0);
    CHECK(t.uint(it, id).value() == v);
  }

  auto want = oracle.begin();
  for (auto it = t.begin(); it != t.end(); ++it, ++want) {
    CHECK(it.partition() == want->first.first);
    CHECK(std::memcmp(it.address_bytes().data(), want->first.second.data(), 14) == 0);
  }

  for (int i = 0; i < 3000; ++i) {
    const std::uint64_t p = rng() % 42;
    const addr14 probe = make_addr(rng(), rng());
    const auto got = t.floor_at(p, as_span(probe));
    const auto above = oracle.upper_bound({p, probe});
    if (above == oracle.begin()) {
      CHECK(got == t.end());
    } else {
      const auto expect = std::prev(above);
      CHECK(got != t.end());
      if (got != t.end()) {
        CHECK(got.partition() == expect->first.first);
        CHECK(std::memcmp(got.address_bytes().data(), expect->first.second.data(), 14) == 0);
      }
    }
  }

  // The integer forms have nothing to search on a wide image, and a probe of the
  // wrong length is refused rather than compared short.
  CHECK(t.find_at(0, std::uint64_t{0}) == t.end());
  CHECK(t.lower_bound_at(0, std::uint64_t{0}) == t.end());
  CHECK(t.floor_at(0, ~std::uint64_t{0}) == t.end());
  const std::array<std::byte, 13> short_probe{};
  CHECK(t.find_at(0, std::span<const std::byte>(short_probe)) == t.end());
  CHECK(t.floor_at(0, std::span<const std::byte>(short_probe)) == t.end());
}

void test_wide_address_ordering_is_lexicographic() {
  // Chosen so a little-endian integer reading would order them the other way
  // round, which is what proves the comparison is memcmp on the bytes.
  addr14 a{};
  a[13] = std::byte{0x01};
  addr14 b{};
  b[12] = std::byte{0x01};
  addr14 c{};
  c[0] = std::byte{0x01};
  const std::vector<addr14> keys = {a, b, c};

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (std::size_t i = 0; i < keys.size(); ++i) {
    auto rec = tb.begin_record_at(0, as_span(keys[i]));
    rec.set_uint(f, i);
    tb.commit(rec);
  }
  tb.finish(sink);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  std::vector<std::uint64_t> order;
  for (auto it = t.begin(); it != t.end(); ++it) order.push_back(t.uint(it, id).value());
  CHECK((order == std::vector<std::uint64_t>{0, 1, 2}));

  addr14 probe = b;
  probe[13] = std::byte{0xff};  // between b and c lexicographically
  const auto fl = t.floor_at(0, as_span(probe));
  CHECK(fl != t.end());
  if (fl != t.end()) CHECK(t.uint(fl, id).value() == 1);
}

void test_wide_address_build_errors() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  const addr14 wide{};
  {  // a second record of a different width breaks the fixed stride
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(0, as_span(wide));
    a.set_uint(f, 1);
    tb.commit(a);
    const std::array<std::byte, 10> other{};
    CHECK_THROWS(tb.begin_record_at(0, std::span<const std::byte>(other)), mmpack::build_error);
  }
  {  // an empty address has nothing to order by
    mmpack::table_builder tb(sb, {});
    CHECK_THROWS(tb.begin_record_at(0, std::span<const std::byte>()), mmpack::build_error);
  }
  {  // wide and integer forms are different modes and cannot be mixed
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(0, as_span(wide));
    a.set_uint(f, 1);
    tb.commit(a);
    CHECK_THROWS(tb.begin_record_at(1, std::uint64_t{2}), mmpack::build_error);
  }
  {
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(0, std::uint64_t{1});
    a.set_uint(f, 1);
    tb.commit(a);
    CHECK_THROWS(tb.begin_record_at(0, as_span(wide)), mmpack::build_error);
  }
  {  // out-of-order and duplicate wide addresses are caught as narrow ones are
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(0, as_span(make_addr(5, 5)));
    a.set_uint(f, 1);
    tb.commit(a);
    auto b = tb.begin_record_at(0, as_span(make_addr(5, 4)));
    b.set_uint(f, 2);
    CHECK_THROWS(tb.commit(b), mmpack::build_error);
  }
  {
    mmpack::table_builder tb(sb, {});
    auto a = tb.begin_record_at(0, as_span(make_addr(5, 5)));
    a.set_uint(f, 1);
    tb.commit(a);
    auto b = tb.begin_record_at(0, as_span(make_addr(5, 5)));
    b.set_uint(f, 2);
    CHECK_THROWS(tb.commit(b), mmpack::build_error);
  }
  {  // sort_if_needed recovers unordered wide input
    mmpack::build_options o;
    o.order = mmpack::input_order::sort_if_needed;
    mmpack::table_builder tb(sb, o);
    for (std::uint64_t v : {30ull, 10ull, 20ull}) {
      auto rec = tb.begin_record_at(0, as_span(make_addr(0, v)));
      rec.set_uint(f, v);
      tb.commit(rec);
    }
    mmpack::vector_sink sink;
    const auto rep = tb.finish(sink);
    CHECK(rep.sorted_during_finish);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    const auto id = t.field("v").value();
    std::vector<std::uint64_t> seen;
    for (auto it = t.begin(); it != t.end(); ++it) seen.push_back(t.uint(it, id).value());
    CHECK((seen == std::vector<std::uint64_t>{10, 20, 30}));
  }
}

void test_odd_narrow_widths() {
  // 3, 5, 6 and 7 were only excluded by the old {1,2,4,8} check; the integer
  // path already handled them.
  for (const std::uint64_t top :
       {std::uint64_t{0xffffff}, std::uint64_t{0xffffffffff}, std::uint64_t{0xffffffffffff},
        std::uint64_t{0xffffffffffffff}}) {
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::vector_sink sink;
    mmpack::table_builder tb(sb, {});
    for (std::uint64_t i = 0; i < 8; ++i) {
      auto rec = tb.begin_record_at(0, top - (7 - i));
      rec.set_uint(f, i);
      tb.commit(rec);
    }
    const auto rep = tb.finish(sink);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    CHECK(t.narrow_address());
    CHECK(t.address_width() == rep.address_width);
    const auto id = t.field("v").value();
    for (std::uint64_t i = 0; i < 8; ++i) {
      const auto it = t.find_at(0, top - (7 - i));
      CHECK(it != t.end());
      if (it != t.end()) {
        CHECK(t.uint(it, id).value() == i);
        CHECK(it.address().value() == top - (7 - i));
      }
    }
  }
}

void test_span_overload_on_narrow_images() {
  // The span form is the general one and must serve narrow images too, so a
  // caller need not branch on the width.
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::build_options o;
  o.address_bits = 8;
  mmpack::table_builder tb(sb, o);
  for (std::uint64_t a = 0; a < 20; ++a) {
    auto rec = tb.begin_record((1ull << 8) | (a * 3));
    rec.set_uint(f, a);
    tb.commit(rec);
  }
  tb.finish(sink);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  CHECK(t.narrow_address());
  const auto id = t.field("v").value();
  const unsigned width = t.address_width();

  for (std::uint64_t a = 0; a < 20; ++a) {
    std::array<std::byte, 8> probe{};
    const std::uint64_t value = a * 3;
    std::memcpy(probe.data(), &value, sizeof(value));
    const std::span<const std::byte> as_bytes(probe.data(), width);

    const auto by_int = t.find_at(1, value);
    const auto by_span = t.find_at(1, as_bytes);
    CHECK(by_int != t.end());
    CHECK(by_int == by_span);
    if (by_span != t.end()) CHECK(t.uint(by_span, id).value() == a);
    CHECK(t.floor_at(1, value) == t.floor_at(1, as_bytes));
    CHECK(t.lower_bound_at(1, value) == t.lower_bound_at(1, as_bytes));
  }
  const std::array<std::byte, 7> wrong{};
  CHECK(t.find_at(1, std::span<const std::byte>(wrong)) == t.end());
}

void test_wide_address_corruption() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (std::uint64_t i = 0; i < 30; ++i) {
    auto rec = tb.begin_record_at(0, as_span(make_addr(0, i)));
    rec.set_uint(f, i);
    tb.commit(rec);
  }
  tb.finish(sink);
  const auto image = sink.bytes();
  mmpack::status s = mmpack::status::ok;

  const auto patch_schema = [&](auto mutate) {
    auto corrupt = image;
    const auto foot = mmpack::detail::load<mmpack::footer>(
        corrupt.data() + corrupt.size() - sizeof(mmpack::footer));
    auto head = mmpack::detail::load<mmpack::schema_header>(corrupt.data() + foot.schema_offset);
    mutate(head);
    mmpack::detail::store(corrupt.data() + foot.schema_offset, head);
    return corrupt;
  };

  {  // an address wider than the record that holds it
    auto corrupt = patch_schema([](mmpack::schema_header& h) { h.address_width = 200; });
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // a zero-width address has nothing to compare
    auto corrupt = patch_schema([](mmpack::schema_header& h) { h.address_width = 0; });
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // a key mapping cannot coexist with an address too wide to pack into one
    auto corrupt = patch_schema([](mmpack::schema_header& h) { h.address_bits = 16; });
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
}

// --- per-partition address trimming -----------------------------------------

using addr12 = std::array<std::byte, 12>;

/// Big-endian, so byte 0 is the most significant -- the encoding the wide
/// comparison requires.
addr12 make_addr12(std::initializer_list<unsigned> leading) {
  addr12 out{};
  std::size_t i = 0;
  for (const unsigned b : leading) out[i++] = static_cast<std::byte>(b);
  return out;
}

std::span<const std::byte> as_span12(const addr12& a) { return {a.data(), a.size()}; }

/// The case that is easiest to get exactly backwards, and that an earlier draft
/// of this design did: a little-endian integer address whose *low-order* bytes
/// are zero. That is what a table of region starts looks like, and those zeros
/// sit at the front of the field in memory -- so trimming them means skipping a
/// prefix, not shortening a tail. Trimming the tail instead would keep the zeros
/// and throw away the significant byte.
void test_narrow_low_order_trim() {
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> oracle;
  for (std::uint64_t i = 1; i <= 1000; ++i) oracle[{0, i << 16}] = i;         // /16-aligned
  for (std::uint64_t i = 1; i <= 1000; ++i) oracle[{1, i}] = i + 5000;        // unaligned
  oracle[{2, 0xff000000ull}] = 99;  // fixes the global width at 4

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (const auto& [k, v] : oracle) {
    auto rec = tb.begin_record_at(k.first, k.second);
    rec.set_uint(f, v);
    tb.commit(rec);
  }
  const auto report = tb.finish(sink);
  CHECK(report.address_width == 4);
  CHECK(report.narrowed_partitions == 3);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  CHECK(t.trims_addresses());
  CHECK(t.address_width() == 4);
  const auto id = t.field("v").value();

  // The stored range of each partition, read back through the real layout.
  const dir_access acc = open_directory(sink.bytes());
  const auto p0 = get_slot(sink.bytes(), acc, 0);
  const auto p1 = get_slot(sink.bytes(), acc, 1);
  const auto p2 = get_slot(sink.bytes(), acc, 2);
  CHECK(p0.address_skip == 2 && p0.address_width == 2);  // i << 16, i <= 1000
  CHECK(p1.address_skip == 0 && p1.address_width == 2);  // i <= 1000
  CHECK(p2.address_skip == 3 && p2.address_width == 1);  // 0xff000000

  // Storing two bytes where a global width would have stored four.
  CHECK(report.address_bytes_total == 2 * 1000 + 2 * 1000 + 1);
  CHECK(report.address_saved_bytes == 4 * oracle.size() - report.address_bytes_total);

  for (const auto& [k, v] : oracle) {
    const auto it = t.find_at(k.first, k.second);
    CHECK(it != t.end());
    if (it == t.end()) continue;
    CHECK(it.partition() == k.first);
    CHECK(it.address().value() == k.second);  // the trimmed bytes come back
    CHECK(t.uint(it, id).value() == v);
  }

  // Iteration order and every lookup form against the oracle.
  auto want = oracle.begin();
  for (auto it = t.begin(); it != t.end(); ++it, ++want) {
    CHECK(it.partition() == want->first.first);
    CHECK(it.address().value() == want->first.second);
  }
  CHECK(want == oracle.end());

  std::mt19937_64 rng(4242);
  for (int i = 0; i < 4000; ++i) {
    const std::uint64_t p = rng() % 4;
    const std::uint64_t a = (rng() % 4 == 0) ? (rng() % 1200) << 16 : rng() % 0x1000000ull;

    const auto lb = t.lower_bound_at(p, a);
    const auto expect_lb = oracle.lower_bound({p, a});
    CHECK((lb == t.end()) == (expect_lb == oracle.end()));
    if (lb != t.end() && expect_lb != oracle.end()) {
      CHECK(lb.partition() == expect_lb->first.first);
      CHECK(lb.address().value() == expect_lb->first.second);
    }

    const auto fl = t.floor_at(p, a);
    const auto above = oracle.upper_bound({p, a});
    if (above == oracle.begin()) {
      CHECK(fl == t.end());
    } else {
      const auto expect = std::prev(above);
      CHECK(fl != t.end());
      if (fl != t.end()) {
        CHECK(fl.partition() == expect->first.first);
        CHECK(fl.address().value() == expect->first.second);
      }
    }
  }
}

/// The wide half of the same rule, on IPv6-shaped data: a /32 partition with a
/// 12-byte remainder, where region starts leave the tail zero.
void test_wide_partition_trim() {
  std::map<std::pair<std::uint64_t, addr12>, std::uint64_t> oracle;
  std::mt19937_64 rng(31337);

  // /64 starts: only the first four bytes ever vary.
  for (unsigned i = 1; i <= 300; ++i) {
    oracle[{0, make_addr12({i >> 8, i & 0xff, static_cast<unsigned>(rng() & 0xff),
                            static_cast<unsigned>(rng() & 0xff)})}] = i;
  }
  // Full depth: every byte varies, so nothing can be trimmed.
  for (unsigned i = 0; i < 300; ++i) {
    addr12 a{};
    for (auto& b : a) b = static_cast<std::byte>(rng() & 0xff);
    oracle[{1, a}] = 1000 + i;
  }
  // Trimmed at both ends at once: bytes 0-1 zero, 4-11 zero.
  for (unsigned i = 1; i <= 300; ++i) {
    addr12 a{};
    a[2] = static_cast<std::byte>(i >> 8);
    a[3] = static_cast<std::byte>(i & 0xff);
    oracle[{2, a}] = 2000 + i;
  }

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (const auto& [k, v] : oracle) {
    auto rec = tb.begin_record_at(k.first, as_span12(k.second));
    rec.set_uint(f, v);
    tb.commit(rec);
  }
  const auto report = tb.finish(sink);
  CHECK(report.address_width == 12);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  CHECK(!t.narrow_address());
  CHECK(t.trims_addresses());
  const auto id = t.field("v").value();

  const dir_access acc = open_directory(sink.bytes());
  const auto p0 = get_slot(sink.bytes(), acc, 0);
  const auto p1 = get_slot(sink.bytes(), acc, 1);
  const auto p2 = get_slot(sink.bytes(), acc, 2);
  CHECK(p0.address_skip == 0 && p0.address_width == 4);
  CHECK(p1.address_skip == 0 && p1.address_width == 12);
  CHECK(p2.address_skip == 2 && p2.address_width == 2);  // both ends trimmed

  for (const auto& [k, v] : oracle) {
    const auto it = t.find_at(k.first, as_span12(k.second));
    CHECK(it != t.end());
    if (it == t.end()) continue;
    CHECK(t.uint(it, id).value() == v);
    // address_bytes() is only the stored stretch; address_into() is the field.
    CHECK(it.address_bytes().size() ==
          (k.first == 0 ? 4u : k.first == 1 ? 12u : 2u));
    CHECK(it.address_skip() == (k.first == 2 ? 2u : 0u));
    addr12 full{};
    CHECK(it.address_into(std::span<std::byte>(full.data(), full.size())));
    CHECK(full == k.second);
  }

  // Every ordered query form against the oracle, including probes that fall in
  // the trimmed regions.
  for (int i = 0; i < 5000; ++i) {
    const std::uint64_t p = rng() % 4;
    addr12 probe{};
    switch (rng() % 3) {
      case 0:  // deep in the tail: exercises the tail tiebreak
        probe[0] = static_cast<std::byte>(rng() & 0xff);
        probe[1] = static_cast<std::byte>(rng() & 0xff);
        probe[11] = static_cast<std::byte>(rng() | 1u);
        break;
      case 1:  // high in the head: exercises the head shortcut
        for (auto& b : probe) b = static_cast<std::byte>(rng() & 0xff);
        break;
      default:
        probe[2] = static_cast<std::byte>(rng() & 0xff);
        probe[3] = static_cast<std::byte>(rng() & 0xff);
        break;
    }

    const auto lb = t.lower_bound_at(p, as_span12(probe));
    const auto expect_lb = oracle.lower_bound({p, probe});
    CHECK((lb == t.end()) == (expect_lb == oracle.end()));
    if (lb != t.end() && expect_lb != oracle.end()) {
      addr12 got{};
      CHECK(lb.address_into(std::span<std::byte>(got.data(), got.size())));
      CHECK(lb.partition() == expect_lb->first.first);
      CHECK(got == expect_lb->first.second);
    }

    const auto ub = t.upper_bound_at(p, as_span12(probe));
    const auto expect_ub = oracle.upper_bound({p, probe});
    CHECK((ub == t.end()) == (expect_ub == oracle.end()));
    if (ub != t.end() && expect_ub != oracle.end()) {
      addr12 got{};
      CHECK(ub.address_into(std::span<std::byte>(got.data(), got.size())));
      CHECK(ub.partition() == expect_ub->first.first);
      CHECK(got == expect_ub->first.second);
    }

    const auto found = t.find_at(p, as_span12(probe));
    CHECK((found != t.end()) == (oracle.count({p, probe}) != 0));

    const auto fl = t.floor_at(p, as_span12(probe));
    if (expect_ub == oracle.begin()) {
      CHECK(fl == t.end());
    } else {
      const auto expect = std::prev(expect_ub);
      CHECK(fl != t.end());
      if (fl != t.end()) {
        addr12 got{};
        CHECK(fl.address_into(std::span<std::byte>(got.data(), got.size())));
        CHECK(fl.partition() == expect->first.first);
        CHECK(got == expect->first.second);
      }
    }
  }
}

/// The two tiebreaks in isolation. Without them a probe landing in a trimmed
/// region compares equal to a record it is not equal to, which is the kind of
/// bug a randomized oracle can miss for a long time.
void test_trim_probe_edges() {
  // Wide: partition 0 stores bytes 2..3 only, so bytes 0-1 are the head region
  // and 4..11 the tail.
  std::map<std::pair<std::uint64_t, addr12>, std::uint64_t> oracle;
  for (unsigned i = 1; i <= 5; ++i) {
    addr12 a{};
    a[2] = static_cast<std::byte>(i);
    oracle[{0, a}] = i;
  }
  addr12 deep{};  // a second partition, so "past the end" has somewhere to go
  deep[0] = std::byte{0x7f};
  oracle[{1, deep}] = 99;

  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (const auto& [k, v] : oracle) {
    auto rec = tb.begin_record_at(k.first, as_span12(k.second));
    rec.set_uint(f, v);
    tb.commit(rec);
  }
  (void)tb.finish(sink);
  const auto t = mmpack::table::open(sink.data(), sink.size());

  const dir_access acc = open_directory(sink.bytes());
  const auto p0 = get_slot(sink.bytes(), acc, 0);
  CHECK(p0.address_skip == 2 && p0.address_width == 1);

  // tail: equal on the stored range, but with something below it. The record is
  // strictly less, so find misses, lower_bound steps past it, floor returns it.
  addr12 tail{};
  tail[2] = std::byte{3};
  tail[7] = std::byte{1};
  CHECK(t.find_at(0, as_span12(tail)) == t.end());
  {
    const auto lb = t.lower_bound_at(0, as_span12(tail));
    CHECK(lb != t.end());
    if (lb != t.end()) CHECK(t.uint(lb, f).value() == 4);  // stepped past 3
    const auto fl = t.floor_at(0, as_span12(tail));
    CHECK(fl != t.end());
    if (fl != t.end()) CHECK(t.uint(fl, f).value() == 3);
    const auto ub = t.upper_bound_at(0, as_span12(tail));
    CHECK(ub != t.end());
    if (ub != t.end()) CHECK(t.uint(ub, f).value() == 4);
  }

  // head: something above the stored range puts the probe past every record in
  // the partition, whatever the stored bytes say.
  addr12 head{};
  head[0] = std::byte{1};
  head[2] = std::byte{1};  // would compare equal to record 1 on the stored range
  CHECK(t.find_at(0, as_span12(head)) == t.end());
  {
    const auto lb = t.lower_bound_at(0, as_span12(head));
    CHECK(lb != t.end());
    if (lb != t.end()) CHECK(lb.partition() == 1);  // fell through to the next
    const auto fl = t.floor_at(0, as_span12(head));
    CHECK(fl != t.end());
    if (fl != t.end()) CHECK(t.uint(fl, f).value() == 5);  // the partition's last
  }

  // Narrow: the same two situations on an integer address.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> narrow;
  for (std::uint64_t i = 1; i <= 5; ++i) narrow[{0, i << 16}] = i;
  narrow[{1, 0xff000000ull}] = 99;

  mmpack::schema_builder nsb;
  const auto nf = nsb.add_uint("v");
  mmpack::vector_sink nsink;
  mmpack::table_builder ntb(nsb, {});
  for (const auto& [k, v] : narrow) {
    auto rec = ntb.begin_record_at(k.first, k.second);
    rec.set_uint(nf, v);
    ntb.commit(rec);
  }
  (void)ntb.finish(nsink);
  const auto nt = mmpack::table::open(nsink.data(), nsink.size());

  CHECK(nt.find_at(0, (3ull << 16) | 1) == nt.end());  // low bits nothing can hold
  {
    const auto lb = nt.lower_bound_at(0, (3ull << 16) | 1);
    CHECK(lb != nt.end());
    if (lb != nt.end()) CHECK(lb.address().value() == 4ull << 16);
    const auto fl = nt.floor_at(0, (3ull << 16) | 1);
    CHECK(fl != nt.end());
    if (fl != nt.end()) CHECK(fl.address().value() == 3ull << 16);
  }
  {
    // Above everything the stored range can express: the loop must run out
    // rather than wrap into a comparison that happens to match.
    const auto lb = nt.lower_bound_at(0, 0x1'0000'0000ull);
    CHECK(lb != nt.end());
    if (lb != nt.end()) CHECK(lb.partition() == 1);
    const auto fl = nt.floor_at(0, 0x1'0000'0000ull);
    CHECK(fl != nt.end());
    if (fl != nt.end()) CHECK(fl.address().value() == 5ull << 16);
  }
}

/// A partition whose only record has an all-zero address stores no address
/// bytes at all. 80% of partitions in the real /32 IPv6 image are this shape.
void test_trim_zero_width() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  {
    auto rec = tb.begin_record_at(0, std::uint64_t{0});  // the whole partition
    rec.set_uint(f, 7);
    tb.commit(rec);
  }
  for (std::uint64_t i = 1; i <= 20; ++i) {
    auto rec = tb.begin_record_at(1, i << 8);
    rec.set_uint(f, i);
    tb.commit(rec);
  }
  const auto report = tb.finish(sink);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  const dir_access acc = open_directory(sink.bytes());
  const auto p0 = get_slot(sink.bytes(), acc, 0);
  CHECK(p0.address_width == 0 && p0.address_skip == 0);
  CHECK(report.address_width == 2);

  const auto id = t.field("v").value();
  const auto hit = t.find_at(0, std::uint64_t{0});
  CHECK(hit != t.end());
  if (hit != t.end()) {
    CHECK(hit.address().value() == 0);
    CHECK(hit.address_bytes().empty());
    CHECK(t.uint(hit, id).value() == 7);
  }
  // Anything above zero is above the whole partition.
  CHECK(t.find_at(0, std::uint64_t{1}) == t.end());
  const auto lb = t.lower_bound_at(0, std::uint64_t{1});
  CHECK(lb != t.end());
  if (lb != t.end()) CHECK(lb.partition() == 1);
  const auto fl = t.floor_at(0, std::uint64_t{1});
  CHECK(fl != t.end());
  if (fl != t.end()) CHECK(fl.address().value() == 0);
}

/// With nothing to trim, both directory fields stay absent and the image is
/// byte-for-byte what it was with the feature switched off.
void test_trim_costs_nothing_when_uniform() {
  std::mt19937_64 rng(555);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> keys;
  for (std::uint64_t p = 0; p < 8; ++p) {
    for (int i = 0; i < 200; ++i) keys.push_back({p, rng() & 0xffffffffull});
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  const auto build = [&](bool trim) {
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::build_options options;
    options.partition_address_width = trim;
    mmpack::vector_sink sink;
    mmpack::table_builder tb(sb, options);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      auto rec = tb.begin_record_at(keys[i].first, keys[i].second);
      rec.set_uint(f, i % 97);
      tb.commit(rec);
    }
    (void)tb.finish(sink);
    return sink.bytes();
  };

  const auto with_trim = build(true);
  const auto without = build(false);
  CHECK(with_trim == without);  // random addresses leave nothing to trim

  const auto t = mmpack::table::open(with_trim.data(), with_trim.size());
  CHECK(!t.trims_addresses());
  CHECK(t.schema().directory_layout().addr_width == 0);
  CHECK(t.schema().directory_layout().skip_width == 0);

  // And switching it off on data that *could* trim gives the old layout back.
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::build_options options;
  options.partition_address_width = false;
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, options);
  for (std::uint64_t i = 1; i <= 100; ++i) {
    auto rec = tb.begin_record_at(0, i << 16);
    rec.set_uint(f, i);
    tb.commit(rec);
  }
  const auto report = tb.finish(sink);
  CHECK(report.narrowed_partitions == 0);
  CHECK(report.address_bytes_total == 100 * report.address_width);
  const auto plain = mmpack::table::open(sink.data(), sink.size());
  CHECK(!plain.trims_addresses());
  for (std::uint64_t i = 1; i <= 100; ++i) {
    const auto it = plain.find_at(0, i << 16);
    CHECK(it != plain.end());
    if (it != plain.end()) CHECK(it.address().value() == i << 16);
  }
}

/// Trimming and value remap are independent, and a partition can carry both.
void test_trim_with_remap() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::build_options options;
  options.value_interning = mmpack::interning_policy::always;
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, options);
  // Many partitions so the global tuple id needs more than a byte, each holding
  // few distinct values so a local table pays -- the remap fixture -- and
  // /16-aligned addresses so there is also a prefix to skip.
  for (std::uint64_t p = 0; p < 60; ++p) {
    for (std::uint64_t i = 1; i <= 40; ++i) {
      auto rec = tb.begin_record_at(p, i << 16);
      rec.set_uint(f, p * 5 + (i % 5));
      tb.commit(rec);
    }
  }
  const auto report = tb.finish(sink);
  CHECK(report.interned);
  CHECK(report.remapped_partitions > 0);
  CHECK(report.narrowed_partitions == 60);

  const auto t = mmpack::table::open(sink.data(), sink.size());
  const auto id = t.field("v").value();
  const dir_access acc = open_directory(sink.bytes());
  const auto slot0 = get_slot(sink.bytes(), acc, 0);
  CHECK(slot0.schema_id != mmpack::remap::none);  // remapped
  CHECK(slot0.address_skip == 2);                 // and trimmed
  for (std::uint64_t p = 0; p < 60; ++p) {
    for (std::uint64_t i = 1; i <= 40; ++i) {
      const auto it = t.find_at(p, i << 16);
      CHECK(it != t.end());
      if (it == t.end()) continue;
      CHECK(it.address().value() == i << 16);
      CHECK(t.uint(it, id).value() == p * 5 + (i % 5));
    }
  }
}

/// A stored range outside the address field must be refused, however it is
/// spelled -- otherwise the record stride and the zero-fill in address_into()
/// would both be computed from a lie.
void test_trim_corruption() {
  mmpack::schema_builder sb;
  const auto f = sb.add_uint("v");
  mmpack::vector_sink sink;
  mmpack::table_builder tb(sb, {});
  for (std::uint64_t p = 0; p < 4; ++p) {
    for (std::uint64_t i = 1; i <= 50; ++i) {
      auto rec = tb.begin_record_at(p, i << 16);
      rec.set_uint(f, i);
      tb.commit(rec);
    }
  }
  (void)tb.finish(sink);
  const std::vector<std::byte>& image = sink.bytes();
  CHECK(mmpack::table::try_open(image.data(), image.size()).has_value());

  const dir_access acc = open_directory(image);
  CHECK(acc.layout.addr_width == 1);
  CHECK(acc.layout.skip_width == 1);

  const auto rejects = [&](auto mutate) {
    auto corrupt = image;
    auto d = get_slot(corrupt, acc, 0);
    mutate(d);
    put_slot(corrupt, acc, 0, d);
    mmpack::status s = mmpack::status::ok;
    const auto opened = mmpack::table::try_open(corrupt.data(), corrupt.size(), &s);
    CHECK(!opened.has_value());
  };

  rejects([](mmpack::dir_entry& d) { d.address_width = 250; });          // past the field
  rejects([](mmpack::dir_entry& d) { d.address_skip = 250; });           // past the field
  rejects([](mmpack::dir_entry& d) { d.address_skip = 3; });             // skip + width > 4
  rejects([](mmpack::dir_entry& d) {
    d.address_width = 4;  // a wider stride than the partition's extent allows
    d.address_skip = 0;
  });

  // Shrinking the stored width without moving anything else leaves the records
  // consistent but shorter, which must not be read as the original layout.
  {
    auto corrupt = image;
    auto d = get_slot(corrupt, acc, 0);
    d.address_width = 1;
    put_slot(corrupt, acc, 0, d);
    const auto opened = mmpack::table::try_open(corrupt.data(), corrupt.size());
    // Accepting it is fine -- the geometry is still in bounds -- but every read
    // must stay inside the image, which is what the sanitizer build proves.
    if (opened) {
      for (auto it = opened->begin(); it != opened->end(); ++it) {
        (void)it.address();
        (void)opened->uint(it, f);
      }
    }
  }
}

void test_directory_is_narrowed() {
  {  // Dense: the partition is implicit, so it costs nothing at all.
    auto [sink, report] = build_remap_case(true, 60, 100, 5);
    const auto t = mmpack::table::open(sink->data(), sink->size());
    CHECK(t.has_dense_directory());
    const auto layout = t.schema().directory_layout();
    CHECK(layout.partition_width == 0);
    CHECK(layout.schema_width == 1);   // some partition is remapped
    CHECK(layout.remap_width >= 1);
    CHECK(layout.stride == report.directory_stride);
    CHECK(layout.stride < sizeof(mmpack::dir_entry));
    // offset + count + schema_id + remap_count, all narrowed.
    CHECK(layout.stride <= 12);

    // Every slot still decodes to the right partition and record count.
    std::uint64_t seen = 0;
    for (std::uint64_t i = 0; i < t.directory_slots(); ++i) seen += 1;
    CHECK(seen == t.directory_slots());
    for (std::uint64_t p = 0; p < 60; ++p) {
      CHECK(t.find((p << 8) | 0) != t.end());
    }
  }
  {  // Sparse: the partition is stored, but only as wide as it needs to be.
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::vector_sink sink;
    mmpack::build_options o;
    o.address_bits = 8;
    mmpack::table_builder tb(sb, o);
    for (std::uint64_t p : {0ull, 1ull, 1ull << 30}) {
      auto rec = tb.begin_record((p << 8) | 1);
      rec.set_uint(f, p);
      tb.commit(rec);
    }
    const auto report = tb.finish(sink);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    CHECK(!t.has_dense_directory());
    const auto layout = t.schema().directory_layout();
    CHECK(layout.partition_width == 4);  // 2^30 needs four bytes, not eight
    CHECK(layout.schema_width == 0);     // nothing remapped here
    CHECK(layout.remap_width == 0);
    CHECK(layout.stride == report.directory_stride);
    CHECK(layout.stride < sizeof(mmpack::dir_entry));
    for (std::uint64_t p : {0ull, 1ull, 1ull << 30}) {
      const auto it = t.find((p << 8) | 1);
      CHECK(it != t.end());
      if (it != t.end()) CHECK(t.uint(it, f).value() == p);
    }
  }
  {  // A tiny image should land on the narrowest widths available.
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::vector_sink sink;
    mmpack::table_builder tb(sb, {});
    for (std::uint64_t i = 0; i < 5; ++i) {
      auto rec = tb.begin_record(i);
      rec.set_uint(f, i);
      tb.commit(rec);
    }
    const auto report = tb.finish(sink);
    const auto t = mmpack::table::open(sink.data(), sink.size());
    const auto layout = t.schema().directory_layout();
    CHECK(layout.offset_width == 1);  // everything lives in the first 256 bytes
    CHECK(layout.count_width == 1);
    CHECK(report.directory_stride <= 3);
  }
}

void test_remap_corruption() {
  // 60 partitions x 5 distinct = 300 global tuples, so the global reference is
  // 2 bytes and a 1-byte local index is genuinely narrower. Fewer partitions
  // would leave the global reference at 1 byte and nothing would remap.
  auto [sink, report] = build_remap_case(true, 60, 100, 5);
  CHECK(report.ref_width == 2);
  CHECK(report.remapped_partitions == 60);
  const auto& image = sink->bytes();
  mmpack::status s = mmpack::status::ok;
  const dir_access acc = open_directory(image);
  CHECK(acc.layout.schema_width == 1);  // remapped images carry the field
  CHECK(acc.layout.remap_width >= 1);

  {  // an impossible local reference width
    auto corrupt = image;
    auto d = get_slot(corrupt, acc, 0);
    d.schema_id = 7;
    put_slot(corrupt, acc, 0, d);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // remapped, but with no remap table to resolve through
    auto corrupt = image;
    auto d = get_slot(corrupt, acc, 0);
    d.remap_count = 0;
    put_slot(corrupt, acc, 0, d);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // a remap table claiming more entries than the image can hold. The field is
     // only as wide as the real tables needed, so the largest expressible value
     // is what to push it to.
    auto corrupt = image;
    auto d = get_slot(corrupt, acc, 0);
    d.remap_count =
        static_cast<std::uint32_t>(mmpack::detail::max_for_width(acc.layout.remap_width));
    put_slot(corrupt, acc, 0, d);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // a remap table on a partition that is not remapped
    auto corrupt = image;
    auto d = get_slot(corrupt, acc, 0);
    d.schema_id = 0;
    put_slot(corrupt, acc, 0, d);  // remap_count stays non-zero
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // a local index past the end of its remap table must not read the records
     // that follow it -- this is the one that only fails at access time
    auto corrupt = image;
    const auto d = get_slot(corrupt, acc, 0);
    CHECK(d.schema_id == 1);
    std::byte* records = corrupt.data() + d.offset + d.remap_count * sizeof(std::uint32_t);
    const auto t0 = mmpack::table::open(corrupt.data(), corrupt.size());
    const auto id = t0.field("v").value();
    CHECK(t0.uint(t0.begin(), id).has_value());  // fine before corruption

    records[/*address_width=*/1] = std::byte{0xff};  // local index 255, table has 5
    const auto t1 = mmpack::table::open(corrupt.data(), corrupt.size());
    CHECK(!t1.uint(t1.begin(), id).has_value());
    CHECK(t1.begin().key() == 0);  // the key still reads: it lives in the record
  }
  {  // An image with no remapped partition has no schema_id or remap_count field
     // in its directory at all, so "remapped" is not merely rejected there -- it
     // cannot be expressed.
    mmpack::schema_builder sb;
    const auto f = sb.add_uint("v");
    mmpack::vector_sink plain;
    mmpack::build_options o;
    o.value_interning = mmpack::interning_policy::never;
    mmpack::table_builder tb(sb, o);
    for (std::uint64_t i = 0; i < 50; ++i) {
      auto rec = tb.begin_record(i);
      rec.set_uint(f, i);
      tb.commit(rec);
    }
    const auto rep = tb.finish(plain);
    CHECK(rep.remapped_partitions == 0);

    const dir_access plain_acc = open_directory(plain.bytes());
    CHECK(plain_acc.layout.schema_width == 0);
    CHECK(plain_acc.layout.remap_width == 0);
    auto corrupt = plain.bytes();
    auto d = get_slot(corrupt, plain_acc, 0);
    d.schema_id = 1;
    d.remap_count = 4;
    put_slot(corrupt, plain_acc, 0, d);
    // The write is a no-op for those two fields, so the image stays valid and
    // still reads correctly rather than becoming a rejected one.
    const auto t = mmpack::table::open(corrupt.data(), corrupt.size());
    CHECK(t.size() == 50);
    CHECK(t.uint(t.find(7), t.field("v").value()).value() == 7);
  }
}

void test_rejects_bad_images() {
  const auto rows = make_rows(500, 30);
  mmpack::build_options options;
  options.value_interning = mmpack::interning_policy::always;
  const auto b = build(rows, options);
  const auto& image = b->sink.bytes();
  mmpack::status s = mmpack::status::ok;

  CHECK(mmpack::table::try_open(image.data(), image.size(), &s).has_value());
  CHECK(s == mmpack::status::ok);

  CHECK(!mmpack::table::try_open(nullptr, image.size(), &s).has_value());
  CHECK(!mmpack::table::try_open(image.data(), 8, &s).has_value());
  CHECK(s == mmpack::status::too_small);

  // Truncation and trailing slack both mean "not the exact image size".
  CHECK(!mmpack::table::try_open(image.data(), image.size() - 1, &s).has_value());
  CHECK(s == mmpack::status::bad_size);
  {
    auto padded = image;
    padded.push_back(std::byte{0});
    CHECK(!mmpack::table::try_open(padded.data(), padded.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_size);
  }
  {
    auto corrupt = image;
    corrupt[0] = std::byte{'X'};
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_magic);
  }

  const auto load_footer = [](const std::vector<std::byte>& img) {
    return mmpack::detail::load<mmpack::footer>(img.data() + img.size() - sizeof(mmpack::footer));
  };
  const auto store_footer = [](std::vector<std::byte>& img, const mmpack::footer& f) {
    mmpack::detail::store(img.data() + img.size() - sizeof(mmpack::footer), f);
  };

  {  // schema pointing outside the image
    auto corrupt = image;
    auto f = load_footer(corrupt);
    f.schema_offset = corrupt.size() * 4;
    store_footer(corrupt, f);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // directory pointing outside the image
    auto corrupt = image;
    auto f = load_footer(corrupt);
    f.dir_offset = corrupt.size() * 4;
    store_footer(corrupt, f);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // a partition claiming more records than fit
    auto corrupt = image;
    const dir_access acc = open_directory(corrupt);
    auto slot = get_slot(corrupt, acc, 0);
    slot.count = mmpack::detail::max_for_width(acc.layout.count_width);
    put_slot(corrupt, acc, 0, slot);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // a partition whose records start outside the image
    auto corrupt = image;
    const dir_access acc = open_directory(corrupt);
    auto slot = get_slot(corrupt, acc, 0);
    slot.offset = mmpack::detail::max_for_width(acc.layout.offset_width);
    put_slot(corrupt, acc, 0, slot);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // directory widths that disagree with the density flag
    auto corrupt = image;
    const auto f = load_footer(corrupt);
    auto head = mmpack::detail::load<mmpack::schema_header>(corrupt.data() + f.schema_offset);
    head.dir_partition_width = head.dir_partition_width == 0 ? 4 : 0;
    mmpack::detail::store(corrupt.data() + f.schema_offset, head);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
  }
  {  // an illegal directory field width
    auto corrupt = image;
    const auto f = load_footer(corrupt);
    auto head = mmpack::detail::load<mmpack::schema_header>(corrupt.data() + f.schema_offset);
    head.dir_count_width = 3;
    mmpack::detail::store(corrupt.data() + f.schema_offset, head);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // a field reaching past the value tuple
    auto corrupt = image;
    const auto f = load_footer(corrupt);
    const auto head =
        mmpack::detail::load<mmpack::schema_header>(corrupt.data() + f.schema_offset);
    auto fd = mmpack::detail::load<mmpack::field_desc>(corrupt.data() + f.schema_offset +
                                                       sizeof(mmpack::schema_header));
    fd.offset = head.value_stride;  // width would run off the end
    fd.width = 8;
    mmpack::detail::store(
        corrupt.data() + f.schema_offset + sizeof(mmpack::schema_header), fd);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // an illegal field width
    auto corrupt = image;
    const auto f = load_footer(corrupt);
    auto fd = mmpack::detail::load<mmpack::field_desc>(corrupt.data() + f.schema_offset +
                                                       sizeof(mmpack::schema_header));
    fd.width = 3;
    mmpack::detail::store(
        corrupt.data() + f.schema_offset + sizeof(mmpack::schema_header), fd);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // an over-wide bytes field: it is read through the runtime-width path, so
     // anything past 8 would write past a uint64 on load
    mmpack::schema_builder sb;
    sb.add_bytes("b", 4);
    mmpack::table_builder tb(sb, {});
    auto rec = tb.begin_record(1);
    const std::byte four[4] = {};
    rec.set_bytes(0, std::span<const std::byte>(four, 4));
    tb.commit(rec);
    mmpack::vector_sink small;
    tb.finish(small);

    auto corrupt = small.bytes();
    const auto f = load_footer(corrupt);
    auto fd = mmpack::detail::load<mmpack::field_desc>(corrupt.data() + f.schema_offset +
                                                       sizeof(mmpack::schema_header));
    fd.width = 40;
    mmpack::detail::store(corrupt.data() + f.schema_offset + sizeof(mmpack::schema_header), fd);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_schema);
  }
  {  // a dictionary pointing outside the image
    auto corrupt = image;
    const auto f = load_footer(corrupt);
    const auto head =
        mmpack::detail::load<mmpack::schema_header>(corrupt.data() + f.schema_offset);
    const std::uint64_t dicts_at = f.schema_offset + sizeof(mmpack::schema_header) +
                                   head.field_count * sizeof(mmpack::field_desc);
    auto e = mmpack::detail::load<mmpack::dict_entry>(corrupt.data() + dicts_at);
    e.offset = corrupt.size() * 4;
    mmpack::detail::store(corrupt.data() + dicts_at, e);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_dictionary);
  }

  // The throwing form reports the same reason.
  bool threw = false;
  try {
    (void)mmpack::table::open(image.data(), 8);
  } catch (const mmpack::format_error& e) {
    threw = true;
    CHECK(e.code() == mmpack::status::too_small);
  }
  CHECK(threw);
}

void test_corrupt_value_reference() {
  // The interned hot path: a reference beyond the composite dictionary must
  // yield nullopt rather than reading wherever it points.
  const auto rows = make_rows(200, 8);
  mmpack::build_options options;
  options.address_bits = 8;
  options.value_interning = mmpack::interning_policy::always;
  const auto b = build(rows, options);

  auto corrupt = b->sink.bytes();
  const dir_access acc = open_directory(corrupt);
  const mmpack::dir_entry slot = get_slot(corrupt, acc, 0);
  const auto head = mmpack::detail::load<mmpack::schema_header>(corrupt.data() +
                                                                acc.foot.schema_offset);

  // Point the first record's value reference at a wildly out-of-range entry.
  // Records sit after the remap table when the partition has one, and the
  // reference is local rather than global in that case -- either way, an
  // all-ones reference must resolve to nothing.
  const unsigned width = slot.schema_id != 0 ? slot.schema_id : head.ref_width;
  std::byte* ref = corrupt.data() + slot.offset +
                   slot.remap_count * sizeof(std::uint32_t) + head.address_width;
  for (unsigned i = 0; i < width; ++i) ref[i] = std::byte{0xff};

  const auto t = mmpack::table::open(corrupt.data(), corrupt.size());
  const auto it = t.begin();
  CHECK(it != t.end());
  CHECK(!t.text(it, b->country).has_value());
  CHECK(!t.uint(it, b->population).has_value());
  CHECK(!t.sint(it, b->temp).has_value());
  CHECK(!t.bytes(it, b->country).has_value());
  // The key still reads fine: it lives in the record, not the dictionary.
  CHECK(it.key() == rows.front().key);
}

void test_large_randomized_both_modes() {
  const auto rows = make_rows(20000, 500, 99);
  for (const auto policy :
       {mmpack::interning_policy::never, mmpack::interning_policy::always}) {
    mmpack::build_options options;
    options.address_bits = 12;
    options.value_interning = policy;
    const auto b = build(rows, options);
    const auto t = mmpack::table::open(b->sink.data(), b->sink.size());
    CHECK(t.size() == rows.size());
    verify_contents(t, *b, rows);

    std::size_t n = 0;
    for (auto it = t.begin(); it != t.end(); ++it) {
      CHECK(it.key() == rows[n].key);
      ++n;
    }
    CHECK(n == rows.size());
  }
}

}  // namespace

int main() {
  run("roundtrip in both value modes", test_roundtrip_both_value_modes);
  run("cost model decides", test_cost_model_decides);
  run("width and bias selection", test_width_and_bias_selection);
  run("text dictionary widths", test_text_dictionary_widths);
  run("lookup matches std::map", test_lookup_matches_std_map);
  run("floor and ceil match oracle", test_floor_and_ceil_match_oracle);
  run("floor edge cases", test_floor_edge_cases);
  run("floor sparse directory", test_floor_sparse_directory);
  run("cross partition fallthrough", test_cross_partition_fallthrough);
  run("input ordering", test_input_ordering);
  run("all field kinds", test_all_field_kinds);
  run("schema build errors", test_schema_build_errors);
  run("empty and single", test_empty_and_single);
  run("dense and sparse directories", test_dense_and_sparse_directories);
  run("align fields option", test_align_fields_option);
  run("dedup give up", test_dedup_give_up);
  run("partition remap", test_partition_remap);
  run("partition remap is selective", test_partition_remap_is_selective);
  run("explicit parts, wide keys", test_explicit_parts_wide_keys);
  run("keyed wrapper", test_keyed_wrapper);
  run("keyed wrapper, wide", test_keyed_wrapper_wide);
  run("key modes", test_key_modes);
  run("keyed and explicit agree", test_keyed_and_explicit_agree);
  run("key mapping sentinel", test_key_mapping_sentinel_rejected);
  run("wide addresses", test_wide_addresses);
  run("wide address ordering is lexicographic", test_wide_address_ordering_is_lexicographic);
  run("wide address build errors", test_wide_address_build_errors);
  run("odd narrow widths", test_odd_narrow_widths);
  run("span overload on narrow images", test_span_overload_on_narrow_images);
  run("wide address corruption", test_wide_address_corruption);
  run("narrow low-order trim", test_narrow_low_order_trim);
  run("wide partition trim", test_wide_partition_trim);
  run("trim probe edges", test_trim_probe_edges);
  run("trim zero width", test_trim_zero_width);
  run("trim costs nothing when uniform", test_trim_costs_nothing_when_uniform);
  run("trim with remap", test_trim_with_remap);
  run("trim corruption", test_trim_corruption);
  run("directory is narrowed", test_directory_is_narrowed);
  run("remap corruption", test_remap_corruption);
  run("rejects bad images", test_rejects_bad_images);
  run("corrupt value reference", test_corrupt_value_reference);
  run("large randomized both modes", test_large_randomized_both_modes);

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
