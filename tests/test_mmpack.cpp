// Self-contained test suite: no external framework, just CHECK macros.
#include <algorithm>
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
      seen.push_back(it.key());
      CHECK(t.uint(it, id).value() == it.key() * 2);
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
    const auto f = load_footer(corrupt);
    auto slot = mmpack::detail::load<mmpack::dir_entry>(corrupt.data() + f.dir_offset);
    slot.count = 1ull << 40;
    mmpack::detail::store(corrupt.data() + f.dir_offset, slot);
    CHECK(!mmpack::table::try_open(corrupt.data(), corrupt.size(), &s).has_value());
    CHECK(s == mmpack::status::bad_directory);
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
  const auto f =
      mmpack::detail::load<mmpack::footer>(corrupt.data() + corrupt.size() - sizeof(mmpack::footer));
  const auto slot = mmpack::detail::load<mmpack::dir_entry>(corrupt.data() + f.dir_offset);
  const auto head = mmpack::detail::load<mmpack::schema_header>(corrupt.data() + f.schema_offset);

  // Point the first record's value reference at a wildly out-of-range entry.
  std::byte* ref = corrupt.data() + slot.offset + head.address_width;
  for (unsigned i = 0; i < head.ref_width; ++i) ref[i] = std::byte{0xff};

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
  run("rejects bad images", test_rejects_bad_images);
  run("corrupt value reference", test_corrupt_value_reference);
  run("large randomized both modes", test_large_randomized_both_modes);

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
