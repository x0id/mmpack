# mmpack

Header-only C++20 library for `lower_bound`-style key → record lookups over a
read-only, memory-mapped image — with the **record shape computed from the data**
rather than declared at compile time.

You cannot know how many distinct countries there are, whether a population fits
in two bytes, or how often whole records repeat, until the input has been
consumed. mmpack measures first and lays out afterwards, then records the shape
in the image so the reader can follow it.

```cpp
#include <mmpack/mmpack.hpp>

mmpack::schema_builder sb;
const auto city = sb.add_text("city");        // interned; index width chosen later
const auto pop  = sb.add_uint("population");  // width and bias chosen from the range

mmpack::table_builder tb(sb, {.address_bits = 16});
for (const auto& row : rows) {                // keys must arrive in order
  auto rec = tb.begin_record(row.key);
  rec.set_text(city, row.city);
  rec.set_uint(pop, row.population);
  tb.commit(rec);
}
mmpack::vector_sink sink;
const mmpack::build_report report = tb.finish(sink);

const auto t = mmpack::table::open(sink.data(), report.image_bytes);
const auto it = t.find(key);
if (it != t.end()) {
  std::string_view name = t.text(it, city).value_or("?");
  std::uint64_t people  = t.uint(it, pop).value_or(0);
}
```

## What the compaction buys

Two independent mechanisms, both driven by measurement.

**Per-field compaction.** Every integer field records its min and max, and is
stored as `value - bias` in the narrowest width that fits the range. Magnitude is
irrelevant; only the span matters. Text fields are interned and stored as an
index whose width follows the cardinality.

**Whole-value interning.** The entire value tuple is deduplicated and the record
holds one reference into a composite dictionary. This catches repeated
*combinations* that per-field compaction cannot see — correlated fields (city
implies region implies country) still cost one index each under per-field alone,
versus one reference under interning.

From `examples/ipv4_routes.cpp`, 500k routes over 3000 distinct value tuples:

```
derived record shape (value tuple = 7 bytes):
  next_hop   uint  offset  0  width 2  bias 167772160
  asn        uint  offset  2  width 2  bias 64500
  metric     uint  offset  4  width 1  bias 10
  interface  text  offset  5  width 1  bias 0
  region     text  offset  6  width 1  bias 0

layout            bytes   stride     /entry     ns/query
naive          23970000       48       48.0            -
inline          4625941        9        9.3         45.3
interned        2150101        4        4.3         41.0
```

`naive` is what a fixed-width struct with inline 16-byte strings would cost. The
`next_hop` line is the bias at work: values around 0.17 billion, stored in two
bytes, because they only span a narrow range.

### Per-partition value remap

Under interning every record holds a reference into the global composite
dictionary, so the reference is as wide as the *total* distinct tuple count
demands — 4 bytes past 65536 tuples. But an individual partition usually touches
only a handful of them. Giving such a partition its own table of the global ids
it actually uses lets its records carry a 1- or 2-byte local index instead:

```
[remap table: remap_count * 4 bytes of global ids][records at address_width + local_width]
```

That is pure stride reduction on the search path, which is where it shows up.
Measured on geo-IP-shaped data — 4.8M records, /16 partitions, ~30 distinct
tuples each, 4-byte global reference:

| | stride | image | ns/lookup |
|---|---|---|---|
| global reference | 6 (2+4) | 32.1 MB | 87.4 |
| per-partition remap | **3** (2+1) | **22.4 MB** | **68.3** |

Halving the stride doubles the records per cache line, and the binary search
touches correspondingly fewer. Expect a larger effect than the −22% above once
the image no longer fits in cache at all, since every avoided line becomes an
avoided DRAM miss.

It is decided per partition, and only where it pays: a table costs
`distinct * 4` bytes, so it must save more than that in records. Partitions whose
tuples are nearly all distinct keep the global reference, and one image freely
mixes both — the reader takes the stride from the directory entry. Turn it off
with `build_options.partition_remap = false`. `build_report` reports
`remapped_partitions` and `remap_saved_bytes`, and the saving is folded into the
interning cost model so the two decisions stay consistent.

## The interning cost model

Interning is decided automatically at `finish()` and pays when

```
(N - U) * V  >  N * R
```

for N records, U distinct values, V the compacted value stride and R the
reference width implied by U. The decision and every input to it come back in the
`build_report`, so it is auditable rather than magic;
`build_options.value_interning` overrides it with `always` / `never`.

**It is not always a win.** Sweeping the duplication ratio over the same 500k
routes:

| records | distinct | ratio | inline MB | intern MB | ns inline | ns intern | auto picks |
|---|---|---|---|---|---|---|---|
| 499375 | 1000 | 499.4x | 4.63 | 2.14 | 45.3 | 41.0 | interned |
| 499375 | 10000 | 49.9x | 4.63 | 2.20 | 46.4 | 42.2 | interned |
| 499375 | 99307 | 5.0x | 4.63 | 3.82 | 45.6 | 44.2 | interned |
| 499375 | 285218 | 1.8x | 4.63 | **5.12** | 45.4 | 45.3 | inline |

Below roughly 2x duplication, forced interning makes the image *larger* — the
dictionary costs more than the references save. That last row is exactly what the
cost model exists to prevent.

Two caveats on the timings, so they are not read as more than they are. The
lookup deltas above are within noise: at 500k records every dictionary still fits
in cache, so this sweep never reaches the regime where the extra indirection
becomes a DRAM miss. Expect interning to cost lookup time once the value
dictionary outgrows cache. And **sequential scans get worse under interning
regardless of size**, since iterating in key order touches the dictionary
randomly — scan-heavy workloads should set `value_interning = never`.

## Keys

mmpack stores two ordering coordinates per record — a **partition** and an
**address**, both `uint64` — and nothing else. A 64-bit key is only a packing of
them, so any key type of any width works as long as you can split it.

**Keyed builds** take the packing for you, splitting a `uint64` by
`address_bits`:

```cpp
mmpack::table_builder tb(sb, {.address_bits = 16});
tb.begin_record(ipv4);              // partition = ipv4 >> 16, address = ipv4 & 0xffff
...
t.find(ipv4);  t.floor(ipv4);       // and the key-taking lookups work
```

**Caller-split builds** take the coordinates directly, which is what lets a key
be wider than 64 bits or not a number at all:

```cpp
tb.begin_record_at(v6.high, v6.low);   // a 128-bit key, split 64/64
...
t.find_at(v6.high, v6.low);  t.floor_at(v6.high, v6.low);
```

A build is one or the other, fixed by its first record; mixing throws. Ordering
is checked on `(partition, address)` either way, so duplicates still fall out of
the same comparison.

**A caller-split image reports no key mapping**, and `iterator::key()` returns
`std::nullopt` there. That is deliberate rather than a missing feature: the
library does not know your encoding. Split an IPv4 as (high 16, low 16) but never
store an address above 1000, and a key synthesized from the observed range would
be `(p << 10) | a` — correctly ordered, and not your key. The key-taking lookups
likewise find nothing on such an image; `has_key_mapping()` distinguishes that
from a genuine miss.

### Typed keys without the boilerplate

`mmpack/keyed.hpp` holds the split once instead of at each call site. `Join` is
optional and gates `key()`, the same way mmseek's `join_key` gated it through
`joinable_traits`:

```cpp
struct split_v6 { std::pair<std::uint64_t, std::uint64_t> operator()(const ipv6& k) const
                  { return {k.high, k.low}; } };
struct join_v6  { ipv6 operator()(std::uint64_t p, std::uint64_t a) const { return {p, a}; } };

mmpack::keyed_builder<ipv6, split_v6> kb(tb, split_v6{});
auto rec = kb.begin_record(addr);            // splits, then begin_record_at

mmpack::keyed_table<ipv6, split_v6, join_v6> kt(t, split_v6{}, join_v6{});
auto it = kt.floor(addr);
ipv6 back = kt.key(it);                      // only compiles with a join supplied
```

Both are non-owning views that forward to the `_at` forms after an inlined
split, so they cost nothing at runtime and the format never learns about them.

### Addresses wider than 64 bits

An address may be any fixed width. Past 8 bytes it stops being an integer and
becomes an opaque byte string compared with `memcmp`:

```cpp
std::array<std::byte, 14> host;      // the low 112 bits of an IPv6 address
tb.begin_record_at(prefix, std::span<const std::byte>(host));
...
t.find_at(prefix, std::span<const std::byte>(host));
t.floor_at(prefix, std::span<const std::byte>(host));
```

**You must supply an order-preserving encoding**, because the library never
interprets those bytes — it only compares them. Big endian is the usual answer,
and IPv6 network order already is one, so it works unchanged. This is the same
contract a split already carries.

The width is fixed by the first such record and every later one must match; a
uniform stride is what makes the binary search possible. `narrow_address()` says
which regime an image is in, `iterator::address()` returns `std::nullopt` on a
wide one, and `iterator::address_bytes()` is always valid. Each `*_at` form takes
either an integer or a span: the span form serves both regimes, and the integer
form finds nothing on a wide image, since the probe is not in its address space.

Widths 3, 5, 6 and 7 are legal too — they were only ever excluded by a check that
listed 1, 2, 4 and 8.

**Wide addresses cost build memory**: staging carries 16 bytes plus the address
width per record, so 290M records with a 14-byte address is roughly 8.7 GB
against 4.6 GB for the narrow path. Only builds that use them pay it.

**The partition stays a `uint64`, deliberately.** It is a bucket selector you
choose, not key payload: a dense directory indexes *by* partition, so widening it
would cost O(1) partition select and turn it into a `memcmp` binary search. Key
payload belongs in the address, which is why that is the side with no cap.

## Searching

| method | returns |
|---|---|
| `find(k)` | exact match, or `end()` |
| `lower_bound(k)` / `ceil(k)` | least entry with key ≥ k, or `end()` |
| `upper_bound(k)` | least entry with key > k, or `end()` |
| `floor(k)` | greatest entry with key ≤ k, or `end()` |

Each has a `_at(partition, address)` form for callers who split keys themselves.

`floor()` is the **range-containment** primitive: given boundary entries, it
names the range covering a point. That query is otherwise a two-step dance with a
begin() check that every caller has to remember:

```cpp
auto it = t.floor(ip);                    // one call
if (it == t.end()) { /* not covered */ }

auto it = t.upper_bound(ip);              // the same thing, by hand
if (it == t.begin()) { /* not covered */ }
else --it;
```

`ceil()` is `lower_bound()` under a name that reads symmetrically next to
`floor()`; it is a trivial forwarding call, not a second implementation.

**On speed:** `floor()` is measurably but modestly faster than
`upper_bound()` + `--it` — around 3–7% on shallow searches, and within noise when
the binary search dominates. It was worth measuring, because the intuition that
it saves a cache line does not hold: the record at the upper-bound position is
never dereferenced by either formulation. What `floor()` actually saves is the
`normalize()` that walks an iterator forward past the partition end, and the
`retreat()` that walks it back — a handful of branches and directory loads.
Measured over 500k boundary entries, 2M probes:

| workload | `floor` | `upper_bound` + `--it` | delta |
|---|---|---|---|
| probes just past a stored key | 18.8 ns | 20.2 ns | −6.7% |
| random probes, many empty partitions | 19.4 ns | 20.3 ns | −4.5% |
| random probes, 10 large partitions | 68.0 ns | 66.6 ns | +2.2% |

The last row is the honest one: with few, large partitions the search itself is
cache-bound and dominates, so the saving disappears into noise. Reach for
`floor()` because it makes containment queries a single correct call, not because
it is fast.

## Fields

| kind | stored as | width |
|---|---|---|
| `add_uint` | `value - min` | 1/2/4/8, from the range |
| `add_sint` | `value - min`, unsigned | 1/2/4/8, from the range |
| `add_f32` / `add_f64` | IEEE bit pattern | 4 / 8 |
| `add_text` | dictionary index | 1/2/4/8, from cardinality |
| `add_bytes` | raw, inline | 1..8, declared |

`bytes` is capped at 8 because it is read through the same runtime-width path as
the numeric kinds. Anything longer belongs in a `text` field, which handles
arbitrary length and deduplicates as well.

Reads are `optional`-returning and kind-checked: `t.uint(it, id)` on a text field
gives `nullopt`, never a reinterpretation. Fields are addressed by id, or looked
up by name — the image carries its own field names, so it is self-describing.

## Input ordering

Input is **expected to arrive key-sorted**. `commit()` checks each key against
the previous one, which is O(1) and makes duplicate rejection fall out of the
same comparison. That removes a multi-gigabyte sort from large builds.

- `input_order::assume_sorted` (default) — throws naming both keys.
- `input_order::sort_if_needed` — tolerates disorder and sorts during `finish()`.
  Opt-in, because at scale that cost should be a deliberate choice.

## Build memory

The builder stages every record before it can choose widths, so build memory is
O(records):

| | |
|---|---|
| staging, 16 bytes per record | `16 N` |
| tuple table | `U * fields * 8` |
| dedup hash index | ~`24 U` |

For 290M records over 60M distinct tuples that is roughly 7 GB. Interning runs
during staging regardless of the final decision, because storing a tuple id
instead of every field value is what keeps staging small; if the distinct ratio
stays above `dedup_give_up_ratio` after `dedup_sample` records, the hash index is
dropped rather than paid for.

`record_staging` is an interface so this can be replaced. Because input is
already ordered and re-readable, the natural escape hatch is a two-pass builder
— pass 1 interns and measures, pass 2 re-reads and streams records — rather than
spilling to disk.

## Format

```
[header 16B]
[dictionaries]            text dictionaries, then the composite value dictionary
[partition 0 records][partition 1 records]...
[directory]
[schema]                  field descriptors, dictionary table, name blob
[footer 64B]
```

The directory, schema and footer sit at the end so the writer never seeks:
partition offsets are only known after their records are emitted, and the schema
is only final once the data has been measured. The reader finds the footer at
`base + length - 64` and works backwards.

A record is `[address][value]`. The address comes first at a fixed width so the
binary search never consults the schema. The value is either the fields inline,
or one reference into the composite dictionary:

- inline: `base = record + address_width`
- interned: `base = value_dict + load(record + address_width, ref_width) * value_stride`

Record stride is fixed *within* a partition but may differ *between* them: a
remapped partition prefixes its records with a table of global tuple ids and
narrows its references accordingly. `dir_entry::schema_id` carries the local
reference width (0 = global) and `remap_count` the table length, so the reader
resolves the geometry before the search starts.

**The directory itself is packed the same way the records are.** `dir_entry` is
the decoded form, not the stored one: each field is narrowed to what the image
needs, and a dense directory omits the partition entirely because slot *i*
describes partition *i* by construction — storing it would only be a value to
check against its own index. A geo-IP-shaped image lands on 5–7 bytes per slot
against the 32 a struct would take, so seven slots share a cache line instead of
two. The widths live in the schema block, which the reader parses before it
touches the directory.

Fields are read as one masked 8-byte load rather than a switch on the width,
which is what keeps the packed form from costing more than it saves; `open()`
proves the eight bytes are in range. Measured against the fixed-struct
directory on 4.8M records: 32.07 → 30.99 MB and 84.0 → 82.8 ns without remap,
22.39 → 21.39 MB and 67.7 → 63.8 ns with it.

Lookup is two steps as ever: select the partition (O(1) when partition indices
are dense, O(log P) when sparse), then binary search the fixed-stride records.
Key splitting is runtime configuration — `partition = key >> address_bits` — with
`lower_bound_at(partition, address)` available for keys that split differently.

`length` must be the **exact** image size. The footer is found by counting back
from the end, so a region with trailing slack is rejected rather than misread.

### Untrusted images

The schema is itself parsed from untrusted bytes, so a bad field offset or width
would turn every record access into an out-of-bounds read. `try_open` proves the
schema self-consistent — widths legal for the kind, every field inside the value
tuple, names NUL-terminated inside the blob, dictionary indices in range — and
validates the directory and every dictionary, before any accessor is built. A
value reference past the composite dictionary yields `nullopt`, not a wild read.

`tests/fuzz_open.cpp` is the evidence rather than the assertion: it corrupts
valid images with bytes aimed at the schema, the directory and the records, then
fully exercises every survivor under ASan.

## Building

Header-only — copy `include/` and add it to your include path, or:

```cmake
add_subdirectory(mmpack)
target_link_libraries(my_app PRIVATE mmpack::mmpack)
```

Without cmake:

```
make test      # unit tests
make check     # unit tests + corruption fuzzer, under ASan/UBSan
make example   # the routing example, end to end with a real mmap
```

Requires C++20. Tested with Apple Clang 14 on arm64.

## Limits

- **Little-endian hosts only**, enforced by a `static_assert`. Runtime-width
  access reads exactly `width` bytes, which only matches the native integer
  layout on little-endian; a big-endian build would silently produce wrong
  values for the odd widths, so it refuses to compile instead.
- **No compile-time type safety.** Field access is by id with runtime kind
  checks; a wrong `uint` vs `text` call is a `nullopt`, not a build error.
- **Runtime stride and field offsets** cost some lookup throughput against a
  compile-time layout. That is the trade this project exists to make.
- **One value-tuple layout per image.** Field widths and offsets are global;
  only the *reference* width varies per partition (see per-partition remap).
  Per-partition field widths would be a larger change for, on these workloads,
  a much smaller return.
- **Iterators borrow the table**, which borrows the mapping. Keep both alive.
