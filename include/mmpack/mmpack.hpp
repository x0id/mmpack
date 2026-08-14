#pragma once

/// mmpack -- memory-mapped key -> record lookups whose record shape is chosen
/// from the data rather than declared at compile time.
///
///   #include <mmpack/mmpack.hpp>
///
///   mmpack::schema_builder sb;
///   const auto city = sb.add_text("city");
///   const auto pop  = sb.add_uint("population");
///
///   mmpack::table_builder tb(sb, {.address_bits = 16});
///   for (const auto& row : sorted_rows) {          // keys must arrive in order
///     auto rec = tb.begin_record(row.key);
///     rec.set_text(city, row.city);
///     rec.set_uint(pop, row.population);
///     tb.commit(rec);
///   }
///   mmpack::vector_sink sink;
///   const mmpack::build_report report = tb.finish(sink);
///
///   const auto t = mmpack::table::open(sink.data(), report.image_bytes);
///   const auto it = t.find(some_key);
///   if (it != t.end()) use(t.text(it, city), t.uint(it, pop));

#include "mmpack/builder.hpp"
#include "mmpack/dictionary.hpp"
#include "mmpack/error.hpp"
#include "mmpack/format.hpp"
#include "mmpack/keyed.hpp"
#include "mmpack/schema.hpp"
#include "mmpack/sink.hpp"
#include "mmpack/table.hpp"
