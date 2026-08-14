# Convenience wrapper so the project builds without cmake.
#   make test      unit tests
#   make check     unit tests + corruption fuzzer, both under ASan/UBSan
#   make example   build and run the IPv4 example
#   make all       everything

CXX      ?= c++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -O2
INCLUDES  = -Iinclude
SANFLAGS  = -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1
BUILD     = build

HEADERS = $(wildcard include/mmpack/*.hpp include/mmpack/detail/*.hpp)

.PHONY: all test check example fuzz asan clean

all: test example fuzz

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/test_mmpack: tests/test_mmpack.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BUILD)/test_mmpack_asan: tests/test_mmpack.cpp $(HEADERS) | $(BUILD)
	$(CXX) -std=c++20 -Wall -Wextra $(SANFLAGS) $(INCLUDES) -o $@ $<

$(BUILD)/fuzz_open: tests/fuzz_open.cpp $(HEADERS) | $(BUILD)
	$(CXX) -std=c++20 -Wall -Wextra $(SANFLAGS) $(INCLUDES) -o $@ $<

$(BUILD)/ipv4_routes: examples/ipv4_routes.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

test: $(BUILD)/test_mmpack
	./$<

asan: $(BUILD)/test_mmpack_asan
	./$<

fuzz: $(BUILD)/fuzz_open
	./$< 20000

example: $(BUILD)/ipv4_routes
	./$<

# What CI should run: correctness, then memory safety against corrupt input.
check: asan fuzz

clean:
	rm -rf $(BUILD)
