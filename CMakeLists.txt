cmake_minimum_required(VERSION 3.16)
project(mmpack VERSION 0.1.0 LANGUAGES CXX)

# Header-only: consumers just need the include path and C++20.
add_library(mmpack INTERFACE)
add_library(mmpack::mmpack ALIAS mmpack)

target_include_directories(mmpack INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(mmpack INTERFACE cxx_std_20)

# Only build tests/examples when mmpack is the top-level project, so that
# add_subdirectory() from a parent project stays quiet.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  option(MMPACK_BUILD_TESTS "Build mmpack tests" ON)
  option(MMPACK_BUILD_EXAMPLES "Build mmpack examples" ON)
  option(MMPACK_SANITIZE "Build tests with ASan/UBSan" OFF)

  add_library(mmpack_warnings INTERFACE)
  target_compile_options(mmpack_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>)

  if(MMPACK_SANITIZE)
    target_compile_options(mmpack_warnings INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(mmpack_warnings INTERFACE -fsanitize=address,undefined)
  endif()

  if(MMPACK_BUILD_TESTS)
    enable_testing()

    add_executable(test_mmpack tests/test_mmpack.cpp)
    target_link_libraries(test_mmpack PRIVATE mmpack::mmpack mmpack_warnings)
    add_test(NAME mmpack.unit COMMAND test_mmpack)

    # The corruption harness only proves anything under a sanitizer, but it is
    # cheap enough to run either way as a smoke test.
    add_executable(fuzz_open tests/fuzz_open.cpp)
    target_link_libraries(fuzz_open PRIVATE mmpack::mmpack mmpack_warnings)
    add_test(NAME mmpack.fuzz_open COMMAND fuzz_open 5000)
  endif()

  if(MMPACK_BUILD_EXAMPLES AND UNIX)
    add_executable(ipv4_routes examples/ipv4_routes.cpp)
    target_link_libraries(ipv4_routes PRIVATE mmpack::mmpack mmpack_warnings)
  endif()
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS mmpack EXPORT mmpackTargets)
install(EXPORT mmpackTargets
  FILE mmpackConfig.cmake
  NAMESPACE mmpack::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mmpack)
