cmake_minimum_required(VERSION 3.20)
project(PAIRS VERSION 1.0.0 LANGUAGES CXX)

if(MSVC)
  message(FATAL_ERROR
    "RCFS currently supports GCC or Clang, including Windows MinGW. "
    "The MSVC toolchain has not been validated.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RCFS_BUILD_TESTS "Enable the Python release smoke test" ON)
if(RCFS_BUILD_TESTS)
  enable_testing()
endif()
set(RCFS_ARCH "native" CACHE STRING "CPU architecture for release optimization (for example native or haswell)")

add_library(rcfs_common STATIC
  src/io/distance.cpp
  src/io/dataset_io.cpp
  src/io/predicate_io.cpp
  src/query/predicate_compiler.cpp
)
target_include_directories(rcfs_common
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_library(rcfs_index STATIC
  src/index/attribute_order.cpp
  src/index/posting_index.cpp
  src/index/range_routing_index.cpp
  src/index/index_builder.cpp
  src/io/index_io.cpp
)
target_include_directories(rcfs_index
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
find_package(Threads REQUIRED)
target_link_libraries(rcfs_index PUBLIC rcfs_common Threads::Threads)

add_library(rcfs_search STATIC
  src/query/quality.cpp
  src/query/planner.cpp
  src/search/searcher.cpp
)
target_include_directories(rcfs_search
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(rcfs_search PUBLIC rcfs_index rcfs_common)
target_link_libraries(rcfs_search PUBLIC Threads::Threads)

set(RCFS_LIBRARY_TARGETS rcfs_common rcfs_index rcfs_search)
foreach(target IN LISTS RCFS_LIBRARY_TARGETS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<CONFIG:Release>:-O3>
      $<$<CONFIG:Release>:-march=${RCFS_ARCH}>
      -Wall -Wextra -Wpedantic
    )
  endif()
endforeach()

add_executable(build_pairs_index src/cli/build_rcfs_index.cpp)
target_include_directories(build_pairs_index PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(build_pairs_index PRIVATE rcfs_index)

add_executable(search_pairs_index src/cli/search_rcfs_index.cpp)
target_include_directories(search_pairs_index PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(search_pairs_index PRIVATE rcfs_search)

add_executable(generate_pairs_groundtruth src/cli/generate_rcfs_groundtruth.cpp)
target_include_directories(generate_pairs_groundtruth PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(generate_pairs_groundtruth PRIVATE rcfs_search)

install(TARGETS
  rcfs_common rcfs_index rcfs_search
  build_pairs_index search_pairs_index generate_pairs_groundtruth
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES README.md LICENSE requirements.txt
  DESTINATION share/pairs)
install(DIRECTORY docs scripts configs DESTINATION share/pairs
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

# The portable release smoke test needs only Python's standard library.
if(RCFS_BUILD_TESTS)
  find_package(Python3 3.9 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME pairs_release_smoke COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/smoke_test.py"
      --build-dir "$<TARGET_FILE_DIR:search_pairs_index>"
      --work-dir "${CMAKE_CURRENT_BINARY_DIR}/release-smoke")
    set_tests_properties(pairs_release_smoke PROPERTIES TIMEOUT 180)
  endif()
endif()
