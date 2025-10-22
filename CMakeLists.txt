# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(Aida)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# general compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fdiagnostics-color=always")

# debugging flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

if (NOT CMAKE_VERBOSE_MAKEFILE)
   set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/Persistence-Algebra/include
)

# Find Boost libraries
find_package(Boost REQUIRED timer chrono)

# Link Boost libraries
link_libraries(Boost::timer Boost::chrono)

# Collect source files
set(AIDA_SOURCES
    ${CMAKE_SOURCE_DIR}/src/aida_decompose.cpp
    ${CMAKE_SOURCE_DIR}/src/aida_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/aida_helpers.cpp
    ${CMAKE_SOURCE_DIR}/src/aida_interface.cpp
    ${CMAKE_SOURCE_DIR}/src/config.cpp
    ${CMAKE_SOURCE_DIR}/src/option_parser.cpp
)

# Create executables with all sources
add_executable(aida ${CMAKE_SOURCE_DIR}/aida.cpp ${AIDA_SOURCES})
add_executable(aida_with_stats ${CMAKE_SOURCE_DIR}/aida.cpp ${AIDA_SOURCES})

# For Decomposition of Vectorspaces temporarily deactivated
add_executable(generate_decompositions ${CMAKE_SOURCE_DIR}/generate_decompositions.cpp)

target_compile_options(aida PUBLIC -DAIDA_WITH_STATS=0)
target_compile_options(aida_with_stats PUBLIC -DAIDA_WITH_STATS=1)