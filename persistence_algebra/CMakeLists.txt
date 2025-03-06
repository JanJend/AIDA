# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(Aida)

# Set the C++ compiler to g++-10? Doesnt seem like a good idea
# set(CMAKE_CXX_COMPILER "g++-10")

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
include_directories(graded_linear_algebra/include)

# Find Boost libraries
find_package(Boost REQUIRED timer chrono)

# Link Boost libraries
link_libraries(Boost::timer Boost::chrono)

add_executable(aida ${CMAKE_SOURCE_DIR}/aida.cpp)
add_executable(aida_with_stats ${CMAKE_SOURCE_DIR}/aida.cpp)
add_executable(aida_without_stats ${CMAKE_SOURCE_DIR}/aida.cpp)

# For Decomposition of Vectorspaces
add_executable(generate_decompositions ${CMAKE_SOURCE_DIR}/generate_decompositions.cpp)

target_compile_options(aida_with_stats PUBLIC -DAIDA_WITH_STATS=1)
target_compile_options(aida_without_stats PUBLIC -DAIDA_WITH_STATS=0)

