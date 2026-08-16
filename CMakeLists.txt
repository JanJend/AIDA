# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(Aida CXX)

# C++ standard (overridable on the command line for testing newer standards)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT CMAKE_VERBOSE_MAKEFILE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# Portable warning flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
    # MSVC needs this to report the correct __cplusplus value
    add_compile_options(/Zc:__cplusplus)
else()
    add_compile_options(-fdiagnostics-color=always)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Treat Persistence-Algebra as a system include — suppresses warnings from its headers
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/../Persistence-Algebra/include)

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS timer chrono)

# OpenMP is optional: used when available, ignored otherwise.
# Apple clang ships no OpenMP runtime, so point CMake at Homebrew's libomp if present.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE _brew_libomp
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_brew_libomp AND EXISTS "${_brew_libomp}")
        set(OpenMP_ROOT "${_brew_libomp}")
    endif()
endif()
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

# Collect source files
set(AIDA_SOURCES
    ${CMAKE_SOURCE_DIR}/src/aida_decompose.cpp
    ${CMAKE_SOURCE_DIR}/src/aida_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/aida_helpers.cpp
    ${CMAKE_SOURCE_DIR}/src/aida_interface.cpp
    ${CMAKE_SOURCE_DIR}/src/config.cpp
    ${CMAKE_SOURCE_DIR}/src/option_parser.cpp
    ${CMAKE_SOURCE_DIR}/src/block.cpp
)

# Static library
add_library(aida_static STATIC ${AIDA_SOURCES})
target_include_directories(aida_static PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(aida_static SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/../Persistence-Algebra/include
)
target_link_libraries(aida_static PUBLIC Boost::timer Boost::chrono)
set_target_properties(aida_static PROPERTIES OUTPUT_NAME aida)

install(TARGETS aida_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Executables
add_executable(aida ${CMAKE_SOURCE_DIR}/aida.cpp ${AIDA_SOURCES})
target_link_libraries(aida PRIVATE Boost::timer Boost::chrono)
target_compile_definitions(aida PUBLIC AIDA_WITH_STATS=0)

add_executable(aida_with_stats ${CMAKE_SOURCE_DIR}/aida.cpp ${AIDA_SOURCES})
target_link_libraries(aida_with_stats PRIVATE Boost::timer Boost::chrono)
target_compile_definitions(aida_with_stats PUBLIC AIDA_WITH_STATS=1)

add_executable(generate_decompositions ${CMAKE_SOURCE_DIR}/generate_decompositions.cpp)
target_link_libraries(generate_decompositions PRIVATE aida_static)

install(TARGETS aida aida_with_stats generate_decompositions
    RUNTIME DESTINATION bin
)