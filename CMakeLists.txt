cmake_minimum_required(VERSION 3.1)
project(torchinfer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Options can be turned on and off by providing cmake with a -DOPTIONNAME=ON (or OFF) argument.
#    cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DBUILD_SANITIZE=OFF
option(BUILD_SANITIZE  "Adds sanitive flags" ON)
option(BUILD_ARGPARSE  "Argparse" ON)
option(BUILD_SPDLOG    "Logger" ON)

# Compilation flags
set(OPT_FLAGS "-Ofast -march=native -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -pthread -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -Wno-unused-parameter")

if(BUILD_SANITIZE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsanitize=address")
endif()

if (BUILD_ARGPARSE)
    add_subdirectory(third_party/argparse)
endif()

if (BUILD_SPDLOG)
    add_subdirectory(third_party/spdlog)
endif()

# Binary directories
add_subdirectory(targets)

# Status build
message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Sanitize flags       : ${BUILD_SANITIZE}")
message(STATUS "  Argparse             : ${BUILD_ARGPARSE}")
message(STATUS "  Logger               : ${BUILD_SPDLOG}")
message(STATUS "")