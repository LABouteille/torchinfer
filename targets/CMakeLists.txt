cmake_minimum_required(VERSION 3.1)

include_directories(..)
file(GLOB SRC ../src/*.cc ../src/*.hh)
add_executable(${PROJECT_NAME} main.cc ${SRC})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=1)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=0)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE argparse spdlog)