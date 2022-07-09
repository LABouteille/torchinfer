cmake_minimum_required(VERSION 3.1)

include_directories(..)
file(GLOB SRC ../torchinfer/*.cc ../torchinfer/*.hh)
add_executable(${PROJECT_NAME} main.cc ${SRC})