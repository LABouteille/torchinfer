cmake_minimum_required(VERSION 3.1)

if (BUILD_BENCHMARK)
    file(GLOB_RECURSE BENCH 
        ../bench/*.cc
    )
    add_executable(bench ${SRC} ${BENCH})

    target_compile_options(bench PRIVATE -Wall -Wextra -Werror -fsanitize=address -DGOOGLETEST_PATH=../third_party/googletest)
    target_link_options(bench PRIVATE -Wall -Wextra -Werror -fsanitize=address)
    target_link_libraries(bench PRIVATE benchmark::benchmark argparse spdlog)
endif()
