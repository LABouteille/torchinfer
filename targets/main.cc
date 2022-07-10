#include <argparse/argparse.hpp>
#include "spdlog/spdlog.h"
#include <iostream>

#include "options.hh"
#include <torchinfer/conv2d.hh>
#include <torchinfer/model.hh>
#include <torchinfer/io.hh>

int main(int argc, char *argv[])
{
    #if DEBUG_SPDLOG
        spdlog::set_level(spdlog::level::debug);
    #endif

    argparse::ArgumentParser program("torchinfer");
    program.add_argument("--converter")
        .help("Converted Pytorch model")
        .default_value(false);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    spdlog::info("Hello World!");
    spdlog::debug("Debug mode activated!");

    // auto model = torchinfer::Model();
    // model.setup();

    // auto vec = torchinfer::read_bin("../sandbox/array.bin");
    // for (auto elt: vec)
    //     printf("%d\n", elt);
}