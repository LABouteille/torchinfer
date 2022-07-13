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
    program.add_argument("--input")
        .help("Pytorch Intermediate Representation (IR)")
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

    auto filename = program.get<std::string>("--input");
    
    auto model = torchinfer::Model();
    model.setup(filename);
}