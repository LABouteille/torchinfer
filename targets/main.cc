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
    program.add_argument("--data")
        .help("Input data (numpy array dumped as binary)")
        .default_value(false);

    program.add_argument("--onnx_ir")
        .help("ONNX Intermediate Representation (IR)")
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

    // TODO: check if proper file were given to the right param.
    auto filename_data = program.get<std::string>("--data");
    auto filename_onnx_ir = program.get<std::string>("--onnx_ir");
    
    auto x = torchinfer::read_numpy_binary(filename_data);
    auto model = torchinfer::Model();
    model.load(filename_onnx_ir);
    model.summary();
}