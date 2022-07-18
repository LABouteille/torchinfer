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
        .required();
    program.add_argument("--type")
        .help("Input data type [int, float, double]")
        .required();
    program.add_argument("--onnx_ir")
        .help("ONNX Intermediate Representation (IR)")
        .required();

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
    auto data_type = program.get<std::string>("--type");
    auto filename_onnx_ir = program.get<std::string>("--onnx_ir");

    if (data_type == "int")
    {
        auto x = torchinfer::read_numpy_binary<int>(filename_data, data_type);
        auto model = torchinfer::Model<int>();
        model.load(filename_onnx_ir);
        model.summary();
        model.predict(x);
    }
    else if (data_type == "float")
    {
        auto x = torchinfer::read_numpy_binary<float>(filename_data, data_type);
        auto model = torchinfer::Model<float>();
        model.load(filename_onnx_ir);
        model.summary();
        model.predict(x);
    }
    else if (data_type == "double")
    {
        auto x = torchinfer::read_numpy_binary<double>(filename_data, data_type);
        auto model = torchinfer::Model<double>();
        model.load(filename_onnx_ir);
        model.summary();
        model.predict(x);
    }
    else
        throw std::runtime_error("main: Unknown data type");
}