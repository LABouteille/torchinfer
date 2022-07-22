#include <argparse/argparse.hpp>
#include "spdlog/spdlog.h"
#include <iostream>

#include "options.hh"
#include <torchinfer/conv2d.hh>
#include <torchinfer/model.hh>
#include <torchinfer/io.hh>
#include <torchinfer/tensor.hh>

template <typename T> 
static void display_tensor(torchinfer::Tensor<T> &out)
{
    std::cout << "Display value of out.data ... " << std::endl;

    for (int i = 0; i < out.dims[0]; i++)
    {
        for (int j = 0; j < out.dims[1]; j++)
        {
            for (int k = 0; k < out.dims[2]; k++)
            {
                for (int l = 0; l < out.dims[3]; l++)
                {
                    std::cout << out[i * out.dims[1] * out.dims[2] * out.dims[3] + j * out.dims[2] * out.dims[3] + k * out.dims[3] + l] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
#if DEBUG_SPDLOG
    spdlog::set_level(spdlog::level::debug);
#endif

    argparse::ArgumentParser program("torchinfer");
    program.add_argument("--input")
        .help("Input data (numpy array dumped as binary)")
        .required();
    program.add_argument("--type")
        .help("Input data type [int, float, double]")
        .required();
    program.add_argument("--onnx_ir")
        .help("ONNX Intermediate Representation (IR)")
        .required();
    program.add_argument("--output")
        .help("Output model prediction")
        .required();
    program.add_argument("--verbose")
        .help("Verbose mode")
        .default_value(false)
        .implicit_value(true);
    
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
    auto filename_input = program.get<std::string>("--input");
    auto data_type = program.get<std::string>("--type");
    auto filename_onnx_ir = program.get<std::string>("--onnx_ir");
    auto filename_output = program.get<std::string>("--output");
    auto verbose = program.get<bool>("--verbose");

    if (data_type == "int")
    {
        auto x = torchinfer::read_numpy_binary<int>(filename_input, data_type);
        auto model = torchinfer::Model<int>();
        model.load(filename_onnx_ir, verbose);
        
        if (verbose)
            model.summary();

        auto input_layer = dynamic_cast<torchinfer::Inputs<int> *>(model.layers[0].get());
        auto input = torchinfer::Tensor<int>(x, input_layer->dims);
        auto out = model.predict(input);
        
        if (verbose)
            display_tensor<int>(out);

        torchinfer::write_numpy_binary<int>(out, filename_output);

    }
    else if (data_type == "float")
    {
        auto x = torchinfer::read_numpy_binary<float>(filename_input, data_type);
        auto model = torchinfer::Model<float>();
        model.load(filename_onnx_ir, verbose);

        if (verbose)
            model.summary();
        
        auto input_layer = dynamic_cast<torchinfer::Inputs<float> *>(model.layers[0].get());
        auto input = torchinfer::Tensor<float>(x, input_layer->dims);
        auto out = model.predict(input);
        
        if (verbose)
            display_tensor<float>(out);
        
        torchinfer::write_numpy_binary<float>(out, filename_output);
    }
    else if (data_type == "double")
    {
        auto x = torchinfer::read_numpy_binary<double>(filename_input, data_type);
        auto model = torchinfer::Model<double>();
        model.load(filename_onnx_ir, verbose);

        if (verbose)
            model.summary();
        
        auto input_layer = dynamic_cast<torchinfer::Inputs<double> *>(model.layers[0].get());
        auto input = torchinfer::Tensor<double>(x, input_layer->dims);
        auto out = model.predict(input);

        if (verbose)
            display_tensor<double>(out);

        torchinfer::write_numpy_binary<double>(out, filename_output);
    }
    else
        throw std::runtime_error("main: Unknown data type");
}