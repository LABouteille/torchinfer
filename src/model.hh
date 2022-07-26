#pragma once

#include <string>
#include <fstream>
#include <memory>
#include "spdlog/spdlog.h"

#include "io.hh"
#include "layers.hh"
#include "inputs.hh"
#include "conv2d.hh"

namespace torchinfer
{
    template <typename T>
    class Model
    {
    public:
        Model();
        void add(Layers<T> *layer);
        void load(const std::string &filename_onnx_ir, const bool &verbose);
        void compile();

        void summary();

        Tensor<T> predict(Tensor<T> &x);

        std::vector<std::unique_ptr<Layers<T>>> layers;
    };

    template <typename T>
    Model<T>::Model() {}

    template <typename T>
    void Model<T>::add(Layers<T> *layer)
    {
        layers.push_back(std::unique_ptr<Layers<T>>(layer));
    }

    template <typename T>
    void Model<T>::load(const std::string &filename_onnx_ir, const bool &verbose)
    {
        /*
        - Nb_layer
        for all layers:
            - layer_id
            - name size
            - name
            - op_type
            if Input:
                - dims
            if Conv2d:
                - nb_params
                - dims (weight)
                - weight
                    - strides
                - dims (bias)
                - bias
        */
        // Load filename
        if (verbose)
            spdlog::info("filename: {}", filename_onnx_ir);
        std::ifstream file(filename_onnx_ir, std::ios::binary);

        if (!file.is_open())
            spdlog::error("model.load: file not opened");

        auto nb_layer = read_scalar_from_stream<int>(file, "nb_layer");

        if (verbose)
            spdlog::info("Nb layer: {}", nb_layer);

        if (verbose)
        {
            if (std::is_same<T, int>::value)
                spdlog::info("Type: {}", "INT");
            else if (std::is_same<T, float>::value)
                spdlog::info("Type: {}", "FLOAT");
            else if (std::is_same<T, double>::value)
                spdlog::info("Type: {}", "DOUBLE");
            else
                throw std::runtime_error("model.load: Not supported type");
        }

        for (int i = 0; i < nb_layer; i++)
        {
            auto layer_id = read_scalar_from_stream<int>(file, "layer_id");
            if (verbose)
                spdlog::info("Layer id: {}", layer_id);

            auto name_size = read_scalar_from_stream<int>(file, "name_size");

            auto tmp_name = read_vector_from_stream<char>(file, name_size, "name");
            auto name = std::string(tmp_name.begin(), tmp_name.end());
            if (verbose)
                spdlog::info("Name: {}", name);

            auto op_type = read_scalar_from_stream<int>(file, "op_type");

            if (op_type == static_cast<int>(OPTYPE::INPUT))
            {
                auto dims_input = read_vector_from_stream<int>(file, 4, "dims_input");

                if (verbose)
                {
                    spdlog::info("Op type: INPUT");
                    spdlog::info("\t- dims (input):");
                    for (auto elt : dims_input)
                        spdlog::info("\t\t {}", elt);
                }

                this->add(new Inputs<T>(name, dims_input));
            }
            else if (op_type == static_cast<int>(OPTYPE::CONV2D))
            {
                auto nb_params = read_scalar_from_stream<int>(file, "nb_params");
                auto dim_weights = read_vector_from_stream<int>(file, 4, "dim_weights");
                auto weights = read_raw_data_from_stream<T>(file, dim_weights, "weights");
                auto strides = read_vector_from_stream<int>(file, 2, "strides");
                if (verbose)
                {
                    spdlog::info("Op type: CONV2D");
                    spdlog::info("\t- nb_params: {}", nb_params);
                    spdlog::info("\t- dims (weights):");
                    for (auto elt : dim_weights)
                        spdlog::info("\t\t {}", elt);
                    spdlog::info("\t- weights:");
                    for (int i = 0; i < 4 && i < static_cast<int>(weights.size()); i++)
                        spdlog::info("\t\t {} ", weights[i]);
                    spdlog::info("\t\t ...");
                    spdlog::info("\t- strides:");
                    for (auto elt : strides)
                        spdlog::info("\t\t {}", elt);
                }

                if (nb_params == 2)
                {
                    auto dims_bias = read_vector_from_stream<int>(file, 1, "dims_bias");
                    auto bias = read_raw_data_from_stream<T>(file, dims_bias, "bias");

                    if (verbose)
                    {
                        spdlog::info("\t- dims (bias):");
                        for (auto elt : dims_bias)
                            spdlog::info("\t\t {}", elt);
                        spdlog::info("\t- bias:");
                        for (int i = 0; i < 4 && i < static_cast<int>(bias.size()); i++)
                            spdlog::info("\t\t {} ", bias[i]);
                        spdlog::info("\t\t ...");
                    }

                    this->add(new Conv2D<T>(name, Tensor<T>(weights, dim_weights), Tensor<T>(bias, dims_bias), strides));
                }
                else
                {
                    this->add(new Conv2D<T>(name, Tensor<T>(weights, dim_weights), strides));
                }
            }
            else
                throw std::runtime_error("model.load: Layer not implemented yet");
        }
    }

    template <typename T>
    void Model<T>::compile()
    {
        // Setup layers output dims/data of layers
        auto ptr = this->layers[0].get();

        for (int i = 1; i < static_cast<int>(this->layers.size()); i++)
        {
            if (auto layer = dynamic_cast<Conv2D<T> *>(this->layers[i].get()))
            {
                int batch, height, width;

                if (auto prev_layer = dynamic_cast<Inputs<T> *>(ptr))
                {
                    batch = prev_layer->dims[0];
                    height = prev_layer->dims[2];
                    width = prev_layer->dims[3];
                }
                else if (auto prev_layer = dynamic_cast<Conv2D<T> *>(ptr))
                {
                    batch = prev_layer->out.dims[0];
                    height = prev_layer->out.dims[2];
                    width = prev_layer->out.dims[3];
                }
                else
                    throw std::runtime_error("model.compile: Layer not implemented yet");

                int nb_filters = layer->weights.dims[0];
                int kernel_height = layer->weights.dims[2];
                int kernel_width = layer->weights.dims[3];

                int out_height = std::floor(((height - kernel_height) / layer->strides[0]) + 1);
                int out_width = std::floor(((width - kernel_width) / layer->strides[1]) + 1);

                layer->out.dims = {batch, nb_filters, out_height, out_width};
                layer->out.data.assign(batch * nb_filters * out_height * out_width, (T)0);
            }
            else
                continue;

            ptr = this->layers[i].get();
        }
    }

    template <typename T>
    void Model<T>::summary()
    {
        std::stringstream info;

        spdlog::info("Model Summary:");
        info << std::endl;

        for (auto &layer : this->layers)
        {
            info << layer->info() << std::endl;
        }

        spdlog::info(info.str());
    }

    template <typename T>
    Tensor<T> Model<T>::predict(Tensor<T> &x)
    {
        auto input_layer = dynamic_cast<Inputs<T> *>(this->layers[0].get());
        if (x.dims != input_layer->dims)
            throw std::runtime_error("model.predict: Input data size does not match Input layer dims.");

        auto out = x;
        for (size_t i = 0; i < this->layers.size(); i++)
            out = this->layers[i]->forward(out);

        return out;
    }
} // namespace torchinfer