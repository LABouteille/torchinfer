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
        void load(const std::string &filename_onnx_ir);
        void summary();

        void predict(std::vector<T> &x);

        std::vector<std::unique_ptr<Layers<T>>> layers;
    };

    template <typename T>
    Model<T>::Model() {}

    template <typename T>
    void Model<T>::load(const std::string &filename_onnx_ir)
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
                - dims (bias)
                - bias
        */
        spdlog::info("filename: {}", filename_onnx_ir);
        std::ifstream file(filename_onnx_ir, std::ios::binary);

        auto nb_layer = read_scalar_from_stream<int>(file, "nb_layer");
        spdlog::info("Nb layer: {}", nb_layer);

        for (int i = 0; i < nb_layer; i++)
        {
            auto layer_id = read_scalar_from_stream<int>(file, "layer_id");
            spdlog::info("Layer id: {}", layer_id);

            auto name_size = read_scalar_from_stream<int>(file, "name_size");

            auto tmp_name = read_vector_from_stream<char>(file, name_size, "name");
            auto name = std::string(tmp_name.begin(), tmp_name.end());
            spdlog::info("Name: {}", name);

            auto op_type = read_scalar_from_stream<int>(file, "op_type");

            if (op_type == static_cast<int>(OPTYPE::INPUT))
            {
                auto dims_input = read_vector_from_stream<int>(file, 4, "dims_input");
                auto layer = Inputs<T>(name, dims_input);

                spdlog::info("Op type: INPUT");
                spdlog::info("\t- dims (input):");
                for (auto elt : layer.dims_input)
                    spdlog::info("\t\t {}", elt);

                this->layers.push_back(std::make_unique<Inputs<T>>(layer));
            }
            else if (op_type == static_cast<int>(OPTYPE::CONV2D))
            {
                auto nb_params = read_scalar_from_stream<int>(file, "nb_params");
                auto dim_weights = read_vector_from_stream<int>(file, 4, "dim_weights");
                auto weights = read_raw_data_from_stream<T>(file, dim_weights, "weights");
                auto dims_bias = read_vector_from_stream<int>(file, 1, "dims_bias");

                spdlog::info("Op type: CONV2D");
                spdlog::info("\t- nb_params: {}", nb_params);
                spdlog::info("\t- dims (weights):");
                for (auto elt : dim_weights)
                    spdlog::info("\t\t {}", elt);
                spdlog::info("\t- weights:");
                for (int i = 0; i < 4 && i < static_cast<int>(weights.size()); i++)
                    spdlog::info("\t\t {} ", weights[i]);
                spdlog::info("\t\t ...");
                spdlog::info("\t- dims (bias):");
                for (auto elt : dims_bias)
                    spdlog::info("\t\t {}", elt);

                if (nb_params == 2)
                {
                    auto bias = read_raw_data_from_stream<T>(file, dims_bias, "bias");
                    spdlog::info("\t- bias:");
                    for (int i = 0; i < 4 && i < static_cast<int>(bias.size()); i++)
                        spdlog::info("\t\t {} ", bias[i]);
                    spdlog::info("\t\t ...");

                    auto layer = Conv2D<T>(name, weights, dim_weights, bias, dims_bias);
                    this->layers.push_back(std::make_unique<Conv2D<T>>(layer));
                }
                else
                {
                    auto layer = Conv2D<T>(name, weights, dim_weights);
                    this->layers.push_back(std::make_unique<Conv2D<T>>(layer));
                }
            }
            else
                throw std::runtime_error("model.load: Layer not implemented yet");
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
    void Model<T>::predict(std::vector<T> &x)
    {
        spdlog::info("Predicting...");
        spdlog::info("Input data size: {}", x.size());

        auto out = this->layers[0]->forward(x);

        for (size_t i = 1; i < this->layers.size(); i++)
            out = this->layers[i]->forward(out);
    }
} // namespace torchinfer