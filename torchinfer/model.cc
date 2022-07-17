#include "model.hh"

#include "spdlog/spdlog.h"
#include <iostream>

namespace torchinfer
{
    Model::Model() {}

    void Model::load(const std::string &filename)
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

        spdlog::info("filename: {}", filename);

        std::ifstream file(filename, std::ios::binary);

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
                spdlog::info("Op type: INPUT");
                auto dims_input = read_vector_from_stream<int>(file, 4, "dims_input");
                spdlog::info("\t- dims (input):");
                for (auto elt : dims_input)
                    spdlog::info("\t\t {}", elt);

                auto layer = Inputs(name, dims_input);
                this->layers.push_back(std::make_unique<Layers>(layer));
            }
            else if (op_type == static_cast<int>(OPTYPE::CONV2D))
            {
                spdlog::info("Op type: CONV2D");

                auto nb_params = read_scalar_from_stream<int>(file, "nb_params");
                spdlog::info("\t- nb_params: {}", nb_params);

                auto dim_weights = read_vector_from_stream<int>(file, 4, "dim_weights");
                spdlog::info("\t- dims (weights):");
                for (auto elt : dim_weights)
                    spdlog::info("\t\t {}", elt);

                auto weights = read_raw_data_from_stream<float>(file, dim_weights, "weights");
                spdlog::info("\t- weights:");
                for (int i = 0; i < 4 && i < static_cast<int>(weights.size()); i++)
                    spdlog::info("\t\t {} ", weights[i]);
                spdlog::info("\t\t ...");

                auto dims_bias = read_vector_from_stream<int>(file, 1, "dims_bias");
                spdlog::info("\t- dims (bias):");
                for (auto elt : dims_bias)
                    spdlog::info("\t\t {}", elt);

                if (nb_params == 2)
                {
                    auto bias = read_raw_data_from_stream<float>(file, dims_bias, "bias");
                    spdlog::info("\t- bias:");
                    for (int i = 0; i < 4 && i < static_cast<int>(bias.size()); i++)
                        spdlog::info("\t\t {} ", bias[i]);
                    spdlog::info("\t\t ...");

                    auto layer = Conv2D(name, weights, dim_weights, bias, dims_bias);
                    this->layers.push_back(std::make_unique<Layers>(layer));
                }
                else {
                    auto layer = Conv2D(name, weights, dim_weights);
                    this->layers.push_back(std::make_unique<Layers>(layer));
                }
            }
            else
                throw std::runtime_error("model.load: Layer not implemented yet");
        }
    }

    void Model::summary()
    {
        std::stringstream info;

        spdlog::info("Model Summary:");

        for (auto &layer: this->layers) {
            info << layer->info();
        }

        spdlog::info(info.str());
    }

} // namespace torchinfer
