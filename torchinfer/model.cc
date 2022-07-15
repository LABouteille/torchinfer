#include "model.hh"
#include <numeric>

#include "spdlog/spdlog.h"
#include <iostream>

#define GET_VARIABLE_NAME(Variable) (#Variable)

template <typename T>
static void read_param_scalar(std::ifstream &file, T &param, std::string param_name)
{
    file.read(reinterpret_cast<char *>(&param), sizeof(T));
    if (param == -1)
        throw std::runtime_error("model.load(): No " + param_name + " dumped in binary");
}

static  void read_param_string(std::ifstream &file, std::string &param, int &size, std::string param_name)
{
    std::vector<char> tmp(size);
    file.read(reinterpret_cast<char *>(tmp.data()), tmp.size() * sizeof(char));
    param = std::string(tmp.begin(), tmp.end());
    if (param.empty())
        throw std::runtime_error("model.load(): No " + param_name + " dumped in binary");
}

template <typename T>
static void read_param_vector(std::ifstream &file, std::vector<T> &param, std::string param_name)
{
    file.read(reinterpret_cast<char *>(param.data()), param.size() * sizeof(T));
    if (param.empty())
        throw std::runtime_error("model.load(): No " + param_name + " dumped in binary");
}

template <typename T>
static std::vector<T> read_raw_data(std::ifstream &file, std::vector<int> &dims, std::string param_name)
{
    // Read weight/bias from an already open file stream.

    auto size = std::reduce(dims.cbegin(), dims.cend(), 1, [](int a, int b)
                            { return a * b; });
    std::vector<T> vec(size);
    file.read(reinterpret_cast<char *>(vec.data()), size * sizeof(T));
    if (vec.empty())
        throw std::runtime_error("model.load(): No " + param_name + " dumped in binary");
    return vec;
}

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
            if Conv2d:
                - nb_params
                - dims (weight)
                - weight
                - dims (bias)
                - bias
        */

        spdlog::info("filename: {}", filename);

        std::ifstream file(filename, std::ios::binary);

        int nb_layer = -1;
        read_param_scalar<int>(file, nb_layer, GET_VARIABLE_NAME(nb_layer));
        spdlog::info("Nb layer: {}", nb_layer);

        for (int i = 0; i < nb_layer; i++)
        {
            int layer_id = -1;
            read_param_scalar<int>(file, layer_id, GET_VARIABLE_NAME(layer_id));
            spdlog::info("Layer id: {}", layer_id);

            int name_size = -1;
            read_param_scalar<int>(file, name_size, GET_VARIABLE_NAME(name_size));

            std::string name;
            read_param_string(file, name, name_size, GET_VARIABLE_NAME(name));
            spdlog::info("Name: {}", name);

            int op_type = -1;
            read_param_scalar<int>(file, op_type, GET_VARIABLE_NAME(op_type));

            if (op_type == static_cast<int>(OPTYPE::CONV2D))
            {
                spdlog::info("Op type: CONV2D");

                int nb_params = -1;
                read_param_scalar<int>(file, nb_params, GET_VARIABLE_NAME(nb_params));
                spdlog::info("\t- nb_params: {}", nb_params);

                std::vector<int> dim_weights(4);
                read_param_vector<int>(file, dim_weights, GET_VARIABLE_NAME(dim_weights));
                spdlog::info("\t- dims (weights):");
                for (auto elt : dim_weights)
                    spdlog::info("\t\t {}", elt);

                auto weights = read_raw_data<float>(file, dim_weights, GET_VARIABLE_NAME(weights));
                spdlog::info("\t- weights:");
                for (int i = 0; i < 4 && i < static_cast<int>(weights.size()); i++)
                    spdlog::info("\t\t {} ", weights[i]);
                spdlog::info("\t\t ...");

                std::vector<int> dims_bias(1);
                read_param_vector<int>(file, dims_bias, GET_VARIABLE_NAME(dims_bias));
                spdlog::info("\t- dims (bias):");
                for (auto elt : dims_bias)
                    spdlog::info("\t\t {}", elt);

                std::vector<float> bias;

                if (nb_params == 2)
                {
                    std::vector<float> bias = read_raw_data<float>(file, dims_bias, GET_VARIABLE_NAME(bias));
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
                throw std::runtime_error("model.load(): Layer not implemented yet");
        }
    }

    void Model::summary()
    {
        spdlog::info("Model Summary:");

        for (auto &layer: this->layers) {
            spdlog::info(layer->info());
        }
    }

} // namespace torchinfer
