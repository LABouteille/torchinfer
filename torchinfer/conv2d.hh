#pragma once

#include "layers.hh"
#include <vector>

namespace torchinfer
{
    class Conv2D : public Layers
    {
    public:
        Conv2D(std::string &name_arg, std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg, std::vector<float> &bias_arg, std::vector<int> &dims_bias_arg);
        Conv2D(std::string &name_arg, std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg);

        void forward();

        std::string name;
        std::vector<float> weights;
        std::vector<int> dims_weights;
        std::vector<float> bias;
        std::vector<int> dims_bias;

    };
} // namespace torchinfer