#pragma once

#include "layers.hh"
#include <vector>
#include <sstream>

namespace torchinfer
{
    class Conv2D : public Layers
    {
    public:
        Conv2D(std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg, std::vector<float> &bias_arg, std::vector<int> &dims_bias_arg);
        Conv2D(std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg);

        void forward();

        std::vector<float> weights;
        std::vector<int> dims_weights;
        std::vector<float> bias;
        std::vector<int> dims_bias;

        std::string info();
    };
} // namespace torchinfer