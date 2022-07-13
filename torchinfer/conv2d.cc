#include "conv2d.hh"
#include <iostream>

namespace torchinfer
{
    Conv2D::Conv2D(std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg, std::vector<float> &bias_arg, std::vector<int> &dims_bias_arg)
        : weights(weights_arg),
          dims_weights(dims_weights_arg),
          bias(bias_arg),
          dims_bias(dims_bias_arg)
    {
    }

    Conv2D::Conv2D(std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg)
        : weights(weights_arg),
          dims_weights(dims_weights_arg)
    {
    }

    std::string Conv2D::info()
    {
        std::stringstream ss;
        ss << "weights: ";

        for (auto dim : dims_weights)
            ss << dim << " ";
        ss << std::endl;

        if (!dims_bias.empty())
        {
            ss << "bias: ";
            for (auto dim : dims_bias)
                ss << dim << " ";
            ss << std::endl;
        }
        return ss.str();
    }

    // Conv2D::forward() {

    // }

} // namespace torchinfer
