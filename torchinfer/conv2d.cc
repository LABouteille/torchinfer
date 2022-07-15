#include "conv2d.hh"
#include <iostream>

namespace torchinfer
{
    Conv2D::Conv2D(std::string &name_arg, std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg, std::vector<float> &bias_arg, std::vector<int> &dims_bias_arg)
        : Layers(name_arg, dims_weights_arg, dims_bias_arg),
          name(name_arg),
          weights(weights_arg),
          dims_weights(dims_weights_arg),
          bias(bias_arg),
          dims_bias(dims_bias_arg)
    {
    }

    Conv2D::Conv2D(std::string &name_arg, std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg)
        : Layers(name_arg, dims_weights_arg),
          name(name_arg),
          weights(weights_arg),
          dims_weights(dims_weights_arg)
    {
    }

    // Conv2D::forward() {

    // }

} // namespace torchinfer
