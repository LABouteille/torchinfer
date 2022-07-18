#include "conv2d.hh"
#include <iostream>

namespace torchinfer
{
  Conv2D::Conv2D(std::string &name_arg, std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg, std::vector<float> &bias_arg, std::vector<int> &dims_bias_arg)
      : name(name_arg),
        weights(weights_arg),
        dims_weights(dims_weights_arg),
        bias(bias_arg),
        dims_bias(dims_bias_arg)
  {
  }

  Conv2D::Conv2D(std::string &name_arg, std::vector<float> &weights_arg, std::vector<int> &dims_weights_arg)
      : name(name_arg),
        weights(weights_arg),
        dims_weights(dims_weights_arg)
  {
  }

    // Conv2D::forward() {

  std::string Conv2D::info()
  {
    std::stringstream ss;
    ss << "Conv2D: " << name << " [" << dims_weights[0] << " " << dims_weights[1] << " " << dims_weights[2] << " " << dims_weights[3] << " " << dims_bias[0]  << "]";
    return ss.str();
  }

} // namespace torchinfer
