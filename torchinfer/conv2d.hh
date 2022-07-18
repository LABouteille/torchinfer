#pragma once

#include "layers.hh"
#include <vector>

namespace torchinfer
{
    template <typename T>
    class Conv2D : public Layers<T>
    {
    public:
        Conv2D(std::string &name_arg, std::vector<T> &weights_arg, std::vector<int> &dims_weights_arg, std::vector<T> &bias_arg, std::vector<int> &dims_bias_arg);
        Conv2D(std::string &name_arg, std::vector<T> &weights_arg, std::vector<int> &dims_weights_arg);

        std::string info() override;
        std::vector<T> forward(std::vector<T> &x) override;

        std::string name;
        std::vector<T> weights;
        std::vector<int> dims_weights;
        std::vector<T> bias;
        std::vector<int> dims_bias;
    };

    template <typename T>
    Conv2D<T>::Conv2D(std::string &name_arg, std::vector<T> &weights_arg, std::vector<int> &dims_weights_arg, std::vector<T> &bias_arg, std::vector<int> &dims_bias_arg)
        : name(name_arg),
          weights(weights_arg),
          dims_weights(dims_weights_arg),
          bias(bias_arg),
          dims_bias(dims_bias_arg)
    {
    }

    template <typename T>
    Conv2D<T>::Conv2D(std::string &name_arg, std::vector<T> &weights_arg, std::vector<int> &dims_weights_arg)
        : name(name_arg),
          weights(weights_arg),
          dims_weights(dims_weights_arg)
    {
    }

    template <typename T>
    std::string Conv2D<T>::info()
    {
        std::stringstream ss;
        ss << "Conv2D: " << name << " [" << dims_weights[0] << " " << dims_weights[1] << " " << dims_weights[2] << " " << dims_weights[3] << " " << dims_bias[0] << "]";
        return ss.str();
    }

    template <typename T>
    std::vector<T> Conv2D<T>::forward(std::vector<T> &x)
    {
        spdlog::info("Forward conv2D");
        return x;
    }

} // namespace torchinfer