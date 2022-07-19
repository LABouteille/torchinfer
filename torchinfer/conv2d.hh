#pragma once

#include "layers.hh"
#include <vector>

namespace torchinfer
{
    template <typename T>
    class Conv2D : public Layers<T>
    {
    public:
        Conv2D(std::string &name_arg, Tensor<T> weights_arg, Tensor<T> bias_arg);
        Conv2D(std::string &name_arg, Tensor<T> weights_arg);

        std::string info() override;
        Tensor<T> forward(Tensor<T> &x) override;

        std::string name;
        Tensor<T> weights;
        Tensor<T> bias;
    };

    template <typename T>
    Conv2D<T>::Conv2D(std::string &name_arg, Tensor<T> weights_arg, Tensor<T> bias_arg)
        : name(name_arg),
          weights(weights_arg),
          bias(bias_arg)
    {
    }

    template <typename T>
    Conv2D<T>::Conv2D(std::string &name_arg, Tensor<T> weights_arg)
        : name(name_arg),
          weights(weights_arg)
    {
    }

    template <typename T>
    std::string Conv2D<T>::info()
    {
        std::stringstream ss;
        ss << "Conv2D: " << name;
        ss << " [" << weights.dims[0] << " " << weights.dims[1] << " " << weights.dims[2] << " " << weights.dims[3];
        ss << " [" << bias.dims[0] << "]";
        return ss.str();
    }

    template <typename T>
    Tensor<T> Conv2D<T>::forward(Tensor<T> &x)
    {
        spdlog::info("Forward conv2D");
        return x;
    }

} // namespace torchinfer