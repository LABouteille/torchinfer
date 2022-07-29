#pragma once

#include <string>
#include <vector>

#include "layers.hh"

namespace torchinfer
{
    template <typename T>
    class Inputs : public Layers<T>
    {
    public:
        Inputs(std::string name_arg, std::vector<unsigned int> dims_arg);

        std::string info() override;
        Tensor<T> forward(Tensor<T> &x) override;

        std::string name;
        std::vector<unsigned int> dims;
    };

    template <typename T>
    Inputs<T>::Inputs(std::string name_arg, std::vector<unsigned int> dims_arg)
        : name(name_arg),
          dims(dims_arg)
    {
    }

    template <typename T>
    std::string Inputs<T>::info()
    {
        std::stringstream ss;
        ss << "Inputs: " << name << " [" << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << "]";
        return ss.str();
    }

    template <typename T>
    Tensor<T> Inputs<T>::forward(Tensor<T> &x)
    {
        return x;
    }
} // namespace torchinfer
