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
        Inputs(std::string &name_arg, std::vector<int> &dims_input_arg);

        std::vector<T> forward(std::vector<T> &x) override;
        std::string info() override;

        std::string name;
        std::vector<int> dims_input;
    };

    template <typename T>
    Inputs<T>::Inputs(std::string &name_arg, std::vector<int> &dims_input_arg)
        : name(name_arg),
          dims_input(dims_input_arg)
    {
    }

    template <typename T>
    std::vector<T> Inputs<T>::forward(std::vector<T> &x)
    {
        spdlog::info("Forward inputs");
        return x;
    }

    template <typename T>
    std::string Inputs<T>::info()
    {
        std::stringstream ss;
        ss << "Inputs: " << name << " [" << dims_input[0] << " " << dims_input[1] << " " << dims_input[2] << " " << dims_input[3] << "]";
        return ss.str();
    }
} // namespace torchinfer
