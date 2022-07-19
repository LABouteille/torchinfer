#pragma once

#include <vector>

namespace torchinfer
{
    template <typename T>
    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(std::vector<T> &data_arg, std::vector<int> dims_arg);
        std::vector<T> data;
        std::vector<int> dims;
    };

    template <typename T>
    Tensor<T>::Tensor(std::vector<T> &data_arg, std::vector<int> dims_arg)
        : data(data_arg),
          dims(dims_arg)
    {
    }

} // namespace torchinfer
