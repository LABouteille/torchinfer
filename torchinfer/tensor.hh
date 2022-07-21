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

        T &operator[](int idx); // Writing
        T operator[](int idx) const; // Reading

        std::vector<T> data;
        std::vector<int> dims;
    };

    template <typename T>
    Tensor<T>::Tensor(std::vector<T> &data_arg, std::vector<int> dims_arg)
        : data(data_arg),
          dims(dims_arg)
    {
    }

    template <typename T>
    T &Tensor<T>::operator[](int idx)
    {
        return data[idx];
    }

    template <typename T>
    T Tensor<T>::operator[](int idx) const
    {
        return data[idx];
    }

} // namespace torchinfer