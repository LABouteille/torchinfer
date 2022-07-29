#pragma once

#include <random>

#include "../src/tensor.hh"

namespace torchinfer
{
    template <typename T>
    Tensor<T> get_uniform_tensor(std::vector<unsigned int> dims)
    {
        std::vector<T> data(dims[0] * dims[1] * dims[2] * dims[3]);
        std::random_device seed;
        std::mt19937 gen(seed()); 
        std::uniform_real_distribution<T> dis(0., 3.L);

        for (size_t i = 0; i < data.size(); i++)
            data[i] = dis(gen);

        return Tensor<T>(data, dims);
    }
}
