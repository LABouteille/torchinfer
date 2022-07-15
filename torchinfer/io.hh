#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include <numeric>

namespace torchinfer
{
    std::vector<int> read_numpy_binary(std::string filename);
    
    template <typename T>
    T read_scalar_from_stream(std::ifstream &file, std::string param_name)
    {
        T param = -1;
        file.read(reinterpret_cast<char *>(&param), sizeof(T));
        if (param == -1)
            throw std::runtime_error("read_scalar_from_stream: No " + param_name + " dumped in binary");
        return param;
    }

    template <typename T>
    std::vector<T> read_vector_from_stream(std::ifstream &file, int size, std::string param_name)
    {
        std::vector<T> param(size);
        file.read(reinterpret_cast<char *>(param.data()), param.size() * sizeof(T));
        if (param.empty())
            throw std::runtime_error("read_vector_from_stream: No " + param_name + " dumped in binary");
        return param;
    }
    
    template <typename T>
    std::vector<T> read_raw_data_from_stream(std::ifstream &file, std::vector<int> &dims, std::string param_name)
    {
        // Read weight/bias from an already open file stream.

        auto size = std::reduce(dims.cbegin(), dims.cend(), 1, [](int a, int b)
                                { return a * b; });
        return read_vector_from_stream<T>(file, size, param_name);
    }

} // namespace torchinfer
