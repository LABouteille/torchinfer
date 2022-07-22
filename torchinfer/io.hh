#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include <numeric>

namespace torchinfer
{
    template <typename T>
    std::vector<T> read_numpy_binary(const std::string &filename, const std::string &data_type)
    {
        /*
         * Reads a numpy binary file (dumped with write_bin()) and returns a vector of type T.
         *
         * File format:
         *    - n (int)
         *    - c (int)
         *    - h (int)
         *    - w (int)
         *    - format [int, float, double] (byte)
         *    - data
         *
         * Parameters:
         *   filename: path to the numpy binary file
         *   data_type: data type of the numpy binary file
         *
         * Returns:
         *   A vector of type T.
         */

        std::map<std::string, char> datatype_to_format = {
            {"int", 'i'},
            {"float", 'f'},
            {"double", 'd'}};

        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("read_numpy_binary: file not opened");

        // Dimensions
        int n = -1, c = -1, h = -1, w = -1;
        file.read(reinterpret_cast<char *>(&n), sizeof(int));
        file.read(reinterpret_cast<char *>(&c), sizeof(int));
        file.read(reinterpret_cast<char *>(&h), sizeof(int));
        file.read(reinterpret_cast<char *>(&w), sizeof(int));

        if (n == -1 || c == -1 || h == -1 || w == -1)
            throw std::runtime_error("read_numpy_binary: No dimensions (n,c,h,w) dumped in binary");

        // Format
        char format = '\0';
        file.read(reinterpret_cast<char *>(&format), sizeof(char));

        if (format == '\0')
            throw std::runtime_error("read_numpy_binary: No format character dumped in binary");

        if (datatype_to_format[data_type] != format)
            throw std::runtime_error("read_numpy_binary: Specified data type does not match dumped data type");
        
        std::vector<T> vec(n * c * h * w);
        file.read(reinterpret_cast<char *>(vec.data()), n * c * h * w * sizeof(T));
        return vec;
    }

    template <typename T>
    void write_numpy_binary(Tensor<T> &tensor, const std::string &filename)
    {
        /*
            * Writes a tensor following numpy binary format.
            *
            * File format:
            *    - n (int)
            *    - c (int)
            *    - h (int)
            *    - w (int)
            *    - format [int, float, double] (byte)
            *    - data
            *
            * Parameters:
            *   tensor: tensor to be dumped
            *   filename: path to the numpy binary file
        */ 

        std::ofstream file(filename, std::ios::binary);
        
        if (!file.is_open())
            throw std::runtime_error("write_numpy_binary: file not opened");

        // Dimensions
        int n = tensor.dims[0];
        int c = tensor.dims[1];
        int h = tensor.dims[2];
        int w = tensor.dims[3];

        file.write(reinterpret_cast<char *>(&n), sizeof(int));
        file.write(reinterpret_cast<char *>(&c), sizeof(int));
        file.write(reinterpret_cast<char *>(&h), sizeof(int));
        file.write(reinterpret_cast<char *>(&w), sizeof(int));

        // format
        if (std::is_same<T, int>::value) {
            char format = 'i';
            file.write(reinterpret_cast<char *>(&format), sizeof(char));
        }
        else if (std::is_same<T, float>::value)
        {
            char format = 'f';
            file.write(reinterpret_cast<char *>(&format), sizeof(char));
        }
        else if (std::is_same<T, double>::value) {
            char format = 'd';
            file.write(reinterpret_cast<char *>(&format), sizeof(char));
        }
        else
            throw std::runtime_error("write_numpy_binary: Unsupported data type");

        // data
        file.write(reinterpret_cast<char *>(tensor.data.data()), n * c * h * w * sizeof(T));

        file.close();
    }

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
