#pragma once

#include <vector>

namespace torchinfer
{
    template <typename T>
    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(std::vector<T> &data_arg, std::vector<unsigned int> dims_arg);

        T &operator[](int idx);      // Writing
        T operator[](int idx) const; // Reading

        std::string to_string() const;

        std::vector<T> data;
        std::vector<unsigned int> dims;
    };

    template <typename T>
    Tensor<T>::Tensor(std::vector<T> &data_arg, std::vector<unsigned int> dims_arg)
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

    template <typename T>
    std::string Tensor<T>::to_string() const
    {
        unsigned int batch = this->dims[0];
        unsigned int channel = this->dims[1];
        unsigned int height = this->dims[2];
        unsigned int width = this->dims[3];
        
        std::stringstream ss;
        
        ss << std::endl;

        for (unsigned int n = 0; n < batch; n++)
        {
            for (unsigned int c = 0; c < channel; c++)
            {
                for (unsigned int i = 0; i < height; i++)
                {
                    for (unsigned int j = 0; j < width; j++)
                    {
                        ss << this->data[n * channel * height * width + c * height * width + i * width + j] << " ";
                    }
                    ss << std::endl;
                }
                ss << std::endl;
            }
            ss << std::endl;
        }

        return ss.str();
    }

} // namespace torchinfer