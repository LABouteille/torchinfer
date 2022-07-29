#pragma once

#include "layers.hh"
#include <vector>
#include <iostream>

namespace torchinfer
{
    template <typename T>
    class Conv2D : public Layers<T>
    {
    public:
        Conv2D(std::string name_arg, Tensor<T> weights_arg, Tensor<T> bias_arg, std::vector<int> strides_arg);
        Conv2D(std::string name_arg, Tensor<T> weights_arg, std::vector<int> strides_arg);

        std::string info() override;
        Tensor<T> forward(Tensor<T> &x) override;

        std::string name;
        Tensor<T> weights;
        Tensor<T> bias;
        std::vector<int> strides;
        Tensor<T> out;
    };

    template <typename T>
    Conv2D<T>::Conv2D(std::string name_arg, Tensor<T> weights_arg, Tensor<T> bias_arg, std::vector<int> strides_arg)
        : name(name_arg),
          weights(weights_arg),
          bias(bias_arg),
          strides(strides_arg)
    {
    }

    template <typename T>
    Conv2D<T>::Conv2D(std::string name_arg, Tensor<T> weights_arg, std::vector<int> strides_arg)
        : name(name_arg),
          weights(weights_arg),
          strides(strides_arg)
    {
        this->bias.dims = {this->weights.dims[0]};
        this->bias.data.assign(this->weights.dims[0], (T)0);
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
        // TODO: stride + padding (C++ and Python converter)
        unsigned int batch = x.dims[0];
        unsigned int channel = x.dims[1];
        unsigned int height = x.dims[2];
        unsigned int width = x.dims[3];

        unsigned int kernel_height = weights.dims[2];
        unsigned int kernel_width = weights.dims[3];

        //TODO: Implement padding
        unsigned int nb_filters = weights.dims[0];
        unsigned int out_height = std::floor(((height - kernel_height) / strides[0]) + 1);
        unsigned int out_width = std::floor(((width - kernel_width) / strides[1]) + 1);

        for (unsigned int n = 0; n < batch; n++)
        {
            auto batch_offset_x = n * (width * height * channel);
            auto batch_offset_out = n * (out_width * out_height * nb_filters);

            for (unsigned int f = 0; f < nb_filters; f++)
            {
                auto filter_offset_kernel = f * (kernel_width * kernel_height * weights.dims[1]);
                auto filter_offset_out = f * (out_width * out_height);

                for (unsigned int i = 0; i < out_height; i+=1)
                {
                    for (unsigned int j = 0; j < out_width; j+=1)
                    {
                        T val = 0;

                        for (unsigned int k_i = 0; k_i < kernel_height; k_i++)
                        {
                            for (unsigned int k_j = 0; k_j < kernel_width; k_j++)
                            {
                                for (unsigned int c = 0; c < channel; c++)
                                {
                                    auto channel_offset_x = c * (width * height);
                                    auto channel_offset_kernel = c * (kernel_width * kernel_height);

                                    val += x[batch_offset_x + channel_offset_x + (k_i + i * strides[0]) * width + (k_j + j * strides[1])] * \
                                           weights[filter_offset_kernel + channel_offset_kernel + k_i * kernel_width + k_j];
                                }
                            }
                        }
                        this->out[batch_offset_out + filter_offset_out + i * out_width + j] = val + bias[f];
                    }
                }
            }
        }

        return this->out;
    }

} // namespace torchinfer