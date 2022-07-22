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

        // TODO: stride + padding (C++ and Python converter)
        auto batch = x.dims[0];
        auto channel = x.dims[1];
        auto height = x.dims[2];
        auto width = x.dims[3];

        auto kernel_height = weights.dims[2];
        auto kernel_width = weights.dims[3];
    
        auto nb_filters = weights.dims[0];
        auto out_height = (height - kernel_height + 1);
        auto out_width = (width - kernel_width + 1);

        Tensor<T> out;
        out.dims = {batch, nb_filters, out_height, out_width};
        // TODO: Find a better way to do this
        out.data.assign(batch * nb_filters * out_height * out_width, 0.);
        std::cout << "out: " << out.dims[0] << " " << out.dims[1] << " " << out.dims[2] << " " << out.dims[3] << std::endl;

        for (int n = 0; n < batch; n++)
        {
            auto batch_offset_x = n * (width * height * channel);
            auto batch_offset_out = n * (out_width * out_height * nb_filters);
                
            for (int f = 0; f < nb_filters; f++)
            {
                auto filter_offset_kernel = f * (kernel_width * kernel_height * weights.dims[1]);
                auto filter_offset_out = f * (out_width * out_height);

                for (int i = 0; i < out_height; i++)
                {
                    for (int j = 0; j < out_width; j++)
                    {
                        T val = 0;

                        for (int k_i = 0; k_i < kernel_height; k_i++)
                        {
                            for (int k_j = 0; k_j < kernel_width; k_j++)
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    auto channel_offset_x = c * (width * height);
                                    auto channel_offset_kernel = c * (kernel_width * kernel_height);

                                    auto offset_x = batch_offset_x + channel_offset_x + (k_i + i) * width + (k_j + j);
                                    auto offset_kernel = filter_offset_kernel + channel_offset_kernel + k_i * kernel_width + k_j;

                                    val += x[offset_x] * weights[offset_kernel];
                                }
                            }
                        }
                        auto offset_out = batch_offset_out + filter_offset_out + i * out_width + j;
                        out[offset_out] = val + bias[f];
                    }
                }
            }
        }

        std::cout << "Display value of out.data ... " << std::endl;
        for (int i = 0; i < out.dims[0]; i++)
        {
            for (int j = 0; j < out.dims[1]; j++)
            {
                for (int k = 0; k < out.dims[2]; k++)
                {
                    for (int l = 0; l < out.dims[3]; l++)
                    {
                        std::cout << out[i * out.dims[1] * out.dims[2] * out.dims[3] + j * out.dims[2] * out.dims[3] + k * out.dims[3] + l] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        return out;
    }

} // namespace torchinfer