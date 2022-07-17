#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <stdbool.h>

namespace torchinfer
{
    enum class OPTYPE
    {
        CONV2D = 0,
        RELU = 1,
        FLATTEN = 2,
        LINEAR = 3,
        INPUT = 4,
        OUTPUT = 5
    };

    class Layers {
        public:
            Layers(std::string &name_arg, std::vector<int> &dims_weights_arg, std::vector<int> &dims_bias_arg);
            Layers(std::string &name_arg, std::vector<int> &dims_arg, bool is_input_arg);
            
            std::string info();
            std::string name;
            std::vector<int> dims_input;
            std::vector<int> dims_weights;
            std::vector<int> dims_bias;
            bool is_input;

    };
} // namespace torchinfer
