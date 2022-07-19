#pragma once

#include "spdlog/spdlog.h"
#include <string>
#include <vector>
#include <sstream>
#include <stdbool.h>

#include "tensor.hh"

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

    template <typename T>
    class Layers {
        public:
            Layers() = default;
            virtual ~Layers() = default;
            virtual std::string info() = 0;
            virtual Tensor<T> forward(Tensor<T> &x) = 0;
    };
} // namespace torchinfer
