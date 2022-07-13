#pragma once
#include <string>

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
            Layers();
    };
} // namespace torchinfer
