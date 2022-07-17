#pragma once

#include <string>
#include <vector>

#include "layers.hh"

namespace torchinfer
{
    class Inputs : public Layers
    {
    public:
        Inputs(std::string &name_arg, std::vector<int> &dims_input_arg);

        void forward();

        std::string name;
        std::vector<int> dims_input;
    };

} // namespace torchinfer
