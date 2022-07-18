#include "inputs.hh"

namespace torchinfer
{
    Inputs::Inputs(std::string &name_arg, std::vector<int> &dims_input_arg)
        : name(name_arg),
          dims_input(dims_input_arg)
    {
    }

    std::string Inputs::info()
    {
        std::stringstream ss;
        ss << "Inputs: " << name << " [" << dims_input[0] << " " << dims_input[1] << " " << dims_input[2] << " " << dims_input[3] << "]";
        return ss.str();
    }
} // namespace torchinfer
