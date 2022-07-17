#include "inputs.hh"

namespace torchinfer
{
    Inputs::Inputs(std::string &name_arg, std::vector<int> &dims_input_arg)
        : Layers(name_arg, dims_input_arg, true),
          name(name_arg),
          dims_input(dims_input_arg)
    {
    }

} // namespace torchinfer
