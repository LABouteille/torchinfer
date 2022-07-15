#include "layers.hh"

namespace torchinfer
{
    Layers::Layers(std::string &name_arg, std::vector<int> &dims_weights_arg, std::vector<int> &dims_bias_arg)
        : name(name_arg),
          dims_weights(dims_weights_arg),
          dims_bias(dims_bias_arg)
    {
    }
    
    Layers::Layers(std::string &name_arg, std::vector<int> &dims_weights_arg)
        : name(name_arg),
          dims_weights(dims_weights_arg)
    {
    }

    std::string Layers::info()
    {
        std::stringstream ss;
        
        ss << "\t" << this->name << std::endl;
        ss << "\t\tweights: ";

        for (auto dim : dims_weights)
            ss << dim << " ";
        
        ss << std::endl;

        if (!dims_bias.empty())
        {
            ss << "\t\tbias: ";
            for (auto dim : dims_bias)
                ss << dim << " ";
            ss << std::endl;
        }
        return ss.str();
    }
} // namespace torchinfer
