#include "layers.hh"

namespace torchinfer
{
    Layers::Layers(std::string &name_arg, std::vector<int> &dims_weights_arg, std::vector<int> &dims_bias_arg)
        : name(name_arg),
          dims_weights(dims_weights_arg),
          dims_bias(dims_bias_arg)
    {
    }
    
    Layers::Layers(std::string &name_arg, std::vector<int> &dims_arg, bool is_input_arg)
    {
        this->name = name_arg;
        this->is_input = is_input_arg;
        if (this->is_input)
            this->dims_input = dims_arg;
        else
            this->dims_weights = dims_arg;
    }

    std::string Layers::info()
    {
        std::stringstream ss;
        
        ss << "\n" << this->name << std::endl;
        ss << "\tweights: ";

        for (auto dim : dims_weights)
            ss << dim << " ";
        
        ss << std::endl;

        if (!dims_bias.empty())
        {
            ss << "\tbias: ";
            for (auto dim : dims_bias)
                ss << dim << " ";
            ss << std::endl;
        }
        return ss.str();
    }
} // namespace torchinfer
