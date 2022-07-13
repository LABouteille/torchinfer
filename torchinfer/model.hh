#pragma once

#include <string>
#include <fstream>

#include "layers.hh"
#include "conv2d.hh"

namespace torchinfer
{
    class Model {
        public:
            Model();
            void setup(std::string filename);
            // void summary();

            std::vector<Layers> layers;
    };
    
} // namespace torchinfer