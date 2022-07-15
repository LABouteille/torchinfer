#pragma once

#include <string>
#include <fstream>
#include <memory>

#include "layers.hh"
#include "conv2d.hh"

namespace torchinfer
{
    class Model {
        public:
            Model();
            void load(const std::string &filename);
            void summary();

            std::vector<std::unique_ptr<Layers>> layers;
    };
    
} // namespace torchinfer