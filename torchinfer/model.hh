#pragma once

#include <string>
#include <fstream>
#include <memory>

#include "io.hh"
#include "layers.hh"
#include "inputs.hh"
#include "conv2d.hh"

namespace torchinfer
{
    class Model {
        public:
            Model();
            void load(const std::string &filename_onnx_ir);
            void summary();

            std::vector<std::unique_ptr<Layers>> layers;
    };
    
} // namespace torchinfer