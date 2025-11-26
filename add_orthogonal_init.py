import re

# Read the file
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\Util\Models.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Add the includes for init functions at the top
old_includes = """#include "Models.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>"""

new_includes = """#include "Models.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/init.h>"""

content = content.replace(old_includes, new_includes)

# Add orthogonal initialization after optimizer creation
old_constructor_end = """\tregister_module("seq", seq);
\tseq->to(device);
\toptim = MakeOptimizer(config.optimType, this->parameters(), 0);
}"""

new_constructor_end = """\tregister_module("seq", seq);
\tseq->to(device);
\toptim = MakeOptimizer(config.optimType, this->parameters(), 0);

\t// === ORTHOGONAL INITIALIZATION (OPTIMAL FOR RL) ===
\t// Better gradient flow and faster convergence
\t{
\t\ttorch::NoGradGuard no_grad;
\t\tfor (auto& p : this->parameters()) {
\t\t\tif (p.dim() >= 2) {
\t\t\t\t// Weight matrix - use orthogonal init
\t\t\t\tfloat gain = std::sqrt(2.0f);  // Default for LeakyReLU
\t\t\t\t
\t\t\t\t// Special gains for output layers
\t\t\t\tif (config.addOutputLayer && p.size(0) == config.numOutputs) {
\t\t\t\t\tif (std::string(modelName) == "critic") {
\t\t\t\t\t\tgain = 1.0f;  // Value function
\t\t\t\t\t} else if (std::string(modelName) == "policy") {
\t\t\t\t\t\tgain = 0.01f;  // Policy (promotes exploration)
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\t
\t\t\t\ttorch::nn::init::orthogonal_(p, gain);
\t\t\t} else {
\t\t\t\t// Bias vector - zero init
\t\t\t\ttorch::nn::init::constant_(p, 0.0f);
\t\t\t}
\t\t}
\t}
}"""

content = content.replace(old_constructor_end, new_constructor_end)

# Write back
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\Util\Models.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("Orthogonal Initialization successfully implemented!")
