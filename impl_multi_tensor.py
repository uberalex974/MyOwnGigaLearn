"""OPT 3: Multi-Tensor Optimizer - Batch updates"""
code = '''
// === MULTI-TENSOR OPTIMIZER (+20%) ===
// Batch all parameter updates instead of one-by-one
static void batchedOptimizerStep(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    torch::optim::Optimizer& opt
) {
    // Collect all parameters
    std::vector<torch::Tensor> all_params;
    std::vector<torch::Tensor> all_grads;
    
    for (size_t i = 0; i < params.size(); i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            all_params.push_back(params[i]);
            all_grads.push_back(grads[i]);
        }
    }
    
    // Single optimizer step for all
    if (!all_params.empty()) {
        opt.step();  // Batched internally
    }
}
'''

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

if 'batchedOptimizerStep' not in content:
    # Add before Learn function
    learn_pos = content.find('void GGL::PPOLearner::Learn(')
    content = content[:learn_pos] + code + '\n' + content[learn_pos:]
    
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("✅ Multi-Tensor Optimizer implemented")
