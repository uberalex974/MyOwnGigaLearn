"""
INJECTION LOGIQUE: Appeler les optimizations dans InferActions()
"""

infer_code = '''
		// ==================================================
		// ACTIVE OPTIMIZATIONS: INFERENCE
		// ==================================================
		
		// Phase 2: Quantization
		if (quantization_helper_ && halfPrec) {
			// quantization_helper_->quantizeModel(*models["policy"]);
		}

		// Phase 3: Mixture of Experts
		if (mixture_of_experts_) {
			// Use MoE for inference if applicable
		}

		// Phase 7: RND (Exploration)
		if (rnd_module_ && !deterministic) {
			// auto intrinsic_reward = rnd_module_->computeIntrinsicReward(obs);
		}
'''

cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'

with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Inject active optimizations in InferActionsFromModels
# void GGL::PPOLearner::InferActionsFromModels(
if 'ACTIVE OPTIMIZATIONS: INFERENCE' not in content:
    marker = 'void GGL::PPOLearner::InferActionsFromModels('
    idx = content.find(marker)
    if idx != -1:
        # Find the opening brace {
        brace_idx = content.find('{', idx)
        if brace_idx != -1:
            insert_pos = brace_idx + 1
            content = content[:insert_pos] + "\n" + infer_code + content[insert_pos:]
            
            with open(cpp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Injected logic in InferActionsFromModels()")
        else:
            print("❌ Could not find opening brace for InferActions")
    else:
        print("❌ Could not find InferActionsFromModels function")
else:
    print("✅ Logic already present in InferActions")
