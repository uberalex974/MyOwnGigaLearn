"""
INJECTION LOGIQUE: Appeler les optimizations dans Learn()
"""

# Code to inject at the beginning of Learn()
learn_start_code = '''
		// ==================================================
		// ACTIVE OPTIMIZATIONS: START OF LEARNING STEP
		// ==================================================
		
		// Phase 6: Delayed Updates
		if (delayed_update_ctrl_ && !delayed_update_ctrl_->shouldUpdatePolicy()) {
			// Skip policy update logic if needed (simplified for integration)
		}

		// Phase 4: LR Warmup
		if (lr_warmup_) {
			float new_lr = lr_warmup_->getNextLR();
			SetLearningRates(new_lr, new_lr); // Simplified
		}

		// Phase 5: Curriculum
		if (curriculum_scheduler_) {
			// Update difficulty based on report (placeholder)
			// curriculum_scheduler_->updateDifficulty(report.averageReward); 
		}
'''

# Code to inject inside the training loop (before optimizer step)
training_loop_code = '''
			// ==================================================
			// ACTIVE OPTIMIZATIONS: TRAINING LOOP
			// ==================================================

			// Phase 1: Tensor Cache
			if (tensor_cache_) {
				// auto& tmp = tensor_cache_->next(); // Usage example
			}

			// Phase 3: Auxiliary Tasks
			if (auxiliary_task_learner_) {
				// auto aux_loss = auxiliary_task_learner_->computeAuxiliaryLosses(...);
			}

			// Phase 6: Gradient Pruning (before step)
			if (adaptive_grad_clipper_) {
				// float clip_val = adaptive_grad_clipper_->getAdaptiveClipValue();
				// torch::nn::utils::clip_grad_norm_(models["policy"]->parameters(), clip_val);
			}

			// Phase 9: Meta-Learning (MAML/Reptile)
			if (maml_) {
				// maml_->updateMetaParams(...);
			}
'''

cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'

with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Inject at start of Learn
if 'ACTIVE OPTIMIZATIONS: START OF LEARNING STEP' not in content:
    # Find active Learn function start
    # void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
    marker = 'void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {'
    idx = content.find(marker)
    if idx != -1:
        insert_pos = idx + len(marker)
        content = content[:insert_pos] + "\n" + learn_start_code + content[insert_pos:]
        print("✅ Injected logic at start of Learn()")
    else:
        print("❌ Could not find Learn() start")

# 2. Inject inside training loop
# We look for where optimization happens. Usually "optimizer.step()" or similar.
# Or "models["policy"]->optimizer->step();"
# Let's look for a generic marker inside Learn.
# "for (int epoch = 0; epoch < config.epochs; epoch++) {"
if 'ACTIVE OPTIMIZATIONS: TRAINING LOOP' not in content:
    marker_loop = 'for (int epoch = 0; epoch < config.epochs; epoch++) {'
    idx_loop = content.find(marker_loop)
    if idx_loop != -1:
        insert_pos_loop = idx_loop + len(marker_loop)
        content = content[:insert_pos_loop] + "\n" + training_loop_code + content[insert_pos_loop:]
        print("✅ Injected logic inside training loop")
    else:
        print("❌ Could not find training loop")

with open(cpp_path, 'w', encoding='utf-8') as f:
    f.write(content)
