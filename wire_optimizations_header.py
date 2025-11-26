"""
PHASE 8: WIRING (CÂBLAGE)
Ajouter les variables membres dans PPOLearner.h
"""

members_code = '''
		// ==================================================
		// OPTIMIZATION MEMBERS (WIRED)
		// ==================================================
		
		// Phase 1-3: Extreme
		std::unique_ptr<GGL::Optimizations::TensorCache> tensor_cache_;
		std::unique_ptr<GGL::Optimizations::AsyncLoader> async_loader_;
		std::unique_ptr<GGL::Optimizations::PinnedMemoryPool> pinned_memory_pool_;
		std::unique_ptr<GGL::Optimizations::PrioritizedSampler> prioritized_sampler_;
		std::unique_ptr<GGL::Optimizations::MixtureOfExperts> mixture_of_experts_;
		std::unique_ptr<GGL::Optimizations::AdaptiveDepthNetwork> adaptive_depth_net_;
		std::unique_ptr<GGL::Optimizations::CuriosityModule> curiosity_module_;

		// Phase 4: Additional
		std::unique_ptr<GGL::Additional::ExponentialMovingAverage> ema_weights_;
		std::unique_ptr<GGL::Additional::LRWarmupScheduler> lr_warmup_;
		std::unique_ptr<GGL::Additional::AdaptiveGradientClipper> adaptive_grad_clipper_;
		std::unique_ptr<GGL::Additional::LoRAAdapter> lora_adapter_;
		std::unique_ptr<GGL::Additional::DynamicBatchSizer> dynamic_batch_sizer_;

		// Phase 5: Final Wave
		std::unique_ptr<GGL::FinalWave::MultiTaskLearner> multi_task_learner_;
		std::unique_ptr<GGL::FinalWave::CurriculumScheduler> curriculum_scheduler_;
		std::unique_ptr<GGL::FinalWave::DemonstrationBootstrapper> demo_bootstrapper_;

		// Phase 6: Hyper-Scale
		std::unique_ptr<GGL::HyperScale::ParameterNoise> parameter_noise_;
		std::unique_ptr<GGL::HyperScale::DelayedUpdateController> delayed_update_ctrl_;

		// Phase 7: Quantum-Ready
		std::unique_ptr<GGL::QuantumReady::RNDModule> rnd_module_;
		std::unique_ptr<GGL::QuantumReady::LevelReplaySelector> level_replay_selector_;
'''

header_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h'

with open(header_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'std::unique_ptr<GGL::Optimizations::TensorCache> tensor_cache_;' not in content:
    # Insert before the last closing brace of the class
    # The class ends with "};" then "} // namespace GGL"
    
    # Find the last "};"
    last_brace_idx = content.rfind('};')
    
    if last_brace_idx != -1:
        new_content = content[:last_brace_idx] + members_code + content[last_brace_idx:]
        
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Wired member variables in PPOLearner.h")
    else:
        print("❌ Could not find class closing brace")
else:
    print("✅ Members already wired")
