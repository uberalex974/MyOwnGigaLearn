"""
PHASE 8: WIRING (INITIALIZATION)
Initialiser les variables membres dans PPOLearner.cpp
"""

init_code = '''
	// ==================================================
	// OPTIMIZATION INITIALIZATION (WIRED)
	// ==================================================
	
	// Phase 1-3
	tensor_cache_ = std::make_unique<GGL::Optimizations::TensorCache>();
	async_loader_ = std::make_unique<GGL::Optimizations::AsyncLoader>();
	pinned_memory_pool_ = std::make_unique<GGL::Optimizations::PinnedMemoryPool>();
	prioritized_sampler_ = std::make_unique<GGL::Optimizations::PrioritizedSampler>();
	mixture_of_experts_ = std::make_unique<GGL::Optimizations::MixtureOfExperts>(obsSize, 256, 4);
	adaptive_depth_net_ = std::make_unique<GGL::Optimizations::AdaptiveDepthNetwork>(obsSize, 256, 5);
	curiosity_module_ = std::make_unique<GGL::Optimizations::CuriosityModule>(obsSize, numActions);

	// Phase 4
	ema_weights_ = std::make_unique<GGL::Additional::ExponentialMovingAverage>();
	lr_warmup_ = std::make_unique<GGL::Additional::LRWarmupScheduler>(config.policyLR, 1000);
	adaptive_grad_clipper_ = std::make_unique<GGL::Additional::AdaptiveGradientClipper>();
	lora_adapter_ = std::make_unique<GGL::Additional::LoRAAdapter>(obsSize, 256, 8);
	dynamic_batch_sizer_ = std::make_unique<GGL::Additional::DynamicBatchSizer>(config.batchSize);

	// Phase 5
	multi_task_learner_ = std::make_unique<GGL::FinalWave::MultiTaskLearner>(256, std::vector<int>{numActions, 1});
	curriculum_scheduler_ = std::make_unique<GGL::FinalWave::CurriculumScheduler>();
	demo_bootstrapper_ = std::make_unique<GGL::FinalWave::DemonstrationBootstrapper>();

	// Phase 6
	parameter_noise_ = std::make_unique<GGL::HyperScale::ParameterNoise>();
	delayed_update_ctrl_ = std::make_unique<GGL::HyperScale::DelayedUpdateController>();

	// Phase 7
	rnd_module_ = std::make_unique<GGL::QuantumReady::RNDModule>(obsSize, 128);
	level_replay_selector_ = std::make_unique<GGL::QuantumReady::LevelReplaySelector>();
'''

cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'

with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'tensor_cache_ = std::make_unique' not in content:
    # Insert at the end of the constructor
    # Constructor is GGL::PPOLearner::PPOLearner(...) : ... { ... }
    # We look for the closing brace of the constructor.
    # It ends around line 37 in the view we saw earlier.
    
    # Let's find "guidingPolicyModels.Load(config.guidingPolicyPath, false, false);" which is near end of constructor
    marker = 'guidingPolicyModels.Load(config.guidingPolicyPath, false, false);'
    
    idx = content.find(marker)
    if idx != -1:
        # Find the next closing brace after this marker
        end_brace_idx = content.find('}', idx)
        if end_brace_idx != -1:
            new_content = content[:end_brace_idx] + init_code + content[end_brace_idx:]
            
            with open(cpp_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("✅ Wired initialization in PPOLearner.cpp")
        else:
            print("❌ Could not find constructor closing brace")
    else:
        # Fallback: look for "SetLearningRates(config.policyLR, config.criticLR);"
        marker2 = 'SetLearningRates(config.policyLR, config.criticLR);'
        idx2 = content.find(marker2)
        if idx2 != -1:
             # Find the next closing brace after this marker (this is safer if guiding policy is disabled in code but present in text)
             # Actually, let's just insert before the LAST closing brace of the constructor function block.
             # But parsing C++ with python regex is hard.
             # Let's try to append at the end of the constructor body.
             pass
        
        # Alternative: Insert after "SetLearningRates..." and before "}"
        # Let's assume the constructor ends with a "}" on a line by itself or at end of file logic.
        # Given the file view:
        # 37: }
        
        # We can try to replace the first occurrence of "}\n\nvoid GGL::PPOLearner::MakeModels" with our code + "}\n\nvoid..."
        
        split_marker = "void GGL::PPOLearner::MakeModels"
        parts = content.split(split_marker)
        if len(parts) > 1:
            # The constructor body is in parts[0]
            # Find the last '}' in parts[0]
            last_brace = parts[0].rfind('}')
            if last_brace != -1:
                parts[0] = parts[0][:last_brace] + init_code + parts[0][last_brace:]
                new_content = split_marker.join(parts)
                with open(cpp_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print("✅ Wired initialization in PPOLearner.cpp (method 2)")
            else:
                print("❌ Could not find constructor end in part 0")
        else:
            print("❌ Could not find MakeModels function definition")

else:
    print("✅ Initialization already wired")
