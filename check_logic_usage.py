"""
VERIFICATION: Est-ce que les membres sont APPELÉS dans la logique?
"""

import os

cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'
with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

members_to_check = [
    "tensor_cache_", "async_loader_", "pinned_memory_pool_", "prioritized_sampler_",
    "mixture_of_experts_", "adaptive_depth_net_", "curiosity_module_",
    "ema_weights_", "lr_warmup_", "adaptive_grad_clipper_", "lora_adapter_", "dynamic_batch_sizer_",
    "multi_task_learner_", "curriculum_scheduler_", "demo_bootstrapper_",
    "parameter_noise_", "delayed_update_ctrl_",
    "rnd_module_", "level_replay_selector_",
    "maml_", "reptile_", "task_embedder_"
]

used_in_logic = []
for m in members_to_check:
    # Check for usage other than initialization (std::make_unique)
    # We look for "m->" or "m." usage
    if f"{m}->" in content or f"{m}." in content:
        used_in_logic.append(m)

print(f"Used in Logic: {len(used_in_logic)}/{len(members_to_check)}")
if len(used_in_logic) < len(members_to_check):
    print("⚠️  Optimizations are Initialized but NOT CALLED in logic loops!")
    print("   They are 'Ready' but 'Idle'.")
    print("   Action: Inject calls into Learn() and InferActions()")
else:
    print("✅ Optimizations are actively called!")
