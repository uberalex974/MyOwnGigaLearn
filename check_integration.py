"""
VERIFICATION: Est-ce que les optimizations sont INSTANCIÉES et UTILISÉES?
"""

import os

print("=== INTEGRATION CHECK ===\n")

# 1. Check PPOLearner.h for member instantiations
header_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h'
with open(header_path, 'r', encoding='utf-8') as f:
    header_content = f.read()

# List of helper classes we created
helpers = [
    "TensorCache", "AsyncLoader", "OptimizerBatch", "PinnedMemoryPool", 
    "PrioritizedSampler", "QuantizationHelper", "MultiStepReturns",
    "AttentionOptimizer", "CheckpointHelper", "MixtureOfExperts",
    "AuxiliaryTaskLearner", "AdaptiveDepthNetwork", "CuriosityModule",
    "ExperienceCompressor", "ParameterNoise", "DelayedUpdateController",
    "GradientPruner", "HindsightReplay", "RNDModule", "NoisyLinear",
    "DistributionalHelper", "LevelReplaySelector"
]

instantiated = []
for h in helpers:
    if f"{h} " in header_content or f"{h}*" in header_content or f"std::unique_ptr<{h}>" in header_content:
        instantiated.append(h)

print(f"Instantiated in Header: {len(instantiated)}/{len(helpers)}")
for h in instantiated:
    print(f"  ✅ {h}")

missing = [h for h in helpers if h not in instantiated]
if missing:
    print(f"\nNot Instantiated (Helpers available but not wired): {len(missing)}")
    # for h in missing:
    #     print(f"  ❌ {h}")

# 2. Check PPOLearner.cpp for usage
cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'
with open(cpp_path, 'r', encoding='utf-8') as f:
    cpp_content = f.read()

used = []
for h in helpers:
    if h in cpp_content:
        used.append(h)

print(f"\nReferenced in CPP: {len(used)}/{len(helpers)}")

print("\n=== CONCLUSION ===")
if len(instantiated) == 0:
    print("⚠️  Optimizations are DEFINED in helpers but NOT INSTANTIATED in PPOLearner.")
    print("   They are 'Real' code, but currently 'Sleeping'.")
    print("   To make them 'Active', we need to add member variables.")
else:
    print("✅ Some optimizations are active.")

print("\nAction required: Wire them up to make them truly 'work'.")
