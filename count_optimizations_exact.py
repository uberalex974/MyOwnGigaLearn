"""
FINAL VERIFICATION: Compte EXACTEMENT combien d'optimizations sont dans le code
"""

import os

opt_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\OptimizationsHelpers.h'

if os.path.exists(opt_path):
    with open(opt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("=== OPTIMIZATIONS DANS OptimizationsHelpers.h ===\n")
    
    # List ALL optimization classes
    optimizations = [
        # Phase 1-3 (Extreme)
        ("TensorCache", "Optimizations"),
        ("AsyncLoader", "Optimizations"),
        ("OptimizerBatch", "Optimizations"),
        ("PinnedMemoryPool", "QuickWins"),
        ("PrioritizedSampler", "QuickWins"),
        ("QuantizationHelper", "Phase2"),
        ("MultiStepReturns", "Phase2"),
        ("AttentionOptimizer", "Phase2"),
        ("CheckpointHelper", "Phase2"),
        ("MixtureOfExperts", "Phase3"),
        ("AuxiliaryTaskLearner", "Phase3"),
        ("AdaptiveDepthNetwork", "Phase3"),
        ("CuriosityModule", "Phase3"),
        # Phase 4 (Additional)
        ("ResidualNetworkBuilder", "Additional"),
        ("ExponentialMovingAverage", "Additional"),
        ("LRWarmupScheduler", "Additional"),
        ("AdaptiveGradientClipper", "Additional"),
        ("SparseTrainingHelper", "Additional"),
        ("LoRAAdapter", "Additional"),
        ("DynamicBatchSizer", "Additional"),
        # Phase 5 (Final Wave)
        ("MultiTaskLearner", "FinalWave"),
        ("CurriculumScheduler", "FinalWave"),
        ("DemonstrationBootstrapper", "FinalWave"),
        ("LayerFusionHelper", "FinalWave"),
    ]
    
    found = 0
    missing = []
    
    for opt_name, phase in optimizations:
        # Check both class and struct
        if f"class {opt_name}" in content or f"struct {opt_name}" in content:
            found += 1
            print(f"✅ {opt_name} ({phase})")
        else:
            missing.append((opt_name, phase))
            print(f"❌ {opt_name} ({phase}) - MISSING")
    
    print(f"\n=== TOTAL ===")
    print(f"Found: {found}/{len(optimizations)}")
    print(f"Base code: 27")
    print(f"TOTAL: {27 + found}/51")
    
    if missing:
        print(f"\nMissing: {len(missing)}")
        for opt, phase in missing:
            print(f"  - {opt} ({phase})")
else:
    print("❌ OptimizationsHelpers.h not found!")
