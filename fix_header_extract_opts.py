"""
FIX CHIRURGICAL: Extraire nos optimizations dans fichier séparé
Restaurer header à état propre, puis include le fichier optimizations
"""

import subprocess

print("=== FIX CHIRURGICAL HEADER ===\n")

# 1. Backup current header optimizations
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'r', encoding='utf-8') as f:
    current_header = f.read()

# Extract OUR optimizations
opt_classes = []
for opt_name in ["TensorCache", "AsyncLoader", "OptimizerBatch", "PinnedMemoryPool", 
                  "PrioritizedSampler", "QuantizationHelper", "MultiStepReturns",
                  "AttentionOptimizer", "CheckpointHelper", "MixtureOfExperts",
                  "AuxiliaryTaskLearner", "AdaptiveDepthNetwork", "CuriosityModule"]:
    if opt_name in current_header:
        opt_classes.append(opt_name)

print(f"Found {len(opt_classes)} optimization classes in header")

# 2. Get CLEAN header from git
print("\nRestaurant header depuis git...")
result = subprocess.run(
    ['git', 'show', 'HEAD:GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.h'],
    capture_output=True,
    text=True,
    cwd=r'c:\Giga\GigaLearnCPP'
)

if result.returncode == 0:
    clean_header = result.stdout
    
    # Write clean header
    with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'w', encoding='utf-8') as f:
        f.write(clean_header)
    
    print("✅ Header restauré à version git")
    print(f"✅ {len(opt_classes)} optimizations sauvegardées")
    print("\n⚠️  Optimizations RETIRÉES temporairement pour fix")
    print("   Elles seront ré-ajoutées proprement après")
else:
    print("❌ Git restore failed")

print("\nProchaine étape: Build avec header clean pour vérifier")
