"""
FIX: Le vrai problème est probablement dans Parallel GAE
L'erreur CUDA vient après Learn() - check GAE vectorization
"""

# The issue is likely in our Parallel GAE implementation
# Let's restore the original working GAE

print("Restoring ORIGINAL GAE implementation...")
print("Parallel GAE will be disabled but other optimizations kept!")

import shutil
import os

source = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\GAE.cpp.backup'
dest = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\GAE.cpp'

if os.path.exists(source):
    shutil.copy(source, dest)
    print(f"✅ Restored original GAE from backup")
else:
    print("❌ No backup found - GAE might be the issue")
    print("The Parallel GAE vectorization probably has a bug with empty tensors")

print("\n✅ KEEPING ALL OTHER OPTIMIZATIONS:")
print("  - Fused PPO Loss")
print("  - Gradient Accumulation")
print("  - Progressive Batching")
print("  - Data Filtering helpers (not used yet)")
print("  - Policy Filtration helpers (not used yet)")
print("\nOnly reverting Parallel GAE to fix CUDA error")
