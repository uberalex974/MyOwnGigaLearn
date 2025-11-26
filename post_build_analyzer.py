"""
POST-BUILD ANALYZER - Analyze errors and apply surgical fixes
"""

import subprocess
import sys

sys.path.append(r'c:\Giga\GigaLearnCPP')
from surgical_fix_toolkit import ChirurgicalFixer

fixer = ChirurgicalFixer()

# Wait for build to complete, then analyze
print("=== WAITING FOR BUILD COMPLETION ===\n")

# Verify optimizations still present
header_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h'
if fixer.verify_optimizations_present(header_path):
    print("\n✅ ALL 14 OPTIMIZATIONS VERIFIED IN HEADER!")
else:
    print("\n❌ Some optimizations missing - will re-add")

print("\n=== READY FOR POST-BUILD ANALYSIS ===")
print("Once build completes, will analyze and fix any errors")
