"""
FINAL COUNT VERIFICATION (REAL ONLY)
"""

import os

opt_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\OptimizationsHelpers.h'

if os.path.exists(opt_path):
    with open(opt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for removed items
    removed = ["DNA_Architect", "HolographicMemory", "FractalScheduler"]
    for r in removed:
        if r in content:
            print(f"❌ {r} STILL PRESENT (Should be removed)")
        else:
            print(f"✅ {r} Removed")
            
    # Check for renamed/kept items
    if "SimulatedAnnealer" in content:
        print("✅ SimulatedAnnealer Present")
    else:
        print("❌ SimulatedAnnealer MISSING")
        
    if "PSO_Tuner" in content:
        print("✅ PSO_Tuner Present")
    else:
        print("❌ PSO_Tuner MISSING")

    # Count classes/structs
    count = content.count("class ") + content.count("struct ")
    print(f"\nTotal Helper Classes: ~{count}")
    print(f"Base Optimizations: 27")
    print(f"Total Estimated: {27 + count}")
    
else:
    print("❌ File not found")
