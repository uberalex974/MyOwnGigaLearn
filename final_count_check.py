"""
FINAL COUNT VERIFICATION
"""

import os

opt_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\OptimizationsHelpers.h'

if os.path.exists(opt_path):
    with open(opt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    phases = [
        ("Optimizations", "Phase 1"),
        ("QuickWins", "Phase 1"),
        ("Phase2", "Phase 2"),
        ("Phase3", "Phase 3"),
        ("Additional", "Phase 4"),
        ("FinalWave", "Phase 5"),
        ("HyperScale", "Phase 6"),
        ("QuantumReady", "Phase 7")
    ]
    
    print("=== PHASES FOUND ===\n")
    for ns, name in phases:
        if f"namespace {ns}" in content:
            print(f"✅ {name} ({ns})")
        else:
            print(f"❌ {name} ({ns}) - MISSING")
            
    # Count classes/structs
    count = content.count("class ") + content.count("struct ")
    print(f"\nTotal Helper Classes: ~{count}")
    print(f"Base Optimizations: 27")
    print(f"Total Estimated: {27 + count}")
    
else:
    print("❌ File not found")
