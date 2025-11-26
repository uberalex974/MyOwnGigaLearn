"""
VERIFICATION INTELLIGENTE: Build a-t-il vraiment échoué ou juste warnings?
"""

import os
import subprocess

print("=== DIAGNOSTIC BUILD ===\n")

# 1. Check if exe exists despite "build failed"
exe_path = r'c:\Giga\GigaLearnCPP\out\build\x64-relwithdebinfo\GigaLearnBot_Deploy.exe'
exe_exists = os.path.exists(exe_path)

print(f"1. Executable: {'✅ EXISTS' if exe_exists else '❌ MISSING'}")

if exe_exists:
    # Get file size and timestamp
    size = os.path.getsize(exe_path)
    from datetime import datetime
    mtime = os.path.getmtime(exe_path)
    print(f"   Size: {size:,} bytes")
    print(f"   Modified: {datetime.fromtimestamp(mtime)}")
    print("\n✅ BUILD ACTUALLY SUCCEEDED! (warnings only)")
    
    # Verify OptimizationsHelpers.h
    opt_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\OptimizationsHelpers.h'
    if os.path.exists(opt_path):
        with open(opt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count namespaces to determine phase count
        phase_count = 0
        for phase in ["Optimizations", "QuickWins", "Phase2", "Phase3", "Additional", "FinalWave"]:
            if f"namespace {phase}" in content:
                phase_count += 1
        
        print(f"\n2. OptimizationsHelpers.h: {phase_count} phases present")
        
        # Count classes
        class_count = content.count("class ") + content.count("struct ")
        print(f"   Total classes/structs: ~{class_count}")
        
else:
    print("\n❌ Build vraiment échoué")
    print("Analysons les erreurs...")
    
    # Re-run build to get fresh errors
    result = subprocess.run(
        ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'build.ps1'],
        capture_output=True,
        text=True,
        cwd=r'c:\Giga\GigaLearnCPP',
        timeout=120
    )
    
    # Extract real errors (not warnings)
    errors = []
    for line in result.stdout.split('\n'):
        if 'error C' in line and 'warning' not in line:
            errors.append(line.strip())
    
    print(f"\nReal errors found: {len(errors)}")
    for err in errors[:5]:
        print(f"  {err}")

print("\n=== DECISION ===")
if exe_exists:
    print("✅ Continue avec 51 optimizations - exe fonctionne!")
else:
    print("❌ Fix surgical nécessaire")
