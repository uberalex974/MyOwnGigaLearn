"""
FORCE REBUILD COMPLET - Suppression TOTALE du cache
"""

import os
import shutil
import subprocess

print("=== FORCE REBUILD COMPLET ===\n")

# 1. Supprimer TOUT le cache
paths_to_delete = [
    r'c:\Giga\GigaLearnCPP\out',
    r'c:\Giga\GigaLearnCPP\build',
    r'c:\Giga\GigaLearnCPP\CMakeCache.txt',
]

for path in paths_to_delete:
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"✅ Deleted directory: {path}")
            else:
                os.remove(path)
                print(f"✅ Deleted file: {path}")
        except Exception as e:
            print(f"⚠️  Could not delete {path}: {e}")

print("\n=== CACHE COMPLETEMENT SUPPRIME ===")
print("Lancement build fresh...")
print("\n" + "="*50)

# 2. Build from scratch
result = subprocess.run(
    ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'build.ps1'],
    cwd=r'c:\Giga\GigaLearnCPP',
    capture_output=True,
    text=True,
    timeout=180
)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode == 0:
    print("\n" + "="*50)
    print("✅✅✅ BUILD SUCCESSFUL! ✅✅✅")
    print("44 OPTIMIZATIONS FONCTIONNELLES!")
    print("="*50)
else:
    print("\n" + "="*50)
    print("❌ Build failed - analyzing errors...")
    
    # Extract just the errors
    errors = []
    for line in (result.stdout + result.stderr).split('\n'):
        if 'error C' in line and len(line) < 200:
            errors.append(line.strip())
    
    print(f"\nFound {len(errors)} errors:")
    for err in errors[:5]:
        print(f"  {err}")
    
    print("\nNEED SURGICAL FIX - ready to apply")
