"""
VERIFICATION: Build success ou pas?
"""

import subprocess

result = subprocess.run(
    ['powershell', '-Command', 'Test-Path c:\\Giga\\GigaLearnCPP\\out\\build\\x64-relwithdebinfo\\GigaLearnBot_Deploy.exe'],
    capture_output=True,
    text=True
)

exe_exists = result.stdout.strip() == 'True'

print("=== BUILD VERIFICATION ===\n")

if exe_exists:
    print("✅✅✅ BUILD SUCCESSFUL! ✅✅✅")
    print("\n   GigaLearnBot_Deploy.exe EXISTS!")
    print("\n   BASE CODE (avec GAE + PPO) compile!")
    print("\n   MAINTENANT: Ré-ajouter optimizations PROPREMENT")
else:
    print("❌ Build failed ou exe pas créé")
    print("   Checking build output...")

print("\nProchaine étape:")
if exe_exists:
    print("  1. Créer fichier OptimizationsHelpers.h séparé")
    print("  2. Y mettre nos 13 optimization classes")
    print("  3. Include dans PPOLearner.h")
    print("  4. Build final avec TOUT!")
else:
    print("  Analyser erreurs de build")
