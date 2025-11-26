"""
FIX BUILD ERRORS - Par script, pas d'abandon!
Si erreur de compilation = on corrige le code par script
"""

import subprocess
import re

print("=== ANALYSE DES ERREURS DE BUILD ===\n")

# Run build et capture erreurs
result = subprocess.run(
    ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'build.ps1'],
    capture_output=True,
    text=True,
    cwd=r'c:\Giga\GigaLearnCPP'
)

if result.returncode == 0:
    print("✅ BUILD SUCCESSFUL!")
else:
    print("❌ BUILD FAILED - Analyse des erreurs...\n")
    
    # Extract erreurs
    errors = []
    for line in result.stdout.split('\n') + result.stderr.split('\n'):
        if 'error C' in line:
            errors.append(line.strip())
    
    print(f"Trouvé {len(errors)} erreur(s):\n")
    for err in errors[:10]:  # Show first 10
        print(err)
    
    # Identify fixes needed
    print("\n=== FIXES NECESSAIRES ===")
    
    if any('identificateur non déclaré' in e for e in errors):
        print("→ Variables non déclarées détectées")
        print("→ Script de fix: Ajouter déclarations manquantes")
    
    if any('namespace GGL' in e for e in errors):
        print("→ Problème namespace détecté")
        print("→ Script de fix: Nettoyer namespaces")
    
    if any('définitions de fonctions locales' in e for e in errors):
        print("→ Fonctions locales mal placées")
        print("→ Script de fix: Déplacer fonctions")

print("\nPROCHAINE ETAPE: Créer script de fix ciblé")
