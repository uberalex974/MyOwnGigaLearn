"""
Verification correcte - fix syntax error
"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

print("=== VERIFICATION PRE-BUILD ===\n")

checks = [
    ("MutAvgTracker avgPolicyLoss déclaré", "avgPolicyLoss," in content),
    ("Usage avec .Add() pas +=", "avgPolicyLoss.Add(" in content and "avgPolicyLoss +=" not in content),
    ("avgRelEntropyLoss déclaré", "avgRelEntropyLoss," in content),
    ("avgGuidingLoss déclaré", "avgGuidingLoss," in content),
]

all_ok = True
for desc, check in checks:
    status = "✅" if check else "❌"
    print(f"{status} {desc}")
    if not check:
        all_ok = False

if all_ok:
    print("\n✅ TOUTES VERIFICATIONS OK")
    print("MAIS l'erreur persiste - cherchons le VRAI probleme...")
    
    # Le VRAI problème: peut-être que le build system voit un ANCIEN fichier
    print("\nHYPOTHESE FINALE:")
    print("1. Code source = CORRECT")
    print("2. Erreur persiste = Build cache corrompu ou fichier .obj ancien")
    print("3. Solution = Clean COMPLET + verification fichier timestamp")
    
    import os
    cpp_file = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'
    if os.path.exists(cpp_file):
        mtime = os.path.getmtime(cpp_file)
        from datetime import datetime
        print(f"\nFichier PPOLearner.cpp modifié: {datetime.fromtimestamp(mtime)}")
        print("Si timestamp est ancien, le fichier n'a PAS été écrit!")
else:
    print("\n❌ PROBLEMES DETECTES")

