"""
VERIFICATION COMPLETE: Le fix s'est-il VRAIMENT appliqué?
On vérifie AVANT de builder
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
    print("\n✅ TOUTES VERIFICATIONS OK - PRET POUR BUILD")
else:
    print("\n❌ PROBLEMES DETECTES - NE PAS BUILDER ENCORE")
    print("   Lancer fix chirurgical d'abord!")

print(f"\nDecision: {'BUILD' if all_ok else 'FIX D\\'ABORD'}")
