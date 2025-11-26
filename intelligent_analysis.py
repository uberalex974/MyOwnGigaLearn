"""
ANALYSE INTELLIGENTE: Comprendre le VRAI problème avant de fix
Pas de build prématuré - vérification COMPLETE d'abord
"""

print("=== ANALYSE METHODIQUE DU PROBLEME ===\n")

# 1. LIRE le fichier COMPLETEMENT
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("1. STRUCTURE DU FICHIER")
print(f"   Total lignes: {len(lines)}")

# 2. TROUVER la fonction Learn
learn_start = None
for i, line in enumerate(lines):
    if 'void GGL::PPOLearner::Learn(' in line:
        learn_start = i + 1  # 1-indexed
        print(f"   Fonction Learn() commence: ligne {learn_start}")
        break

# 3. VERIFIER les declarations MutAvgTracker
print("\n2. DECLARATIONS MutAvgTracker (lignes 145-156):")
for i in range(144, 156):
    if i < len(lines):
        print(f"   {i+1}: {lines[i].rstrip()}")

# 4. VERIFIER les USAGES
print("\n3. USAGES de avgPolicyLoss:")
for i, line in enumerate(lines):
    if 'avgPolicyLoss' in line and i+1 >= learn_start:
        print(f"   Ligne {i+1}: {line.rstrip()}")

# 5. IDENTIFIER LE PROBLEME
print("\n4. DIAGNOSTIC:")
print("   Si MutAvgTracker est déclaré DANS Learn(),")
print("   et usage est AUSSI dans Learn(),")
print("   alors c'est dans la même portée - devrait marcher!")

print("\n5. HYPOTHESE:")
print("   Peut-être qu'une ACCOLADE MANQUANTE ou MAL PLACÉE")
print("   a cassé la portée (scope)?")

# 6. VERIFIER les accolades
brace_count = 0
for i in range(144, min(250, len(lines))):
    line = lines[i]
    brace_count += line.count('{') - line.count('}')
    if brace_count < 0:
        print(f"\n❌ PROBLEME TROUVE! Ligne {i+1}: Accolade fermante en trop!")
        print(f"   {line.rstrip()}")
        break

print(f"\n   Balance accolades après déclarations: {brace_count}")
print("   (devrait être > 0 car on est dans la fonction)")

print("\n=== FIN ANALYSE - PRET POUR FIX CHIRURGICAL ===")
