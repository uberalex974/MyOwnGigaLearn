"""
DIAGNOSTIC CRITIQUE: Header PPOLearner.h a des erreurs de syntaxe
qui cassent TOUTE la compilation
"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== DIAGNOSTIC HEADER ===\n")

# Check brace balance
total_open = 0
total_close = 0
for i, line in enumerate(lines):
    total_open += line.count('{')
    total_close += line.count('}')

print(f"Total {{ : {total_open}")
print(f"Total }} : {total_close}")
print(f"Balance: {total_open - total_close}")

if total_open != total_close:
    print("❌ DESEQUILIBRE ACCOLADES!")

# Find problematic lines mentioned in error
print("\n=== LIGNES PROBLEMATIQUES ===")
problem_lines = [309, 310, 474, 475]
for line_num in problem_lines:
    if line_num <= len(lines):
        print(f"Line {line_num}: {lines[line_num-1].rstrip()}")

# Look for namespace closures
print("\n=== NAMESPACE STRUCTURE ===")
for i, line in enumerate(lines):
    if 'namespace' in line.lower() or ('}' in line and '//' in line and 'namespace' in line.lower()):
        print(f"Line {i+1}: {line.rstrip()}")

print("\n=== SOLUTION ===")
print("Probable: Nos optimizations ont cassé la structure du fichier")
print("Fix: Retirer optimizations du header, les mettre dans fichier séparé")
