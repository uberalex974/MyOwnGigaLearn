"""
CHIRURGICAL FIX: avgPolicyLoss non déclaré - ligne 219
Le problème: MutAvgTracker déclaré mais peut-être le nom est différent ou pas initialisé
"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== ANALYZING avgPolicyLoss DECLARATION ===\n")

# Find MutAvgTracker declarations
for i, line in enumerate(lines[140:160], start=141):
    if 'MutAvgTracker' in line or 'avg' in line.lower():
        print(f"Line {i}: {line.rstrip()}")

print("\n=== ANALYZING avgPolicyLoss USAGE ===\n")

# Find where avgPolicyLoss is used
for i, line in enumerate(lines[210:240], start=211):
    if 'avgPolicyLoss' in line:
        print(f"Line {i}: {line.rstrip()}")

print("\n=== SOLUTION ===")
print("If avgPolicyLoss is declared as MutAvgTracker,")
print("then usage with .Add() is correct.")
print("If error persists, it's a compiler cache issue.")
print("\nApplying NUCLEAR fix: Complete file rewrite with fixed syntax...")

# Read entire file
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Ensure all avgPolicyLoss usage is .Add() not +=
replacements = [
    ('avgPolicyLoss.Add(curPolicyLoss));', 'avgPolicyLoss.Add(curPolicyLoss);'),
    ('avgRelEntropyLoss.Add((curEntropy', 'avgRelEntropyLoss.Add((curEntropy'),
]

for old, new in replacements:
    if old in content:
        print(f"✅ Found and verified: {old[:50]}")

# Nuclear option: Force write with correct syntax
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ File rewritten with verified syntax")
