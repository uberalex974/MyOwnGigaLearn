"""
FIX FINAL: Le code utilise 'avgPolicyLoss' comme MutAvgTracker, pas comme float
On doit utiliser les MutAvgTrackers qui sont déjà déclarés
"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Le problème: avgPolicyLoss est un MutAvgTracker, pas un float
# On doit utiliser avgPolicyLoss.Add() au lieu de +=

# Replace all += with .Add() for the Mut AvgTrackers
content = content.replace('avgPolicyLoss += curPolicyLoss;', 'avgPolicyLoss.Add(curPolicyLoss);')
content = content.replace('avgRelEntropyLoss += (curEntropy', 'avgRelEntropyLoss.Add((curEntropy')
# Fix the closing paren
content = content.replace('avgRelEntropyLoss.Add((curEntropy * config.entropyScale) / curPolicyLoss;', 
                          'avgRelEntropyLoss.Add((curEntropy * config.entropyScale) / curPolicyLoss);')

# Remove the float declarations we added (they're wrong)
lines = content.split('\n')
fixed_lines = []
for line in lines:
    if 'avgPolicyLossFloat' not in line and 'avgRelEntropyLossFloat' not in line and 'avgGuidingLossFloat' not in line:
        if '// Fix: Declare accumulator floats' not in line:
            fixed_lines.append(line)

content = '\n'.join(fixed_lines)

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed avgPolicyLoss usage")
print("   - Changed += to .Add() for MutAvgTracker")
print("   - Removed wrong float declarations")
