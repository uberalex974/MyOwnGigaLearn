"""
FIX CORRECT: Replace AVG += patterns with .Add()
"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
changes = 0

for line in lines:
    original = line
    
    # Fix avgPolicyLoss += 
    if 'avgPolicyLoss += ' in line:
        line = line.replace('avgPolicyLoss += ', 'avgPolicyLoss.Add(')
        # Add closing paren before semicolon
        line = line.replace(';', ');')
        changes += 1
        print(f"✅ Fixed avgPolicyLoss: {original.strip()} → {line.strip()}")
    
    # Fix avgRelEntropyLoss +=
    if 'avgRelEntropyLoss += ' in line:
        line = line.replace('avgRelEntropyLoss += ', 'avgRelEntropyLoss.Add(')
        line = line.replace(';', ');')
        changes += 1
        print(f"✅ Fixed avgRelEntropyLoss: {original.strip()} → {line.strip()}")
    
    # Fix avgGuidingLoss += (si existe)
    if 'avgGuidingLoss += ' in line:
        line = line.replace('avgGuidingLoss += ', 'avgGuidingLoss.Add(')
        line = line.replace(';', ');')
        changes += 1
        print(f"✅ Fixed avgGuidingLoss: {original.strip()} → {line.strip()}")
    
    fixed_lines.append(line)

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"\n✅ TOTAL: {changes} fixes applied")
