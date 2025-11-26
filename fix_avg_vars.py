"""
FIX SCRIPT: Ajouter les declarations manquantes avgPolicyLoss, avgRelEntropyLoss, avgGuidingLoss
"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the Learn function and the avgTracker declarations
fixed_lines = []
in_learn_function = False
added_fix = False

for i, line in enumerate(lines):
    fixed_lines.append(line)
    
    # Detect Learn function start
    if 'void GGL::PPOLearner::Learn(' in line:
        in_learn_function = True
    
    # After MutAvgTracker declarations, add missing variables
    if in_learn_function and not added_fix and 'avgClip;' in line:
        # Add float declarations for the missing variables
        fixed_lines.append('\n')
        fixed_lines.append('\t// Fix: Declare accumulator floats\n')
        fixed_lines.append('\tfloat avgPolicyLossFloat = 0.0f;\n')
        fixed_lines.append('\tfloat avgRelEntropyLossFloat = 0.0f;\n')
        fixed_lines.append('\tfloat avgGuidingLossFloat = 0.0f;\n')
        added_fix = True
        print("✅ Added missing variable declarations")

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ Fixed avgPolicyLoss declarations by script")
print("   - avgPolicyLossFloat")  
print("   - avgRelEntropyLossFloat")
print("   - avgGuidingLossFloat")
