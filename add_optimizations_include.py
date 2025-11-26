"""
Add include to PPOLearner.h pour utiliser les optimizations
"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'r', encoding='utf-8') as f:
    content = f.read()

# Add include after existing includes
include_line = '#include "OptimizationsHelpers.h"'

if include_line not in content:
    # Find last include
    lines = content.split('\n')
    last_include_idx = 0
    for i, line in enumerate(lines):
        if '#include' in line:
            last_include_idx = i
    
    # Insert after last include
    lines.insert(last_include_idx + 1, include_line)
    
    content = '\n'.join(lines)
    
    with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added #include OptimizationsHelpers.h to PPOLearner.h")
else:
    print("✅ Include already present")

print("\nREADY FOR FINAL BUILD WITH ALL 40 OPTIMIZATIONS!")
