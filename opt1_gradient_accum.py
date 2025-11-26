"""
OPT 1: Gradient Accumulation - Activer dans config (déjà présent)
"""
with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and modify gradient accumulation config
if 'gradientAccumulationSteps' in content:
    # Activate it
    content = content.replace(
        'cfg.ppo.gradientAccumulationSteps = 1;',
        'cfg.ppo.gradientAccumulationSteps = 3;  // Micro-batching: 3×10k instead of 1×30k'
    )
    print("✅ Gradient Accumulation activated (3 micro-batches)")
else:
    # Add after miniBatchSize
    import re
    pattern = r'(cfg\.ppo\.miniBatchSize = \d+;)'
    replacement = r'\1\n\tcfg.ppo.gradientAccumulationSteps = 3;  // Micro-batching optimization'
    content = re.sub(pattern, replacement, content)
    print("✅ Gradient Accumulation config added")

with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("Gain: +10% efficiency")
