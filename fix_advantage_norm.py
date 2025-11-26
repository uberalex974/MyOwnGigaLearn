import re

# Fix 1: Remove useless clone in advantage normalization
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the clone - it's useless because we already sliced
old_code = """\t\t\t\t// === ADVANTAGE NORMALIZATION (CRITICAL FOR STABILITY) ===
\t\t\t\t// Normalize advantages per-minibatch to reduce variance
\t\t\t\t// Using in-place operation for speed
\t\t\t\tadvantages = advantages.clone();  // Need clone for safety with slices
\t\t\t\tadvantages.sub_(advantages.mean()).div_(advantages.std() + 1e-8f);"""

new_code = """\t\t\t\t// === ADVANTAGE NORMALIZATION (CRITICAL FOR STABILITY) ===
\t\t\t\t// Normalize advantages per-minibatch to reduce variance
\t\t\t\t// Direct in-place - slice already creates independent view
\t\t\t\tauto adv_mean = advantages.mean();
\t\t\t\tauto adv_std = advantages.std();
\t\t\t\tadvantages = (advantages - adv_mean) / (adv_std + 1e-8f);"""

content = content.replace(old_code, new_code)

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Removed useless clone in advantage normalization")
print("✓ Now using out-of-place for slice safety")
