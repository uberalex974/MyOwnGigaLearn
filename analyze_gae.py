import re

# OPTIMIZATION 1: Parallel GAE (vetorized implementation)
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\GAE.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Check current implementation
print("Current GAE.cpp implementation:")
print(content)
print("\n" + "="*80)
print("Analyzing for optimization opportunity...")

with open(r'c:\Giga\GigaLearnCPP\gae_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Current GAE Implementation:\n")
    f.write(content)
    f.write("\n\nNeed to check if it's already vectorized or uses a loop\n")

print("Saved current GAE to gae_analysis.txt for review")
