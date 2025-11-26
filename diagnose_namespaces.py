"""
DIAGNOSTIC: Structure des namespaces
"""

path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\OptimizationsHelpers.h'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

brace_level = 0
namespace_stack = []

print("=== NAMESPACE ANALYSIS ===\n")

for i, line in enumerate(lines):
    line = line.strip()
    
    # Count braces
    open_braces = line.count('{')
    close_braces = line.count('}')
    
    brace_level += open_braces
    brace_level -= close_braces
    
    if line.startswith("namespace "):
        ns_name = line.split()[1]
        namespace_stack.append((ns_name, brace_level))
        print(f"Line {i+1}: Start Namespace '{ns_name}' (Level {brace_level})")
    
    if close_braces > 0 and namespace_stack:
        # Heuristic: if we drop below the level where the last ns started
        last_ns, last_level = namespace_stack[-1]
        if brace_level < last_level: # This logic is a bit flawed for single line braces, but good for structure
             # Actually, let's just track indentation or explicit comments
             pass

print(f"\nFinal Brace Level: {brace_level}")
if brace_level != 0:
    print("❌ UNBALANCED BRACES! File is malformed.")
else:
    print("✅ Braces balanced.")

# Check specific namespaces
content = "".join(lines)
namespaces = ["Optimizations", "Additional", "FinalWave", "HyperScale", "QuantumReady", "MetaLearning", "Neuromorphic", "QuantumBio"]

for ns in namespaces:
    if f"namespace {ns}" in content:
        print(f"✅ Found {ns}")
    else:
        print(f"❌ MISSING {ns}")
