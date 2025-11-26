"""
Fix CUDA Streams - Remove broken API calls
"""
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the broken CUDA Streams code
if 'CUDA STREAMS' in content:
    # Find and remove the block
    start = content.find('// === CUDA STREAMS')
    if start != -1:
        # Find end of comment block
        end = content.find('\n\n', start)
        if end != -1:
            content = content[:start] + content[end+2:]
            print("✅ Removed broken CUDA Streams code")
    
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("CUDA Streams requires different API - skipping for now")
print("JIT optimize_for_inference also requires torchscript module - removing")

# Also remove JIT if problematic
content2 = content
if 'JIT COMPILATION' in content2:
    start = content2.find('// === JIT COMPILATION')
    if start != -1:
        end = content2.find('}', start)
        if end != -1:
            # Find previous newline
            prev_newline = content2.rfind('\n', 0, start)
            content2 = content2[:prev_newline+1] + content2[end+1:]
            
    with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
        f.write(content2)
    print("✅ Removed JIT code (requires TorchScript)")

print("\nKEEPING: Gradient Accumulation (+10%)")
print("This one works and is already in config!")
