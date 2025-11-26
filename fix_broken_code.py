"""Fix broken code - remove functions with wrong syntax"""
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove Data Filtering
if 'DATA FILTERING' in content:
    start = content.find('// === DATA FILTERING')
    end = content.find('}', start) + 1
    prev_nl = content.rfind('\n', 0, start)
    content = content[:prev_nl+1] + content[end+1:]

# Remove Policy Filtration  
if 'POLICY FILTRATION' in content:
    start = content.find('// === POLICY FILTRATION')
    end = content.find('}', start) + 1
    prev_nl = content.rfind('\n', 0, start)
    content = content[:prev_nl+1] + content[end+1:]

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Removed broken functions")
print("Keeping: Gradient Accumulation (config)")
print("Keeping: Progressive Batching (config in ExampleMain)")
