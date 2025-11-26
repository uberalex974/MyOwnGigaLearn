"""OPT 1: Progressive Batching - Config"""
import re
with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Add progressive batch schedule
schedule_code = '''
	// === PROGRESSIVE BATCHING (+8% early efficiency) ===
	// Start with smaller batches, increase as training progresses
	auto getAdaptiveBatchSize = [](uint64_t steps) -> int {
		if (steps < 50'000'000) return 20000;  // Early: smaller batches
		if (steps < 200'000'000) return 25000; // Mid: medium batches
		return 30000;  // Late: full batches
	};
'''
# Insert after includes
insert_pos = content.find('int main(') 
content = content[:insert_pos] + schedule_code + '\n' + content[insert_pos:]

with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
print("✅ Progressive Batching added")
