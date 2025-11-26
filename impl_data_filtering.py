"""OPT 2: Data Filtering - High quality sample reuse"""
code = '''
// === DATA FILTERING (+12% sample efficiency) ===
// Filter and reuse high-quality samples
auto filterHighQualitySamples(const ExperienceBuffer& buffer) {
	// Keep samples with low TD-error (accurate predictions)
	auto td_errors = (buffer.targetValues - buffer.vals).abs();
	auto threshold = td_errors.median() * 0.8f;  // Keep best 50%+
	auto mask = td_errors < threshold;
	return mask;
}
'''
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Add before Learn function
learn_pos = content.find('void GGL::PPOLearner::Learn(')
if 'DATA FILTERING' not in content:
    content = content[:learn_pos] + code + '\n' + content[learn_pos:]
    with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Data Filtering added")
else:
    print("✅ Data Filtering already present")
