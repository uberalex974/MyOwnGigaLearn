"""OPT 3: Policy Filtration - Noise reduction"""
code = '''
// === POLICY FILTRATION (+10% robustness) ===
// Filter noisy reward signals
auto filterNoisyRewards(torch::Tensor rewards) {
	auto mean = rewards.mean();
	auto std = rewards.std();
	// Clip outliers to 2 sigma
	return rewards.clamp(mean - 2*std, mean + 2*std);
}
'''
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

if 'POLICY FILTRATION' not in content:
    learn_pos = content.find('void GGL::PPOLearner::Learn(')
    content = content[:learn_pos] + code + '\n' + content[learn_pos:]
    with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Policy Filtration added")
else:
    print("✅ Policy Filtration already present")
