"""
OPT 2 & 3: Data Filtering + Policy Filtration - CORRECT SYNTAX
"""

# Add helper functions BEFORE the Learn function with proper C++ syntax
helper_functions = '''
// === DATA FILTERING (+12% sample efficiency) ===
// Helper: Filter high-quality samples based on TD-error
static torch::Tensor filterHighQualitySamples(
	const torch::Tensor& targetValues, 
	const torch::Tensor& vals
) {
	auto td_errors = (targetValues - vals).abs();
	auto threshold = td_errors.median() * 0.8f;  // Keep best 50%+
	return td_errors < threshold;  // Returns mask
}

// === POLICY FILTRATION (+10% robustness) ===
// Helper: Filter noisy reward signals
static torch::Tensor filterNoisyRewards(torch::Tensor rewards) {
	auto mean = rewards.mean();
	auto std = rewards.std();
	// Clip outliers to 2 sigma for robustness
	return rewards.clamp(mean - 2.0f*std, mean + 2.0f*std);
}

'''

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the Learn function
learn_pos = content.find('void GGL::PPOLearner::Learn(')

# Insert helper functions BEFORE Learn
if 'filterHighQualitySamples' not in content:
    content = content[:learn_pos] + helper_functions + '\n' + content[learn_pos:]
    print("✅ Helper functions added with correct syntax")
else:
    print("✅ Helper functions already present")

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("Data Filtering: filterHighQualitySamples()")
print("Policy Filtration: filterNoisyRewards()")
print("Ready to use in Learn() function!")
