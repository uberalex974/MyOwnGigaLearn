import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the reward extraction line
target = """
						torch::Tensor tRewards = torch::from_blob(envSet->state.rewards.data(), {numPlayers}, torch::kFloat).to(ppo->device, true);
						torch::Tensor tTerminals = torch::from_blob(envSet->state.terminals.data(), {numPlayers}, torch::kUInt8).to(ppo->device, true);
"""

replacement = """
						torch::Tensor tRewards = torch::from_blob(envSet->state.rewards.data(), {numPlayers}, torch::kFloat).to(ppo->device, true);
						
						// --- SANITIZE REWARDS ---
						// Fix for "Exploding Reward" bug (values ~3050).
						// We clamp and remove NaNs to ensure training stability.
						tRewards = torch::nan_to_num(tRewards, 0.0f, 0.0f, 0.0f);
						tRewards = torch::clamp(tRewards, -100.0f, 100.0f); 
						// ------------------------

						torch::Tensor tTerminals = torch::from_blob(envSet->state.terminals.data(), {numPlayers}, torch::kUInt8).to(ppo->device, true);
"""

if target in content:
    content = content.replace(target, replacement)
    print("Applied Reward Sanitization.")
else:
    print("Could not find target for Reward Sanitization.")
    # Debug
    idx = content.find("torch::from_blob(envSet->state.rewards.data()")
    if idx != -1:
        print(content[idx:idx+300])

with open(file_path, 'w') as f:
    f.write(content)
