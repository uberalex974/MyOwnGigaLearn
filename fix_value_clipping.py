import re

# Read the file
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Old code to replace
old_code = """torch::Tensor criticLoss;
\t\t\t\tif (trainCritic) {
\t\t\t\t\tauto vals = InferCritic(obs);
\t\t\t\t\tvals = vals.view_as(targetValues);

\t\t\t\t\t// === VALUE FUNCTION CLIPPING (PREVENTS SPIKES) ===
\t\t\t\t\t// Clip value updates similar to policy clipping for stability
\t\t\t\t\t// Note: We don't have oldVals in experience buffer, so use simplified MSE for now
\t\t\t\t\t// TODO: Add oldVals to experience buffer for full clipping implementation
\t\t\t\t\tcriticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
\t\t\t\t\tavgCriticLoss += criticLoss.detach().cpu().item<float>();
\t\t\t\t}"""

# New code
new_code = """torch::Tensor criticLoss;
\t\t\t\tif (trainCritic) {
\t\t\t\t\tauto vals = InferCritic(obs);
\t\t\t\t\tvals = vals.view_as(targetValues);
\t\t\t\t\tauto oldVals = batch.vals.slice(0, start, stop).to(device, true, true);

\t\t\t\t\t// === VALUE FUNCTION CLIPPING (PREVENTS CRITIC SPIKES) ===
\t\t\t\t\t// Clip value updates to prevent large spikes in estimation
\t\t\t\t\tauto valsClipped = oldVals + (vals - oldVals).clamp(-config.clipRange, config.clipRange);
\t\t\t\t\tauto loss1 = (vals - targetValues).pow(2);
\t\t\t\t\tauto loss2 = (valsClipped - targetValues).pow(2);
\t\t\t\t\tcriticLoss = torch::max(loss1, loss2).mean() * 0.5f * batchSizeRatio;
\t\t\t\t\tavgCriticLoss += criticLoss.detach().cpu().item<float>();
\t\t\t\t}"""

# Replace
content = content.replace(old_code, new_code)

# Write back
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("Value Function Clipping implemented successfully!")
