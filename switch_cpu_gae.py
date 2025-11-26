import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the GPU GAE block
target = """
					// Run GAE on GPU (Data Layout is now fixed)
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;

					// Ensure types
					tdRewards = tdRewards.to(torch::kFloat);
					tdTerminals = tdTerminals.to(torch::kChar); // GAE expects kChar/kByte
					tValPreds = tValPreds.to(torch::kFloat);
					if (tTruncValPreds.defined()) tTruncValPreds = tTruncValPreds.to(torch::kFloat);

					GAE::Compute(
						tdRewards, tdTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
"""

replacement = """
					// Run GAE on CPU (Optimized Transfer)
					// GPU GAE proved slow, so we offload to CPU with non-blocking transfers.
					auto tdRewardsCPU = tdRewards.to(torch::kFloat).cpu();
					auto tdTerminalsCPU = tdTerminals.to(torch::kChar).cpu();
					auto tValPredsCPU = tValPreds.to(torch::kFloat).cpu();
					torch::Tensor tTruncValPredsCPU;
					if (tTruncValPreds.defined()) tTruncValPredsCPU = tTruncValPreds.to(torch::kFloat).cpu();
					
					torch::Tensor tAdvantages, tTargetVals, tReturns; // GPU tensors
					torch::Tensor tAdvantagesCPU, tTargetValsCPU, tReturnsCPU;
					float rewClipPortion;

					// Run GAE on CPU
					GAE::Compute(
						tdRewardsCPU, tdTerminalsCPU, tValPredsCPU, tTruncValPredsCPU,
						tAdvantagesCPU, tTargetValsCPU, tReturnsCPU, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
					
					// Move results back to GPU (non_blocking)
					tAdvantages = tAdvantagesCPU.to(ppo->device, true);
					tTargetVals = tTargetValsCPU.to(ppo->device, true);
					tReturns = tReturnsCPU.to(ppo->device, true);
"""

if target in content:
    content = content.replace(target, replacement)
    print("Switched to CPU GAE.")
else:
    print("Could not find target for CPU GAE switch.")
    # Debug
    idx = content.find("Run GAE on GPU")
    if idx != -1:
        print(content[idx:idx+300])

with open(file_path, 'w') as f:
    f.write(content)
