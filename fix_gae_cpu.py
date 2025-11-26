import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the previous fix block
target = """
					// Fix GAE Slowness: Ensure tensors are Float and Contiguous
					tdRewards = tdRewards.to(torch::kFloat).contiguous();
					tdTerminals = tdTerminals.to(torch::kChar).contiguous(); // kChar (int8) matches original
					tValPreds = tValPreds.to(torch::kFloat).contiguous();
					if (tTruncValPreds.defined()) tTruncValPreds = tTruncValPreds.to(torch::kFloat).contiguous();

					// Run GAE
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;
					GAE::Compute(
						tdRewards, tdTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
"""

replacement = """
					// Fix GAE Slowness: Move to CPU for GAE computation
					// It seems GAE::Compute is slow on GPU with these tensors, so we offload to CPU.
					auto tdRewardsCPU = tdRewards.to(torch::kFloat).cpu().contiguous();
					auto tdTerminalsCPU = tdTerminals.to(torch::kChar).cpu().contiguous();
					auto tValPredsCPU = tValPreds.to(torch::kFloat).cpu().contiguous();
					torch::Tensor tTruncValPredsCPU;
					if (tTruncValPreds.defined()) tTruncValPredsCPU = tTruncValPreds.to(torch::kFloat).cpu().contiguous();
					
					torch::Tensor tAdvantagesCPU, tTargetValsCPU, tReturnsCPU;
					float rewClipPortion;

					// Run GAE on CPU
					GAE::Compute(
						tdRewardsCPU, tdTerminalsCPU, tValPredsCPU, tTruncValPredsCPU,
						tAdvantagesCPU, tTargetValsCPU, tReturnsCPU, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
					
					// Move results back to GPU
					tAdvantages = tAdvantagesCPU.to(ppo->device, true);
					tTargetVals = tTargetValsCPU.to(ppo->device, true);
					tReturns = tReturnsCPU.to(ppo->device, true);
"""

# Note: The target block in file might have slightly different whitespace or comments due to previous replacements.
# I'll try to match a smaller unique part if exact match fails.

target_start = "// Fix GAE Slowness: Ensure tensors are Float and Contiguous"
target_end = "config.ppo.rewardClipRange\n\t\t\t\t\t);"

start_idx = content.find(target_start)
end_idx = content.find(target_end, start_idx)

if start_idx != -1 and end_idx != -1:
    end_idx += len(target_end)
    content = content[:start_idx] + replacement + content[end_idx:]
    print("Applied GAE CPU fix.")
else:
    print("Could not find target block for GAE CPU fix.")
    # Debug
    print(content[content.find("Timer gaeTimer"):content.find("Timer gaeTimer")+500])

with open(file_path, 'w') as f:
    f.write(content)
