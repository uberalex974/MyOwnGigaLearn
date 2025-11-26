import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# 1. Inject Data Layout Fix (Reshape/Permute)
# We insert this right after extracting tensors from gpuTraj
target_extract = """
					torch::Tensor tdActionMasks = gpuTraj.actionMasks.slice(0, 0, N);
"""

layout_fix = """
					torch::Tensor tdActionMasks = gpuTraj.actionMasks.slice(0, 0, N);

					// --- FIX DATA LAYOUT ---
					// Convert interleaved [T0_G0, T0_G1, ..., T1_G0...] to [G0_T0, G0_T1, ..., G1_T0...]
					// This ensures GAE sees contiguous episodes per game.
					int64_t batchSize = numRealPlayers;
					int64_t timeSteps = N / batchSize;
					
					auto fix_2d = [&](torch::Tensor t) {
						return t.view({timeSteps, batchSize, -1}).permute({1, 0, 2}).contiguous().view({N, -1});
					};
					auto fix_1d = [&](torch::Tensor t) {
						return t.view({timeSteps, batchSize}).permute({1, 0}).contiguous().view({N});
					};

					tdStates = fix_2d(tdStates);
					tdActions = fix_1d(tdActions);
					tdLogProbs = fix_1d(tdLogProbs);
					tdRewards = fix_1d(tdRewards);
					tdTerminals = fix_1d(tdTerminals);
					tdActionMasks = fix_2d(tdActionMasks);
					// -----------------------
"""

if target_extract in content:
    content = content.replace(target_extract, layout_fix)
    print("Applied Data Layout Fix.")
else:
    print("Could not find target for Data Layout Fix.")

# 2. Revert GAE to GPU
# We look for the CPU block we added
target_cpu = """
					// Fix GAE Slowness: Move to CPU for GAE computation
					// It seems GAE::Compute is slow on GPU with these tensors, so we offload to CPU.
					auto tdRewardsCPU = tdRewards.to(torch::kFloat).cpu().contiguous();
					auto tdTerminalsCPU = tdTerminals.to(torch::kChar).cpu().contiguous();
					auto tValPredsCPU = tValPreds.to(torch::kFloat).cpu().contiguous();
					torch::Tensor tTruncValPredsCPU;
					if (tTruncValPreds.defined()) tTruncValPredsCPU = tTruncValPreds.to(torch::kFloat).cpu().contiguous();
					
					torch::Tensor tAdvantages, tTargetVals, tReturns; // Declare GPU tensors
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

# We want to go back to GPU GAE
# But we must ensure types are correct (kFloat, kChar) as we found earlier.
replacement_gpu = """
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

# Note: The target_cpu string might not match exactly due to whitespace or the "Declare GPU tensors" line I added.
# I'll use a regex or simpler replacement if needed.
# Let's try exact match first, but I suspect "Declare GPU tensors" comment might be slightly different or I might have messed up the indentation in previous script.

if target_cpu in content:
    content = content.replace(target_cpu, replacement_gpu)
    print("Reverted GAE to GPU.")
else:
    print("Could not find target for GAE Revert. Trying fuzzy match.")
    # Try to find the block by unique markers
    start_marker = "// Fix GAE Slowness: Move to CPU"
    end_marker = "tReturns = tReturnsCPU.to(ppo->device, true);"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)
    
    if start_idx != -1 and end_idx != -1:
        end_idx += len(end_marker)
        content = content[:start_idx] + replacement_gpu + content[end_idx:]
        print("Reverted GAE to GPU (Fuzzy).")
    else:
        print("Failed to revert GAE.")

with open(file_path, 'w') as f:
    f.write(content)
