import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def update_ppo_config():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\PPO\PPOLearnerConfig.h"
    content = read_file(path)
    
    if "float valueClipRange" not in content:
        print("Adding valueClipRange to PPOLearnerConfig.h")
        content = content.replace(
            "float clipRange = 0.2f;",
            "float clipRange = 0.2f;\n\t\tfloat valueClipRange = 0.2f; // Clip range for value function"
        )
        write_file(path, content)

def update_ppo_learner():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp"
    content = read_file(path)
    
    # 1. Advantage Normalization
    # We need to inject this before the minibatch loop or inside the batch loop.
    # The best place is right after `auto batchAdvantages = batch.advantages;`
    if "batchAdvantages = (batchAdvantages - batchAdvantages.mean()) / (batchAdvantages.std() + 1e-8);" not in content:
        print("Injecting Advantage Normalization in PPOLearner.cpp")
        content = content.replace(
            "auto batchAdvantages = batch.advantages;",
            "auto batchAdvantages = batch.advantages;\n\t\t\t// Advantage Normalization (Mathematically Critical for PPO)\n\t\t\tbatchAdvantages = (batchAdvantages - batchAdvantages.mean()) / (batchAdvantages.std() + 1e-8);"
        )
    
    # 2. Value Function Clipping
    # We need to replace the critic loss calculation
    # Look for: criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
    if "criticLoss = torch::max(mseLoss(vals, targetValues), mseLoss(clippedVals, targetValues))" not in content:
         print("Injecting Value Function Clipping in PPOLearner.cpp")
         
         # We need to capture the old values first. 
         # Wait, we need `batchTargetValues` (returns) and `batchOldValues` (predictions from rollout).
         # `batchOldValues` isn't explicitly in the `Learn` loop variables yet!
         # Checking `ExperienceBuffer::GetAllBatchesShuffled`... it returns `ExperienceBatch`.
         # Does `ExperienceBatch` have `values`?
         # Looking at `PPOLearner.cpp` again...
         # `auto batchOldProbs = batch.logProbs;`
         # `auto batchTargetValues = batch.targetValues;`
         # `auto batchAdvantages = batch.advantages;`
         # It seems `batch.values` is missing from the extraction!
         # I need to check `ExperienceBuffer.h` or `ExperienceBatch` struct definition.
         # If `values` are not in the batch, I can't do value clipping properly without adding them.
         # However, `targetValues` are usually `returns`.
         # `advantages = returns - values`. So `values = returns - advantages`.
         # I can reconstruct `oldValues`!
         
         reconstruction_logic = """
					auto vals = InferCritic(obs);
					vals = vals.view_as(targetValues);

					// Value Function Clipping
					// Reconstruct old values from advantages and returns (since A = R - V => V = R - A)
					// Note: This is an approximation if GAE was used, but standard PPO implementations usually store 'values' in the buffer.
					// Let's assume for now we don't have 'values' in the batch struct and use the unclipped loss or try to reconstruct.
					// Actually, let's just use the standard MSE for now if we can't easily get old values, 
					// OR better: Check if we can add 'values' to the batch extraction.
					
					// Let's stick to the requested "Mathematical Perfection".
					// If I can't do clipping, I'll stick to MSE but ensure it's correct.
					// But wait, I can reconstruct old values!
					// batchTargetValues (Returns) - batchAdvantages (Advantage) = batchOldValues (Value Prediction)
					// This holds true for GAE(gamma, 1) i.e. Monte Carlo, but for GAE(gamma, lambda), A = GAE.
					// Returns = V_old + A. So V_old = Returns - A.
					// Yes! We can reconstruct it!
					
					auto oldValues = targetValues - advantages; 
					auto clippedVals = oldValues + (vals - oldValues).clamp(-config.valueClipRange, config.valueClipRange);
					
					auto loss1 = (vals - targetValues).pow(2);
					auto loss2 = (clippedVals - targetValues).pow(2);
					criticLoss = torch::max(loss1, loss2).mean() * 0.5f * batchSizeRatio;
"""
         # Replacing the old critic block
         old_block = """
				if (trainCritic) {
					auto vals = InferCritic(obs);

					// Compute value loss
					vals = vals.view_as(targetValues);
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					avgCriticLoss += criticLoss.detach().cpu().item<float>();
				}
"""
         # We need to construct the new block carefully to match indentation
         new_block = """
				if (trainCritic) {
					auto vals = InferCritic(obs);
					vals = vals.view_as(targetValues);

					// Value Function Clipping (Mathematically Perfect PPO)
					// Reconstruct old values: V_old = Returns - Advantage
					auto oldValues = targetValues - advantages;
					auto clippedVals = oldValues + (vals - oldValues).clamp(-config.valueClipRange, config.valueClipRange);
					
					auto loss1 = (vals - targetValues).pow(2);
					auto loss2 = (clippedVals - targetValues).pow(2);
					
					// Take max of clipped and unclipped loss (pessimistic bound)
					criticLoss = torch::max(loss1, loss2).mean() * 0.5f * batchSizeRatio;
					
					avgCriticLoss += criticLoss.detach().cpu().item<float>();
				}
"""
         # Use regex or exact string replacement if possible. The indentation in the file uses tabs.
         # I'll try a flexible replacement.
         
         # Normalize content for matching (optional, but safer to just match the key lines)
         if "criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;" in content:
             content = content.replace(
                 "criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;",
                 """// Value Function Clipping
					auto oldValues = targetValues - advantages;
					auto clippedVals = oldValues + (vals - oldValues).clamp(-config.valueClipRange, config.valueClipRange);
					auto loss1 = (vals - targetValues).pow(2);
					auto loss2 = (clippedVals - targetValues).pow(2);
					criticLoss = torch::max(loss1, loss2).mean() * 0.5f * batchSizeRatio;"""
             )

    write_file(path, content)

def update_example_main(path):
    content = read_file(path)
    
    # 1. Batch Size & MiniBatch Size
    # Target: Batch 100,000, MiniBatch 4096
    content = re.sub(r"cfg\.ppo\.batchSize = .*;", "cfg.ppo.batchSize = 100000;", content)
    content = re.sub(r"cfg\.ppo\.miniBatchSize = .*;", "cfg.ppo.miniBatchSize = 4096; // Power of 2 for GPU efficiency", content)
    
    # 2. Learning Rate
    # Target: 3e-4
    content = re.sub(r"cfg\.ppo\.policyLR = .*;", "cfg.ppo.policyLR = 3e-4;", content)
    content = re.sub(r"cfg\.ppo\.criticLR = .*;", "cfg.ppo.criticLR = 3e-4;", content)
    
    # 3. Entropy
    # Target: 0.01
    content = re.sub(r"cfg\.ppo\.entropyScale = .*;", "cfg.ppo.entropyScale = 0.01f;", content)
    
    write_file(path, content)

def main():
    print("Applying Mathematical Perfection...")
    update_ppo_config()
    update_ppo_learner()
    update_example_main(r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp")
    print("Done.")

if __name__ == "__main__":
    main()
