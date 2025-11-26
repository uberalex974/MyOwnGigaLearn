import os
import re
import math

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def optimize_cmake():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\CMakeLists.txt"
    content = read_file(path)
    
    # Add /fp:fast
    if "/fp:fast" not in content:
        print("Adding /fp:fast to CMakeLists.txt")
        content = content.replace(
            "/arch:AVX2",
            "/arch:AVX2 /fp:fast"
        )
    write_file(path, content)

def optimize_config():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\PPO\PPOLearnerConfig.h"
    content = read_file(path)
    
    # Increase Reward Clipping to 100 (Safety for Goals)
    if "float rewardClipRange = 10;" in content:
        print("Increasing rewardClipRange to 100")
        content = content.replace(
            "float rewardClipRange = 10;",
            "float rewardClipRange = 100; // Increased for safety with high rewards (goals)"
        )
    
    write_file(path, content)

def optimize_learner_cpp():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"
    content = read_file(path)
    
    # Optimize TENSOR_TO_VEC with memcpy
    # Look for: auto curActions = TENSOR_TO_VEC<int>(tActions);
    if "std::memcpy(curActions.data()" not in content:
        print("Optimizing Learner.cpp: Replacing TENSOR_TO_VEC with memcpy")
        
        # We need to include <cstring> for memcpy if not present, but it's likely there or transitively included.
        # Safer to just use std::memcpy and hope, or add include.
        # Let's add include to be safe.
        if "#include <cstring>" not in content:
             content = "#include <cstring>\n" + content

        # Replace the call
        # "auto curActions = TENSOR_TO_VEC<int>(tActions);"
        # Replacement:
        # std::vector<int> curActions(tActions.numel());
        # std::memcpy(curActions.data(), tActions.data_ptr<int>(), tActions.numel() * sizeof(int));
        
        content = content.replace(
            "auto curActions = TENSOR_TO_VEC<int>(tActions);",
            "// Optimization: Direct memcpy for speed\n\t\t\t\t\t\tstd::vector<int> curActions(tActions.numel());\n\t\t\t\t\t\tstd::memcpy(curActions.data(), tActions.data_ptr<int>(), tActions.numel() * sizeof(int));"
        )
        
    write_file(path, content)

def optimize_example_main():
    path = r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp"
    content = read_file(path)
    
    # Replace Linear Decay with Cosine Annealing
    # We look for the block we added earlier
    
    cosine_logic = """
	// === Research Optimization: Cosine Annealing Learning Rate ===
	// Cosine schedule provides better convergence than linear.
	// Decays from 3e-4 to 1e-6 over 3 Billion steps.
	float maxSteps = 3000000000.0f; // 3 Billion Steps
	float progress = (float)learner->totalTimesteps / maxSteps;
	if (progress > 1.0f) progress = 1.0f;
	
	float initialLR = 3e-4f;
	float minLR = 1e-6f;
	
	// Cosine Annealing Formula: eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * progress))
	float pi = 3.14159265359f;
	float curLR = minLR + 0.5f * (initialLR - minLR) * (1.0f + cosf(pi * progress));
	
	learner->SetPPO_LR(curLR);
	report["Learning Rate"] = curLR;
	// =========================================================
"""
    
    if "Linear Learning Rate Decay" in content:
        print("Upgrading to Cosine Annealing in ExampleMain.cpp")
        # Regex to replace the previous block
        # The previous block started with "// === Research Optimization: Linear Learning Rate Decay ==="
        # and ended with "// ========================================================="
        
        pattern = r"// === Research Optimization: Linear Learning Rate Decay ===.*?// ========================================================="
        content = re.sub(pattern, cosine_logic.strip(), content, flags=re.DOTALL)
        
    write_file(path, content)

def main():
    print("Applying Deep Optimizations...")
    optimize_cmake()
    optimize_config()
    optimize_learner_cpp()
    optimize_example_main()
    print("Done.")

if __name__ == "__main__":
    main()
