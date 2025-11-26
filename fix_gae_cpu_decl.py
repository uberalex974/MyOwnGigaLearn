import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the block where we introduced the error
target = """
					torch::Tensor tAdvantagesCPU, tTargetValsCPU, tReturnsCPU;
					float rewClipPortion;

					// Run GAE on CPU
"""

replacement = """
					torch::Tensor tAdvantages, tTargetVals, tReturns; // Declare GPU tensors
					torch::Tensor tAdvantagesCPU, tTargetValsCPU, tReturnsCPU;
					float rewClipPortion;

					// Run GAE on CPU
"""

if target in content:
    content = content.replace(target, replacement)
    print("Applied declaration fix.")
else:
    print("Could not find target block.")
    # Debug
    idx = content.find("tAdvantagesCPU")
    if idx != -1:
        print(content[idx:idx+200])

with open(file_path, 'w') as f:
    f.write(content)
