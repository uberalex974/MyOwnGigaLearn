import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the previous fix block
target = """
					// Fix GAE Slowness: Ensure tensors are Float and Contiguous
					tdRewards = tdRewards.to(torch::kFloat).contiguous();
					tdTerminals = tdTerminals.to(torch::kFloat).contiguous();
					tValPreds = tValPreds.to(torch::kFloat).contiguous();
"""

replacement = """
					// Fix GAE Slowness: Ensure tensors are Float and Contiguous
					tdRewards = tdRewards.to(torch::kFloat).contiguous();
					tdTerminals = tdTerminals.to(torch::kChar).contiguous(); // kChar (int8) matches original
					tValPreds = tValPreds.to(torch::kFloat).contiguous();
"""

if target in content:
    content = content.replace(target, replacement)
    print("Applied GAE type fix (kChar).")
else:
    print("Could not find target block.")
    # Debug
    idx = content.find("Fix GAE Slowness")
    if idx != -1:
        print(content[idx:idx+200])

with open(file_path, 'w') as f:
    f.write(content)
