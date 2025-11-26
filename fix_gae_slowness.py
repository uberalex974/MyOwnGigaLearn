import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the GAE block
target = """
					report["Episode Length"] = 1.f / (tdTerminals.float().mean().item<float>() + 1e-6);

					Timer gaeTimer = {};
					// Run GAE
"""

replacement = """
					report["Episode Length"] = 1.f / (tdTerminals.to(torch::kFloat).mean().item<float>() + 1e-6);

					Timer gaeTimer = {};
					
					// Fix GAE Slowness: Ensure tensors are Float and Contiguous
					tdRewards = tdRewards.to(torch::kFloat).contiguous();
					tdTerminals = tdTerminals.to(torch::kFloat).contiguous();
					tValPreds = tValPreds.to(torch::kFloat).contiguous();
					if (tTruncValPreds.defined()) tTruncValPreds = tTruncValPreds.to(torch::kFloat).contiguous();

					// Run GAE
"""

# Note: I also updated the .float() to .to(torch::kFloat) in the report line just in case my previous fix missed it or to be consistent.
# Actually my previous fix replaced `tdTerminals.float()` with `tdTerminals.to(torch::kFloat)`.
# So the target string in file should have `.to(torch::kFloat)`.

target_fixed = """
					report["Episode Length"] = 1.f / (tdTerminals.to(torch::kFloat).mean().item<float>() + 1e-6);

					Timer gaeTimer = {};
					// Run GAE
"""

if target_fixed in content:
    content = content.replace(target_fixed, replacement)
    print("Applied GAE fix.")
else:
    print("Could not find target block. Dumping relevant part:")
    # Dump around "Episode Length"
    idx = content.find("Episode Length")
    if idx != -1:
        print(content[idx:idx+200])
    else:
        print("Could not find 'Episode Length'")

with open(file_path, 'w') as f:
    f.write(content)
