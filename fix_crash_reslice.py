import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the truncation block
target = """
					// Ensure N is divisible by batchSize (drop partial last step if any)
					int64_t remainder = N % batchSize;
					if (remainder != 0) {
						N -= remainder;
					}
"""

replacement = """
					// Ensure N is divisible by batchSize (drop partial last step if any)
					int64_t remainder = N % batchSize;
					if (remainder != 0) {
						N -= remainder;
						// Re-slice tensors to match new N
						tdStates = tdStates.slice(0, 0, N);
						tdActions = tdActions.slice(0, 0, N);
						tdLogProbs = tdLogProbs.slice(0, 0, N);
						tdRewards = tdRewards.slice(0, 0, N);
						tdTerminals = tdTerminals.slice(0, 0, N);
						tdActionMasks = tdActionMasks.slice(0, 0, N);
					}
"""

if target in content:
    content = content.replace(target, replacement)
    print("Applied re-slicing fix.")
else:
    print("Could not find target for re-slicing fix.")

with open(file_path, 'w') as f:
    f.write(content)
