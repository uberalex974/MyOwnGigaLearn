import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Find the block where we calculate N
target = """
					// --- FIX DATA LAYOUT ---
					// Convert interleaved [T0_G0, T0_G1, ..., T1_G0...] to [G0_T0, G0_T1, ..., G1_T0...]
					// This ensures GAE sees contiguous episodes per game.
					int64_t batchSize = numRealPlayers;
					int64_t timeSteps = N / batchSize;
"""

replacement = """
					// --- FIX DATA LAYOUT ---
					// Convert interleaved [T0_G0, T0_G1, ..., T1_G0...] to [G0_T0, G0_T1, ..., G1_T0...]
					// This ensures GAE sees contiguous episodes per game.
					int64_t batchSize = numRealPlayers;
					
					// Ensure N is divisible by batchSize (drop partial last step if any)
					int64_t remainder = N % batchSize;
					if (remainder != 0) {
						N -= remainder;
					}
					
					int64_t timeSteps = N / batchSize;
"""

if target in content:
    content = content.replace(target, replacement)
    print("Applied N truncation fix.")
else:
    print("Could not find target for N truncation fix.")

with open(file_path, 'w') as f:
    f.write(content)
