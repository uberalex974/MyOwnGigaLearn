import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# The block end we are looking for
target_end = """
						"Total Iterations"
					}
				);
			}
		}"""

# We need to add one more closing brace for the try block
replacement = target_end + "\n\t}"

if target_end in content:
    content = content.replace(target_end, replacement)
    print("Found target block and applied fix.")
else:
    # Try with different whitespace if needed, or just print failure
    print("Could not find target block. Dumping last 500 chars for debug:")
    print(content[-500:])

with open(file_path, 'w') as f:
    f.write(content)
