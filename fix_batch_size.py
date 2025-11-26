import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def fix_example_main(path):
    content = read_file(path)
    
    # Fix Batch Size to be a multiple of 4096
    # 100,000 is not divisible by 4096.
    # Closest multiple is 98,304 (24 * 4096) or 102,400 (25 * 4096).
    # We choose 98,304 to be safe on memory while keeping the scale.
    
    if "cfg.ppo.batchSize = 100000;" in content:
        print(f"Fixing batch size in {path}")
        content = content.replace(
            "cfg.ppo.batchSize = 100000;",
            "cfg.ppo.batchSize = 98304; // Multiple of 4096 (24 * 4096) for GPU efficiency"
        )
    elif "cfg.ppo.batchSize" in content:
        # Regex replacement if exact string match fails
        content = re.sub(r"cfg\.ppo\.batchSize = \d+;", "cfg.ppo.batchSize = 98304; // Multiple of 4096 (24 * 4096) for GPU efficiency", content)
        print(f"Fixing batch size in {path} (Regex)")

    write_file(path, content)

def main():
    print("Fixing Batch Size Mismatch...")
    fix_example_main(r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp")
    print("Done.")

if __name__ == "__main__":
    main()
