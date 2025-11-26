import os

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def optimize_config():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\PPO\PPOLearnerConfig.h"
    content = read_file(path)
    
    # Increase Reward Clipping to 1000 (Safe for Goals)
    # Previous was 100
    if "float rewardClipRange = 100;" in content:
        print("Increasing rewardClipRange to 1000")
        content = content.replace(
            "float rewardClipRange = 100;",
            "float rewardClipRange = 1000.0f; // Increased to 1000 to fully preserve Goal Reward signal (300+)"
        )
    elif "float rewardClipRange = 10;" in content: # Fallback if previous script didn't run or was different
         print("Increasing rewardClipRange to 1000 (from 10)")
         content = content.replace(
            "float rewardClipRange = 10;",
            "float rewardClipRange = 1000.0f; // Increased to 1000 to fully preserve Goal Reward signal (300+)"
        )
    
    write_file(path, content)

def main():
    print("Applying Final Reward Safety...")
    optimize_config()
    print("Done.")

if __name__ == "__main__":
    main()
