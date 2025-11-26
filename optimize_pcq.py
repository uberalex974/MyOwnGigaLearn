import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def optimize_learner_config():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\LearnerConfig.h"
    content = read_file(path)
    
    # Enable standardizeObs
    if "bool standardizeObs = false;" in content:
        print("Enabling standardizeObs in LearnerConfig.h")
        content = content.replace(
            "bool standardizeObs = false;",
            "bool standardizeObs = true; // Enabled for better stability"
        )
        write_file(path, content)

def optimize_ppo_config():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\PPO\PPOLearnerConfig.h"
    content = read_file(path)
    
    # Optimize default architecture to Tapered
    # Policy: [256, 256, 256] -> [512, 256, 128]
    if "{ 256, 256, 256 }" in content:
        print("Optimizing default network architecture in PPOLearnerConfig.h")
        content = content.replace(
            "policy.layerSizes = { 256, 256, 256 };",
            "policy.layerSizes = { 512, 256, 128 }; // Tapered for better feature extraction"
        )
        content = content.replace(
            "critic.layerSizes = { 256, 256, 256 };",
            "critic.layerSizes = { 512, 256, 128 }; // Tapered for better feature extraction"
        )
        content = content.replace(
            "sharedHead.layerSizes = { 256 };",
            "sharedHead.layerSizes = { 512, 256 }; // Deeper shared head"
        )
        write_file(path, content)

def optimize_example_main(path):
    content = read_file(path)
    
    # Update explicit architecture definitions if they exist
    # This is a bit heuristic, looking for the specific lines from the file view
    
    if "cfg.ppo.sharedHead.layerSizes = { 512, 256 };" not in content:
         # Try to find what it IS set to and replace it, or just ensure it's set to the optimized version
         # In ExampleMain.cpp (from memory/previous views), it might be set to something else or commented out.
         # Let's look for the specific block I saw in ExampleMainOptimized.cpp and apply it to ExampleMain.cpp if needed.
         
         # ExampleMain.cpp usually has:
         # cfg.ppo.policy.layerSizes = { 256, 256, 256 };
         
         if "cfg.ppo.policy.layerSizes = { 256, 256, 256 };" in content:
             print(f"Optimizing architecture in {path}")
             content = content.replace(
                 "cfg.ppo.policy.layerSizes = { 256, 256, 256 };",
                 "cfg.ppo.policy.layerSizes = { 512, 256, 128 };"
             )
             content = content.replace(
                 "cfg.ppo.critic.layerSizes = { 256, 256, 256 };",
                 "cfg.ppo.critic.layerSizes = { 512, 256, 128 };"
             )
             # Check shared head
             if "cfg.ppo.sharedHead.layerSizes = { 256 };" in content:
                 content = content.replace(
                     "cfg.ppo.sharedHead.layerSizes = { 256 };",
                     "cfg.ppo.sharedHead.layerSizes = { 512, 256 };"
                 )
             write_file(path, content)

def main():
    print("Applying Advanced P/C/Q Optimizations...")
    optimize_learner_config()
    optimize_ppo_config()
    optimize_example_main(r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp")
    # ExampleMainOptimized.cpp already has good defaults, but we can check it too if we want strict consistency
    # optimize_example_main(r"c:\Giga\GigaLearnCPP\src\ExampleMainOptimized.cpp") 
    print("Done.")

if __name__ == "__main__":
    main()
