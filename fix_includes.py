import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def fix_includes(path):
    content = read_file(path)
    
    # Add PPOLearner.h include
    if "#include <GigaLearnCPP/PPO/PPOLearner.h>" not in content:
        print(f"Adding PPOLearner.h include to {path}")
        # Find the last include
        if "#include <GigaLearnCPP/Learner.h>" in content:
            content = content.replace(
                "#include <GigaLearnCPP/Learner.h>",
                "#include <GigaLearnCPP/Learner.h>\n#include <GigaLearnCPP/PPO/PPOLearner.h>"
            )
        else:
            # Just add it at the top
            content = "#include <GigaLearnCPP/PPO/PPOLearner.h>\n" + content
            
    write_file(path, content)

def main():
    print("Fixing Includes...")
    fix_includes(r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp")
    print("Done.")

if __name__ == "__main__":
    main()
