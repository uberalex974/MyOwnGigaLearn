import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def update_learner_h(path):
    content = read_file(path)
    if "void SetPPO_LR(float lr);" not in content:
        print(f"Adding SetPPO_LR to {path}")
        content = content.replace(
            "void LoadStats(std::filesystem::path path);",
            "void LoadStats(std::filesystem::path path);\n\t\tvoid SetPPO_LR(float lr);"
        )
    write_file(path, content)

def update_learner_cpp(path):
    content = read_file(path)
    if "void GGL::Learner::SetPPO_LR(float lr)" not in content:
        print(f"Implementing SetPPO_LR in {path}")
        # Add implementation at the end
        content += "\nvoid GGL::Learner::SetPPO_LR(float lr) {\n\tppo->SetLearningRates(lr, lr);\n}\n"
    write_file(path, content)

def update_example_main(path):
    content = read_file(path)
    
    # Remove the failing include
    if "#include <GigaLearnCPP/PPO/PPOLearner.h>" in content:
        print(f"Removing failing include from {path}")
        content = content.replace("#include <GigaLearnCPP/PPO/PPOLearner.h>\n", "")
        content = content.replace("#include <GigaLearnCPP/PPO/PPOLearner.h>", "")

    # Update the call
    if "learner->ppo->SetLearningRates(curLR, curLR);" in content:
        print(f"Updating SetLearningRates call in {path}")
        content = content.replace(
            "learner->ppo->SetLearningRates(curLR, curLR);",
            "learner->SetPPO_LR(curLR);"
        )
        
    write_file(path, content)

def main():
    print("Implementing SetPPO_LR Wrapper...")
    update_learner_h(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.h")
    update_learner_cpp(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp")
    update_example_main(r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp")
    print("Done.")

if __name__ == "__main__":
    main()
