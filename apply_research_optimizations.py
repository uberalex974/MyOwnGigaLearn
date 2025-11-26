import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def apply_lr_decay(path):
    content = read_file(path)
    
    # Inject Linear LR Decay into StepCallback
    # We look for the StepCallback function body.
    # It usually starts with: void StepCallback(GGL::Learner* learner, const std::vector<RLGC::GameState>& gameStates, GGL::Report& report) {
    
    # We want to insert the decay logic at the beginning of the function or before the report.
    # Let's insert it right at the start.
    
    decay_logic = """
	// === Research Optimization: Linear Learning Rate Decay ===
	// Linearly decay learning rate from 3e-4 to 0 over 3 Billion steps.
	// This significantly improves convergence and final policy quality.
	float maxSteps = 3000000000.0f; // 3 Billion Steps
	float progress = (float)learner->totalTimesteps / maxSteps;
	if (progress > 1.0f) progress = 1.0f;
	
	float initialLR = 3e-4f;
	float minLR = 1e-6f;
	float curLR = initialLR * (1.0f - progress);
	if (curLR < minLR) curLR = minLR;
	
	learner->ppo->SetLearningRates(curLR, curLR);
	report["Learning Rate"] = curLR;
	// =========================================================
"""
    
    if "Linear Learning Rate Decay" not in content:
        print(f"Injecting LR Decay in {path}")
        # Find the start of StepCallback
        # We can search for the function signature or a known line inside it.
        # "void StepCallback(GGL::Learner* learner"
        
        # Regex to find the opening brace of StepCallback
        pattern = r"(void StepCallback\(GGL::Learner\* learner, const std::vector<RLGC::GameState>& gameStates, GGL::Report& report\)\s*\{)"
        
        if re.search(pattern, content):
            content = re.sub(pattern, r"\1" + decay_logic, content)
        else:
            print("Could not find StepCallback signature. Trying alternative match.")
            # Try matching just the function name if signature varies slightly
            if "void StepCallback" in content:
                 # Find the first brace after "void StepCallback"
                 idx = content.find("void StepCallback")
                 brace_idx = content.find("{", idx)
                 if brace_idx != -1:
                     content = content[:brace_idx+1] + decay_logic + content[brace_idx+1:]
    
    write_file(path, content)

def main():
    print("Applying Research Optimizations...")
    apply_lr_decay(r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp")
    print("Done.")

if __name__ == "__main__":
    main()
