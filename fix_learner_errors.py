import os

file_path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"

with open(file_path, 'r') as f:
    content = f.read()

# Fix 1 & 2: .data.data() -> .data()
content = content.replace("envSet->state.rewards.data.data()", "envSet->state.rewards.data()")
content = content.replace("envSet->state.terminals.data.data()", "envSet->state.terminals.data()")

# Fix 3: .float() -> .to(torch::kFloat)
content = content.replace("tdTerminals.float()", "tdTerminals.to(torch::kFloat)")

with open(file_path, 'w') as f:
    f.write(content)

print("Applied fixes to Learner.cpp")
