import re

# Fix TF32 activation - need correct namespace
with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the namespace
content = content.replace('at::globalContext()', 'torch::globalContext()')

with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed TF32 namespace (at:: → torch::)")
