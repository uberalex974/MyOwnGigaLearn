import re

# Remove TF32 for now - seems LibTorch C++ doesn't expose it easily
with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the TF32 lines
content = content.replace("""\t// === ENABLE TF32 (TensorFloat-32) FOR RTX 30xx/40xx ===
\t// Massive speedup (+30-50%) for matrix operations with no precision loss
\ttorch::globalContext().setAllowTF32CuBLAS(true);
\ttorch::globalContext().setAllowTF32CuDNN(true);

""", "")

with open(r'c:\Giga\GigaLearnCPP\src\ExampleMain.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("TF32 removed (not available in LibTorch C++ API)")
print("Note: TF32 might be enabled by default on compatible GPUs")
