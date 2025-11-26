import os
import re

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

def check(name, condition, details=""):
    if condition:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} - {details}")

def verify_optimizations():
    print("=== Starting Safety Audit of Optimizations ===\n")

    # 1. Async Metrics
    content = read_file(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Util\MetricSender.cpp")
    if content:
        check("Async Metrics (Thread)", "std::thread" in content and ".detach()" in content)
        check("Async Metrics (GIL)", "py::gil_scoped_acquire" in content)
    else:
        check("Async Metrics", False, "File not found")

    # 2. Parallel Obs Norm & Memcpy
    content = read_file(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp")
    if content:
        check("Parallel Obs Norm (OpenMP)", "#pragma omp parallel for" in content)
        check("Parallel Obs Norm (Direct Pointer)", "envSet->state.obs.data.data()" in content or "rawData" in content)
        check("Memcpy Optimization", "std::memcpy" in content)
    else:
        check("Learner Optimizations", False, "File not found")

    # 3. Fast Math & OpenMP Config
    content = read_file(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\CMakeLists.txt")
    if content:
        check("Fast Math Flag", "/fp:fast" in content)
        check("OpenMP Flag", "openmp" in content.lower() or "OpenMP" in content)
    else:
        check("CMake Config", False, "File not found")

    # 4. Pinned Memory
    content = read_file(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\ExperienceBuffer.cpp")
    if content:
        check("Pinned Memory", ".pin_memory()" in content)
    else:
        check("ExperienceBuffer", False, "File not found")

    # 5. Cosine Annealing
    content = read_file(r"c:\Giga\GigaLearnCPP\src\ExampleMain.cpp")
    if content:
        check("Cosine Annealing Formula", "cosf(pi * progress)" in content)
        check("Cosine Annealing Min LR", "minLR" in content)
    else:
        check("ExampleMain", False, "File not found")

    # 6. PPO Logic (Grad Accum, Adv Norm, Value Clip)
    content = read_file(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp")
    if content:
        check("Gradient Accumulation", "gradientAccumulationSteps" in content)
        check("Advantage Normalization", "(batchAdvantages - batchAdvantages.mean())" in content)
        check("Value Clipping", "valueClipRange" in content)
    else:
        check("PPOLearner", False, "File not found")

    # 7. Config (Reward Clip, Tapered Layers)
    content = read_file(r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\PPO\PPOLearnerConfig.h")
    if content:
        check("Reward Clipping (1000.0)", "rewardClipRange = 1000.0f" in content or "rewardClipRange = 1000" in content)
        check("Tapered Layers", "512, 256, 128" in content)
    else:
        check("PPOLearnerConfig", False, "File not found")

    # 8. Documentation Integrity
    content = read_file(r"c:\Giga\GigaLearnCPP\PROJECT_STRUCTURE.md")
    if content:
        check("Docs: Real Optimizations Section", "Real Optimizations (Verified Active" in content)
        check("Docs: Original Content Preserved", len(content) > 1000) # Simple check for size
    else:
        check("PROJECT_STRUCTURE.md", False, "File not found")

    print("\n=== Audit Complete ===")

if __name__ == "__main__":
    verify_optimizations()
