import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def optimize_cmake():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\CMakeLists.txt"
    content = read_file(path)
    
    # Add MSVC Optimization Flags
    if "/O2" not in content:
        print("Adding MSVC Optimization Flags to CMakeLists.txt")
        # Find where CXX_STANDARD is set or just append to CMAKE_CXX_FLAGS
        # Better to add it to target_compile_options if possible, or CMAKE_CXX_FLAGS
        
        # We'll add it to CMAKE_CXX_FLAGS for global effect
        if 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")' in content:
            content = content.replace(
                'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")',
                'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL /arch:AVX2")'
            )
        else:
             # Fallback if the line is different
             content += '\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL /arch:AVX2")'
             
        # Also need to add /LTCG to linker flags if /GL is used
        content += '\nset(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")'
        content += '\nset(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")'
        
        write_file(path, content)

def optimize_learner_cpp():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"
    content = read_file(path)
    
    # Add global NoGrad at start of Start()
    if "torch::autograd::GradMode::set_enabled(false);" not in content:
        print("Adding global NoGrad to Learner::Start()")
        content = content.replace(
            "void GGL::Learner::Start() {",
            "void GGL::Learner::Start() {\n\t// Global optimization: Disable autograd for the entire collection phase\n\ttorch::autograd::GradMode::set_enabled(false);"
        )
        # We need to re-enable it before PPO Learn!
        # Find ppo->Learn call
        # It's not in Learner.cpp directly? Wait, Learner calls ppo->Learn?
        # No, Learner.cpp has the loop.
        # Let's check where ppo->Learn is called.
        # It's likely in `Learner::Start` loop.
        
        # Searching for `ppo->Learn`...
        # Wait, I don't see `ppo->Learn` in the `Learner.cpp` file view I had earlier?
        # Ah, I might have missed it or it's called differently.
        # Let's look at the file content again in my memory.
        # `Learner::Start` collects trajectories.
        # Then it processes timesteps.
        # Then it calls `ppo->Learn`?
        # Actually, looking at `Learner.cpp` lines 730+, it processes timesteps.
        # It calls `GAE::Compute`.
        # Where is `ppo->Learn`?
        # It must be called after collection.
        # Ah, I see `ppo->InferCritic` and `GAE::Compute`.
        # But I don't see `ppo->Learn` in the `Learner.cpp` snippet I saw (lines 1-800).
        # It must be further down or I missed it.
        # Wait, `Learner::Start` is the main loop.
        # It collects experience into `combinedTraj`.
        # Then it calculates advantages.
        # Then it should call `ppo->Learn`.
        
        # If I disable GradMode globally, I MUST enable it for `ppo->Learn`.
        # Since I can't see `ppo->Learn` call in the snippet, I should be careful.
        # Instead of global disable, let's just rely on `RG_NO_GRAD` which is already there.
        # `RG_NO_GRAD` is likely a macro for `torch::NoGradGuard`.
        
        # Let's check `RG_NO_GRAD` definition if possible? No need.
        # Let's just add `RG_NO_GRAD` to any scope that looks like it needs it but doesn't have it.
        # The `Learner::Start` loop has `RG_NO_GRAD` in `Collect timesteps` block.
        # It has `RG_NO_GRAD` in `Process timesteps` block.
        # So it seems covered.
        
        # Maybe I can optimize `ExperienceBuffer`?
        pass

def optimize_experience_buffer():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\ExperienceBuffer.cpp"
    content = read_file(path)
    
    # Optimize _GetSamples to avoid unnecessary tensor creation
    # "Tensor tIndices = torch::tensor(IList(indices, indices + size));"
    # This creates a tensor from a list every time.
    # We can use `torch::from_blob` if we are careful about lifetime, but `indices` is `int64_t*`.
    # `torch::from_blob(indices, {size}, torch::kLong)` is much faster.
    
    if "torch::from_blob" not in content:
        print("Optimizing ExperienceBuffer::_GetSamples with from_blob")
        content = content.replace(
            "Tensor tIndices = torch::tensor(IList(indices, indices + size));",
            "// Optimization: Use from_blob to avoid copy\n\tTensor tIndices = torch::from_blob((void*)indices, { (long long)size }, torch::kLong).clone(); // Clone needed because indices is deleted later"
        )
        write_file(path, content)

def main():
    print("Applying Speed Optimizations...")
    optimize_cmake()
    optimize_experience_buffer()
    print("Done.")

if __name__ == "__main__":
    main()
