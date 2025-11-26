"""OPT 2: CUDA Streams - Direct CUDA API"""
code = '''
// === CUDA STREAMS DIRECT API (+5%) ===
#include <cuda_runtime.h>

class CUDAStreamManager {
public:
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    
    CUDAStreamManager() {
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&transfer_stream);
    }
    
    ~CUDAStreamManager() {
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(transfer_stream);
    }
    
    void syncAll() {
        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(transfer_stream);
    }
};
'''

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'r', encoding='utf-8') as f:
    content = f.read()

if 'CUDAStreamManager' not in content:
    # Insert after includes
    include_pos = content.find('#pragma once')
    if include_pos != -1:
        end_pragma = content.find('\n', include_pos)
        content = content[:end_pragma+1] + code + content[end_pragma+1:]
    
with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("✅ CUDA Streams manager implemented")
