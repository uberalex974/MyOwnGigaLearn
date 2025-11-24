#pragma once

// Minimal CUDA Optimizations header - stub implementation
// This avoids compilation errors due to missing standard library headers

#ifdef RG_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#else
// Forward declarations for non-CUDA builds
typedef void* cudaStream_t;
typedef void* cudaError_t;
#endif

namespace GGL {
    
    // Stream priority levels
    enum class StreamPriority {
        HIGH = 0,    // Critical path operations
        NORMAL = 1,  // Standard operations
        LOW = 2      // Background operations
    };
    
    // CUDA Stream Manager
    class CUDACoreStreamManager {
    private:
        int total_streams_;
        int high_priority_count_;
        
    public:
        CUDACoreStreamManager(int num_streams, int high_priority_count = 0);
        ~CUDACoreStreamManager();
        
        cudaStream_t GetStream(StreamPriority priority = StreamPriority::NORMAL);
        void RecordComputeStart(cudaStream_t stream);
        void RecordComputeEnd(cudaStream_t stream);
        void RecordMemoryStart(cudaStream_t stream);
        void RecordMemoryEnd(cudaStream_t stream);
        bool SynchronizeStream(cudaStream_t stream, int timeout_ms = 1000);
        void SynchronizeAll();
        void OptimizeForTensorOperations();
        void UpdateMetrics();
        void PrintPerformanceReport();
        void ResetStreams();
        int GetAvailableStreamCount() const;
        void EnableMemoryPool(cudaStream_t stream, size_t pool_size = 1024 * 1024);
        void SetStreamPriority(cudaStream_t stream, int priority);
        void InitializeStreams();
        void SetupPriorityStreams();
    };
    
    // Custom CUDA Kernels for RL operations
    class CUDARLKernels {
    public:
        // Optimized kernels (stub implementations)
        static void LaunchGAEKernel(float* advantages, const float* rewards, const float* values,
                                   const float* next_values, const float* dones, int batch_size,
                                   int sequence_length, float gamma, float lambda, cudaStream_t stream = nullptr);
        
        static void LaunchParallelAdvantageKernel(float* advantages, const float* rewards,
                                                const float* values, const float* next_values,
                                                const float* dones, int batch_size, float gamma,
                                                float lambda, cudaStream_t stream = nullptr);
        
        static void LaunchPolicyRatioKernel(float* ratios, const float* new_log_probs,
                                          const float* old_log_probs, int batch_size, cudaStream_t stream = nullptr);
        
        static void LaunchAdvantageNormalizationKernel(float* advantages, int batch_size,
                                                     float mean, float std, cudaStream_t stream = nullptr);
        
        static void LaunchMemoryCoalescingKernel(float* output, const float* input,
                                               int batch_size, int feature_size, cudaStream_t stream = nullptr);
        
        static void LaunchTensorCoreMatmul(float* output, const float* matrix_a,
                                         const float* matrix_b, int m, int n, int k,
                                         bool use_fp16 = true, cudaStream_t stream = nullptr);
        
    private:
        static void CheckCudaError(cudaError_t error, const char* message);
        static int GetOptimalBlockSize(int total_elements);
        static int GetOptimalGridSize(int total_elements, int block_size);
    };
    
    // Memory Coalescing Manager (simplified)
    class MemoryCoalescingManager {
    private:
        void* aligned_alloc(size_t size, size_t alignment);
        void aligned_free(void* ptr);
        
    public:
        MemoryCoalescingManager();
        ~MemoryCoalescingManager();
        
        void RegisterMemory(void* ptr, size_t size, size_t alignment = 256);
        void OptimizeMemoryLayout(void* ptr, const size_t* access_pattern, int pattern_size);
        void* AllocCoalesced(size_t size, size_t alignment = 256);
        void FreeCoalesced(void* ptr);
        float AnalyzeAccessEfficiency(void* ptr, const size_t* access_indices, int index_count);
        void CompactMemory();
        void DefragmentMemory();
        float GetAverageAccessEfficiency() const;
        size_t GetTotalCoalescedMemory() const;
        void PrintMemoryReport();
    };
    
    // Tensor Core Manager (simplified)
    class TensorCoreManager {
    private:
        bool fp16_enabled_;
        bool bf16_enabled_;
        bool int8_enabled_;
        int compute_capability_major_;
        int compute_capability_minor_;
        bool tensor_cores_available_;
        
    public:
        TensorCoreManager();
        ~TensorCoreManager();
        
        bool InitializeTensorCores();
        bool IsTensorCoreAvailable() const;
        
        // Tensor operations (stub implementations with void pointers)
        void* TensorCoreFP16Matmul(const void* a, const void* b, void* bias = nullptr, bool use_relu = false);
        void* TensorCoreBF16Matmul(const void* a, const void* b, void* bias = nullptr);
        void* MixedPrecisionForward(const void* input, const void* weights, const void* bias, int target_dtype);
        void* BatchTensorCoreMatmul(const void* batch_a, const void* batch_b, int compute_dtype);
        
        void UpdateMetrics(float speedup, const char* operation_type);
        void PrintTensorCoreReport();
        
    private:
        void DetectComputeCapability();
        void* ConvertToTensorCoreDtype(const void* tensor, int target_dtype);
    };
}