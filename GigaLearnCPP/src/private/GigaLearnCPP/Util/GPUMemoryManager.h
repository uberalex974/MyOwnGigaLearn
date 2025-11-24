#pragma once

#ifdef RG_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#else
// Forward declarations for non-CUDA builds
typedef void* cudaStream_t;
#endif

namespace GGL {

/// 🔥 OPTIMIZED: GPU Memory Manager for Single-GPU Training
/// Provides intelligent memory management, pooling, and optimization
class GPUMemoryManager {
public:
    struct MemoryConfig {
        float max_memory_fraction;      // Keep 15% memory free
        bool enable_memory_pooling;      // Pool tensor allocations
        bool enable_garbage_collection;  // Auto-cleanup unused tensors
        bool enable_memory_compaction;   // Defragment GPU memory
        
        MemoryConfig() : 
            max_memory_fraction(0.85f),
            enable_memory_pooling(true),
            enable_garbage_collection(true),
            enable_memory_compaction(true) {}
    };

    struct MemoryStats {
        size_t total_memory;            // Total GPU memory
        size_t free_memory;             // Free memory
        size_t allocated_memory;        // Currently allocated
        size_t cached_memory;           // Cache pool size
        float fragmentation_ratio;      // Memory fragmentation
        int pool_count;                 // Number of memory pools
        
        MemoryStats() :
            total_memory(0), free_memory(0), allocated_memory(0),
            cached_memory(0), fragmentation_ratio(0.0f), pool_count(0) {}
    };

    struct TensorAllocation {
        void* ptr;                      // GPU pointer
        size_t size;                    // Allocation size
        size_t pool_id;                // Pool identifier
        bool is_active;                // Currently in use
        long long last_used;           // Timestamp instead of chrono
        const char* name;              // Simple char pointer instead of string
        
        TensorAllocation() :
            ptr(nullptr), size(0), pool_id(0), 
            is_active(false), last_used(0), name(nullptr) {}
    };

    // Singleton instance
    static GPUMemoryManager& Instance();

    // Initialize GPU memory manager
    virtual void initialize(const MemoryConfig& config);

    // Memory allocation with pooling
    virtual void* allocate_tensor(size_t size, const char* name = "tensor");
    virtual void deallocate_tensor(void* ptr);

    // Memory optimization
    virtual void optimize_memory_layout();
    virtual void compact_memory_pools();
    virtual void defragment_memory();

    // Memory statistics and monitoring
    virtual MemoryStats get_memory_stats() const;
    virtual size_t get_available_memory() const;
    virtual void set_memory_limit(size_t limit);

    // Single-GPU specific optimizations
    virtual void enable_unified_memory();               // Use CUDA unified memory
    virtual void setup_memory_pools();                  // Pre-allocate memory pools
    virtual void garbage_collect();                     // Free unused tensors

    // Performance monitoring
    virtual float get_memory_efficiency() const;
    virtual void log_memory_usage() const;

    // Constructor and destructor
    GPUMemoryManager();
    virtual ~GPUMemoryManager();

protected:
    MemoryConfig config_;
    static const int MAX_ALLOCATIONS = 1000;
    TensorAllocation allocations_[MAX_ALLOCATIONS];     // Fixed array instead of vector
    int allocation_count_;
    void* memory_pools_[3];                            // Simple pointer array
    size_t total_memory_limit_;
    float fragmentation_threshold_;

    // Internal helper methods
    virtual void initialize_memory_pools(size_t total_memory);
    virtual void* get_from_pool(size_t size);
    virtual void return_to_pool(void* ptr, size_t size);
    virtual size_t find_optimal_pool_size(size_t requested_size) const;
};

/// 🔥 OPTIMIZED: CUDA Stream Manager for Parallel Operations
/// Manages multiple CUDA streams for compute/memory overlap
class CUDAStreamManager {
public:
    struct StreamConfig {
        int num_compute_streams;            // Parallel compute operations
        int num_memory_streams;             // Memory transfer streams  
        bool enable_stream_priorities;
        bool enable_kernel_fusion;
        
        StreamConfig() :
            num_compute_streams(3),
            num_memory_streams(1),
            enable_stream_priorities(true),
            enable_kernel_fusion(true) {}
    };

    struct StreamMetrics {
        float utilization[10];              // Stream utilization rates
        float avg_latency;                  // Average operation latency
        size_t memory_transfer_rate;        // Memory bandwidth utilization
        int concurrent_operations;          // Overlapping operations
        
        StreamMetrics() :
            avg_latency(0.0f),
            memory_transfer_rate(0),
            concurrent_operations(0) {
            for (int i = 0; i < 10; ++i) {
                utilization[i] = 0.0f;
            }
        }
    };

    CUDAStreamManager();
    virtual ~CUDAStreamManager();

    // Stream management
    virtual void initialize_streams(const StreamConfig& config);
    virtual cudaStream_t get_compute_stream(int index = 0);
    virtual cudaStream_t get_memory_stream();

    // Overlap compute and memory operations
    virtual void enable_compute_memory_overlap();
    virtual void fuse_operations();

    // Performance monitoring
    virtual StreamMetrics get_stream_metrics() const;
    virtual float get_stream_utilization(cudaStream_t stream) const;
    virtual void optimize_stream_usage();

    // Cleanup
    virtual void cleanup();

private:
    static const int MAX_STREAMS = 10;
    cudaStream_t compute_streams_[MAX_STREAMS];
    cudaStream_t memory_stream_;
    StreamConfig config_;
    bool initialized_;
    int active_compute_streams_;

    // Internal helper methods
    virtual void setup_stream_priorities();
    virtual void configure_memory_overlap();
};

/// 🔥 OPTIMIZED: Single-GPU Training Manager
/// Coordinates all single-GPU optimizations
class SingleGPUManager {
public:
    struct GPUConfig {
        int gpu_id;                         // Single GPU device ID
        float memory_fraction;              // Memory allocation for single GPU
        bool enable_tensor_cores;           // Use Tensor Cores for acceleration
        bool enable_memory_optimization;    // Optimize memory layout
        bool enable_kernel_fusion;          // Fuse operations for efficiency
        
        GPUConfig() :
            gpu_id(0),
            memory_fraction(0.9f),
            enable_tensor_cores(true),
            enable_memory_optimization(true),
            enable_kernel_fusion(true) {}
    };

    struct TrainingMetrics {
        float gpu_utilization;              // Single GPU utilization percentage
        size_t memory_usage;                // Current memory usage
        float compute_efficiency;           // Compute utilization
        float memory_bandwidth_usage;       // Memory bandwidth utilization
        float training_throughput;          // Samples per second
        float kernel_efficiency;            // Kernel execution efficiency
        
        TrainingMetrics() :
            gpu_utilization(0.0f),
            memory_usage(0),
            compute_efficiency(0.0f),
            memory_bandwidth_usage(0.0f),
            training_throughput(0.0f),
            kernel_efficiency(0.0f) {}
    };

    SingleGPUManager();
    virtual ~SingleGPUManager();

    // Initialization
    virtual void initialize_single_gpu_training(const GPUConfig& config);
    virtual void setup_tensor_cores();
    virtual void optimize_memory_layout();
    virtual void fuse_kernels();

    // Performance monitoring
    virtual TrainingMetrics get_training_metrics() const;
    virtual void monitor_gpu_performance();
    virtual void adjust_resources_dynamically();

    // Utility methods
    virtual float get_current_gpu_utilization() const;
    virtual bool is_training_stable() const;

private:
    GPUConfig config_;
    GPUMemoryManager* memory_manager_;
    CUDAStreamManager* stream_manager_;
    bool initialized_;

    static const int MAX_HISTORY = 100;
    TrainingMetrics performance_history_[MAX_HISTORY];
    int history_size_;
    int max_history_size_;

    // Internal helper methods
    virtual void detect_gpu_capabilities();
    virtual void setup_hardware_optimizations();
    virtual void log_initialization_status();
};

} // namespace GGL