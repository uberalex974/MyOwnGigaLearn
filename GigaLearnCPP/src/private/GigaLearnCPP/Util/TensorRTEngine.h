#pragma once

// Stub implementation for TensorRTEngine.h with basic functionality
// TensorRT includes will be conditionally included based on availability

#ifdef WITH_TENSORRT
// Include TensorRT headers when available
#include <tensorrt_runtime_api.h>
#endif

namespace GGL {
    
    // Forward declarations
#ifdef WITH_TENSORRT
    namespace nvinfer1 {
        class IBuilder;
        class INetworkDefinition;
        class ICudaEngine;
        class IExecutionContext;
        class Logger;
    }
#endif
    
    class TensorRTEngine {
    private:
#ifdef WITH_TENSORRT
        void* builder_;           // nvinfer1::IBuilder*
        void* network_;           // nvinfer1::INetworkDefinition*
        void* engine_;            // nvinfer1::ICudaEngine*
        void* context_;           // nvinfer1::IExecutionContext*
        void* logger_;            // nvinfer1::Logger*
#endif
        char model_path_[256];
        bool is_initialized_;
        
    public:
        TensorRTEngine();
        ~TensorRTEngine();
        
        // Model loading and optimization
        bool Initialize(const char* model_path);
        bool OptimizeModel();
        bool LoadEngine(const char* engine_path);
        
        // Inference methods (stub implementations)
        bool RunInference(const float* input, float* output, int input_size, int output_size);
        
        // Model management
        void SetModelPath(const char* path);
        const char* GetModelPath() const;
        bool IsInitialized() const;
        
        // Memory management
        int GetEngineMemoryUsage() const;
        void ClearEngine();
    };
    
    class TensorRTManager {
    private:
        static const int MAX_ENGINES = 8;
        TensorRTEngine* engines_[MAX_ENGINES];
        int engine_count_;
        int max_engines_;
        
    public:
        TensorRTManager(int max_engines = 4);
        ~TensorRTManager();
        
        // Engine management
        TensorRTEngine* GetEngine(const char* model_id);
        bool RegisterEngine(const char* model_id, TensorRTEngine* engine);
        void RemoveEngine(const char* model_id);
        
        // Performance monitoring
        void PrintEngineStats();
        int GetTotalMemoryUsage() const;
    };
}