#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>
#include <GigaLearnCPP/Util/EnhancedInferenceManager.h>
#include <GigaLearnCPP/Util/CUDAOptimizations.h>
#include <GigaLearnCPP/Util/TensorRTEngine.h>
#include <GigaLearnCPP/Util/EnhancedArchitectures.h>
#include <GigaLearnCPP/Util/GPUMemoryManager.h>

#include "../Util/Models.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>

#include "ExperienceBuffer.h"

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class PPOLearner {
	public:
		ModelSet models = {};
		ModelSet guidingPolicyModels = {};

		PPOLearnerConfig config;
		torch::Device device;

		// 🔥 OPTIMIZED: Gradient accumulation for larger effective batches
		int gradient_accumulation_steps = 4;  // Accumulate 4 mini-batches
		int current_accumulation_step = 0;
		bool gradient_accumulation_enabled = true;
		
		// 🚀 ADVANCED OPTIMIZATIONS: Enhanced inference and training acceleration
		std::unique_ptr<EnhancedInferenceManager> inference_manager_;
		std::unique_ptr<TrainingAccelerationManager> training_accelerator_;
		std::unique_ptr<CUDAStreamManager> cuda_stream_manager_;
		std::unique_ptr<GPUMemoryManager> gpu_memory_manager_;
		
		// 🎯 PERFORMANCE TARGETS: <1ms inference, 90%+ GPU utilization
		bool use_tensorrt_inference = true;
		bool use_cuda_kernels = true;
		bool use_attention_mechanisms = true;
		bool use_progressive_training = false;
		float target_inference_latency_us = 1000.0f;  // 1ms target
		float target_gpu_utilization = 0.90f;         // 90% target
		
		// 📊 ADVANCED METRICS: Real-time performance monitoring
		struct AdvancedMetrics {
			float avg_inference_latency_us = 0.0f;
			float p95_inference_latency_us = 0.0f;
			float gpu_utilization_percent = 0.0f;
			float memory_efficiency = 0.0f;
			float training_throughput = 0.0f;
			int total_inferences = 0;
			float cache_hit_rate = 0.0f;
			float tensor_core_speedup = 0.0f;
		} advanced_metrics_;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device
		);

		static void MakeModels(
			bool makeCritic, 
			int obsSize, int numActions, 
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels
		);
		
		// If models is null, this->models will be used
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL);
		torch::Tensor InferCritic(torch::Tensor obs);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			float temperature,
			bool halfPrec
		);
		static void InferActionsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs
		);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void TransferLearn(
			ModelSet& oldModels, 
			torch::Tensor newObs, torch::Tensor oldObs, 
			torch::Tensor newActionMasks, torch::Tensor oldActionMasks, 
			torch::Tensor actionMaps,
			Report& report, 
			const TransferLearnConfig& transferLearnConfig
		);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		void SetLearningRates(float policyLR, float criticLR);

		ModelSet GetPolicyModels();
		
		// 🚀 ADVANCED OPTIMIZATION METHODS
		
		// Initialize advanced optimizations
		bool InitializeAdvancedOptimizations(
			bool use_tensorrt = true,
			bool use_cuda_kernels = true,
			bool use_attention = true,
			float target_latency_us = 1000.0f
		);
		
		// Enhanced inference with TensorRT optimization
		void InferActionsOptimized(
			torch::Tensor obs, 
			torch::Tensor actionMasks, 
			torch::Tensor* outActions, 
			torch::Tensor* outLogProbs,
			bool use_tensorrt = true,
			bool deterministic = false
		);
		
		// Enhanced critic inference with spatial attention
		torch::Tensor InferCriticOptimized(torch::Tensor obs, bool use_tensorrt = true);
		
		// CUDA-accelerated advantage computation
		void ComputeAdvantagesCUDA(
			torch::Tensor& advantages,
			const torch::Tensor& rewards,
			const torch::Tensor& values,
			const torch::Tensor& next_values,
			const torch::Tensor& dones,
			float gamma,
			float lambda,
			cudaStream_t stream = 0
		);
		
		// Memory-coalesced tensor operations
		torch::Tensor CoalescedTensorOperation(
			const torch::Tensor& input,
			const std::string& operation = "normalize"
		);
		
		// Progressive training integration
		void EnableProgressiveTraining(
			const std::vector<int>& difficulty_levels = {1, 2, 3, 4, 5}
		);
		
		// Auto-optimization for target performance
		bool OptimizeForTargetLatency(float target_latency_us);
		bool OptimizeForTargetGPUUtilization(float target_utilization);
		
		// Real-time performance monitoring
		const AdvancedMetrics& GetAdvancedMetrics() const { return advanced_metrics_; }
		void UpdateAdvancedMetrics(float latency_us, float gpu_util, float memory_eff);
		void PrintAdvancedPerformanceReport();
		
		// Configuration control
		void SetOptimizationTargets(float target_latency_us, float target_gpu_util);
		void EnableModelCaching(bool enable, size_t max_cache_size = 1000);
		void ConfigureCUDAStreams(int num_streams = 8);
		
		// Cleanup
		void CleanupOptimizations();
		void ResetPerformanceCounters();
	};
}