#include "ExperienceBuffer.h"
#include <torch/cuda.h>

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(torch::Tensor indices) const {

	ExperienceTensors result;

	// Ensure indices live on the same device as the source tensor we are slicing.
	auto fnSlice = [&](const torch::Tensor& t) -> torch::Tensor {
		if (!t.defined())
			return t;

		torch::Tensor idx = indices;
		if (idx.device() != t.device())
			idx = idx.to(t.device(), /*non_blocking=*/true);

		return torch::index_select(t, 0, idx);
	};

	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		*toItr = fnSlice(*fromItr);

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	size_t expSize = data.states.size(0);

	// Work on the native device of the stored experience to avoid cross-device copies
	torch::Device dataDevice = data.states.defined() ? data.states.device() : torch::kCPU;
	static bool loggedDevice = false;
	if (!loggedDevice) {
		RG_LOG("DEBUG: ExperienceBuffer data device: " << dataDevice);
		loggedDevice = true;
	}

	// Optimization: Create indices on CPU once
	auto opts = torch::TensorOptions().dtype(torch::kLong).device(dataDevice);
	torch::Tensor allIndices = torch::randperm(expSize, opts);
	if (allIndices.device() != dataDevice)
		allIndices = allIndices.to(dataDevice, /*non_blocking=*/true);

	// Get a sample set from each of the batches
	std::vector<ExperienceTensors> result;
	
	// Check if we should use pinned memory (if we are on CPU but have a CUDA device available)
	bool usePinnedMemory = (device.is_cuda() && torch::cuda::is_available() && !dataDevice.is_cuda());

	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {

		int curBatchSize = batchSize;
		if (startIdx + batchSize * 2 > expSize) {
			// Last batch of the iteration
			if (overbatching) {
				// Extend batch size to the end of the experience
				curBatchSize = expSize - startIdx;
			}
		}

		// Slice indices
		
		torch::Tensor batchIndices = allIndices.slice(0, startIdx, startIdx + curBatchSize);
		if (batchIndices.device() != dataDevice)
			batchIndices = batchIndices.to(dataDevice, /*non_blocking=*/true);
		
		auto batch = _GetSamples(batchIndices);
		
		if (usePinnedMemory) {
			// Pin memory for faster CPU -> GPU transfer (only if tensor is on CPU)
			auto pinIfCPU = [](torch::Tensor t) {
				return (t.defined() && t.device().is_cpu()) ? t.pin_memory() : t;
			};

			batch.actions = pinIfCPU(batch.actions);
			batch.logProbs = pinIfCPU(batch.logProbs);
			batch.states = pinIfCPU(batch.states);
			batch.actionMasks = pinIfCPU(batch.actionMasks);
			batch.targetValues = pinIfCPU(batch.targetValues);
			batch.advantages = pinIfCPU(batch.advantages);
			batch.vals = pinIfCPU(batch.vals);
		}

		result.push_back(batch);
	}

	return result;
}
