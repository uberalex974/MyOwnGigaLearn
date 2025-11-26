#include "GAE.h"
#include <GigaLearnCPP/FrameworkTorch.h>

void GGL::GAE::Compute(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange
) {
	// Fully vectorized per-episode GAE on the learner device (GPU when available).
	// This eliminates CPU round-trips during consumption and improves steps/sec.

	torch::Device device = valPreds.device();
	auto floatOpts = torch::TensorOptions().device(device).dtype(torch::kFloat32);
	int64_t numReturns = rews.size(0);

	// Move inputs once
	torch::Tensor rewards = rews.to(floatOpts);
	torch::Tensor termDev = terminals.to(torch::kByte).to(device, /*non_blocking=*/true);

	// 1) Reward normalization + clipping
	torch::Tensor clippedRews = rewards;
	if (returnStd != 0) {
		clippedRews = clippedRews / returnStd;
		if (clipRange > 0)
			clippedRews = torch::clamp(clippedRews, -clipRange, clipRange);
	}
	{
		float totalRew = rewards.abs().sum().item<float>();
		float totalClippedRew = clippedRews.abs().sum().item<float>();
		outRewClipPortion = (totalRew - totalClippedRew) / std::max(totalRew, 1e-7f);
	}

	// 2) Next values (bootstrapped for truncations)
	torch::Tensor nextVals = torch::zeros_like(valPreds);
	if (numReturns > 1)
		nextVals.slice(0, 0, numReturns - 1).copy_(valPreds.slice(0, 1, numReturns));

	// Zero out real terminals
	torch::Tensor doneMask = (termDev == RLGC::TerminalType::NORMAL);
	nextVals.index_put_({doneMask}, 0);

	// Fill truncations with their predicted continuation value (one-to-one order)
	auto truncIdx = (termDev == RLGC::TerminalType::TRUNCATED).nonzero().flatten();
	if (truncValPreds.defined() && truncIdx.numel() > 0) {
		auto truncVals = truncValPreds.to(valPreds.options());
		nextVals.index_put_({truncIdx}, truncVals);
	}

	// 3) Delta on device
	torch::Tensor delta = clippedRews + (gamma * nextVals) - valPreds;

	// 4) Advantage scan per episode (device, vectorized)
	torch::Tensor advantages = torch::zeros_like(delta);
	auto contMask = (termDev == RLGC::TerminalType::NOT_TERMINAL).to(floatOpts);

	auto terminalIdx = (contMask == 0).nonzero().flatten();
	auto termCPU = terminalIdx.cpu();

	auto computeSegment = [&](int64_t start, int64_t end) {
		auto segDelta = delta.slice(0, start, end);
		auto segCont = contMask.slice(0, start, end);

		auto revDelta = segDelta.flip(0);
		auto revG = (segCont * (gamma * lambda)).flip(0);

		// Keep epsilon to avoid zero products while still killing discounted carry past terminals
		auto safeG = torch::clamp_min(revG, 1e-8f);
		auto running = torch::cumprod(torch::cat({torch::ones({1}, floatOpts), safeG}, 0), 0);

		auto scaled = revDelta / running.slice(0, 1);
		auto prefix = torch::cumsum(scaled, 0);
		auto advRev = prefix * running.slice(0, 1);

		auto advSeg = advRev.flip(0);
		advantages.slice(0, start, end).copy_(advSeg);
	};

	int64_t prev = 0;
	for (int64_t i = 0; i < termCPU.numel(); i++) {
		int64_t end = termCPU[i].item<int64_t>() + 1;
		computeSegment(prev, end);
		prev = end;
	}
	if (prev < numReturns)
		computeSegment(prev, numReturns);

	// 5) Outputs remain on device for PPO
	outAdvantages = advantages;
	outTargetValues = valPreds + advantages;
	outReturns = outTargetValues;
}
