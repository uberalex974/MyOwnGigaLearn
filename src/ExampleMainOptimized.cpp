#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/OBSBuilders/DefaultObs.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
	// Enhanced reward system for better learning
	std::vector<WeightedReward> rewards = {

		// Movement rewards
		{ new AirReward(), 0.25f },

		// Player-ball interaction
		{ new FaceBallReward(), 0.25f },
		{ new VelocityPlayerToBallReward(), 4.f },
		{ new StrongTouchReward(20, 100), 60 },

		// Ball-goal rewards  
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },

		// Boost management
		{ new PickupBoostReward(), 10.f },
		{ new SaveBoostReward(), 0.2f },

		// Game events (enhanced weights)
		{ new ZeroSumReward(new BumpReward(), 0.5f), 25 },     // Increased from 20
		{ new ZeroSumReward(new DemoReward(), 0.5f), 100 },     // Increased from 80
		{ new GoalReward(), 200 }                              // Increased from 150
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Create arena
	int playersPerTeam = 1;
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObs();
	result.stateSetter = new KickoffState();
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;

	return result;
}

// Step callback with performance monitoring
void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	bool doExpensiveMetrics = (rand() % 4) == 0;

	// Enhanced metrics collection
	for (auto& state : states) {
		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				// Basic performance metrics
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);

				// Performance metrics
				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));
				report.AddAvg("Player/Boost", player.boost);

				// Quality metrics
				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

// Main function for deployment-optimized training
#ifndef RLBot_DEPLOYMENT
int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes
	RocketSim::Init("C:\\Giga\\GigaLearnCPP\\collision_meshes");

	// 🚀 DEPLOYMENT-OPTIMIZED CONFIGURATION
	LearnerConfig cfg = {};

	// Device and precision settings
	cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	cfg.ppo.useHalfPrecision = true;     // 50% VRAM reduction

	// Training parameters for deployment
	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1;
	cfg.numGames = 256;                   // Full training capacity
	cfg.randomSeed = 123;

	// Batch training parameters
	int tsPerItr = 50'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = 50'000;           // Full batch size for training
	cfg.ppo.miniBatchSize = 50'000;       // Full mini-batch size
	cfg.ppo.overbatching = true;

	// Training epochs
	cfg.ppo.epochs = 3;                   // Optimized for convergence

	// Optimized hyperparameters
	cfg.ppo.entropyScale = 0.025f;       // Improved stability
	cfg.ppo.gaeGamma = 0.99f;
	cfg.ppo.policyLR = 2.0e-4;           // Optimized for mixed precision
	cfg.ppo.criticLR = 2.0e-4;

	// Network architecture
	cfg.ppo.sharedHead.layerSizes = { 512, 256 };  // Wider shared layers
	cfg.ppo.policy.layerSizes = { 256, 128 };      // Policy head
	cfg.ppo.critic.layerSizes = { 256, 128 };      // Value head

	// Optimizer configuration (AdamW for better convergence)
	auto optim = ModelOptimType::ADAMW;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	// Activation function (LeakyReLU for better gradient flow)
	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	// Layer normalization
	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

	// Metrics and visualization
	cfg.sendMetrics = true;
	cfg.renderMode = false;

	// Save checkpoints to project root (preserved across rebuilds)
	cfg.checkpointFolder = "C:/Giga/GigaLearnCPP/checkpoints_deploy";

	// Create learner with configuration
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning
learner->Start();
	try {
	/learner->Start();
	} catch (const std::exception&) {
		// Basic error handling
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

// RLBot deployment mode with reduced settings for inference speed
#endif // RLBot_DEPLOYMENT

#ifdef RLBot_DEPLOYMENT
int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes
	RocketSim::Init("C:\\Giga\\GigaLearnCPP\\collision_meshes");

	// 🚀 RLBot-specific configuration - OPTIMIZED FOR INFERENCE
	LearnerConfig cfg = {};
	cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	cfg.ppo.useHalfPrecision = true;
	
	// Optimized for real-time inference
	cfg.numGames = 64;                    // Reduced for faster inference
	cfg.tickSkip = 4;                     // Faster response
	cfg.actionDelay = 2;
	cfg.ppo.tsPerItr = 10'000;            // Smaller training batches
	cfg.ppo.batchSize = 8'000;            // Optimized for inference
	cfg.ppo.miniBatchSize = 1'000;        // Individual inference optimization

	// Ultra-optimized network for RLBot
	cfg.ppo.sharedHead.layerSizes = { 256, 128 };
	cfg.ppo.policy.layerSizes = { 128, 64 };
	cfg.ppo.critic.layerSizes = { 128, 64 };
	cfg.ppo.policyLR = 1.0e-4;            // Conservative learning rate
	cfg.ppo.criticLR = 1.0e-4;

	// Save checkpoints to deployment folder
	cfg.checkpointFolder = "C:/Giga/GigaLearnCPP/checkpoints_deploy";

	// Create learner optimized for RLBot
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start RLBot-optimized training
	try {
		 learner->Start();
	} catch (const std::exception&) {
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
#endif