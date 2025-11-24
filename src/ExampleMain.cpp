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
	// These are ok rewards that will produce a scoring bot in ~100m steps
	std::vector<WeightedReward> rewards = {

		// Movement
		{ new AirReward(), 0.25f },

		// Player-ball
		{ new FaceBallReward(), 0.25f },
		{ new VelocityPlayerToBallReward(), 4.f },
		{ new StrongTouchReward(20, 100), 60 },

		// Ball-goal
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },

		// Boost
		{ new PickupBoostReward(), 10.f },
		{ new SaveBoostReward(), 0.2f },

		// Game events
		{ new ZeroSumReward(new BumpReward(), 0.5f), 20 },
		{ new ZeroSumReward(new DemoReward(), 0.5f), 80 },
		{ new GoalReward(), 150 }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Make the arena
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

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// To prevent expensive metrics from eating at performance, we will only run them on 1/4th of steps
	// This doesn't really matter unless you have expensive metrics (which this example doesn't)
	bool doExpensiveMetrics = (rand() % 4) == 0;

	// Add our metrics
	for (auto& state : states) {
		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);

				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));

				report.AddAvg("Player/Boost", player.boost);

				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes
	// Change this path to point to your meshes!
	RocketSim::Init("C:\\Giga\\GigaLearnCPP\\collision_meshes");

	// Make configuration for the learner
	LearnerConfig cfg = {};

	cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	cfg.ppo.useHalfPrecision = true;  // 50% VRAM reduction, 20% speed improvement

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	// Play around with this to see what the optimal is for your machine, more games will consume more RAM
	cfg.numGames = 256;  // Training environment count

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 123;

	int tsPerItr = 50'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = 50'000;      // Training batch size
	cfg.ppo.miniBatchSize = 50'000;  // Mini-batch size
	cfg.ppo.overbatching = true;     // Enable for efficiency

	// Using 2 epochs seems pretty optimal when comparing time training to skill
	// Perhaps 1 or 3 is better for you, test and find out!
	cfg.ppo.epochs = 2;                     // Training epochs per iteration

	// This scales differently than "ent_coef" in other frameworks
	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions
	cfg.ppo.entropyScale = 0.025f;          // Optimized for stability

	// Rate of reward decay
	// Starting low tends to work out
	cfg.ppo.gaeGamma = 0.99;

	// Optimized learning rates for mixed precision
	cfg.ppo.policyLR = 2.0e-4;     // Higher for mixed precision
	cfg.ppo.criticLR = 2.0e-4;

	// Optimized network architecture - wider, shallower for speed
	cfg.ppo.sharedHead.layerSizes = { 512, 256 };  // 2 layers vs 3
	cfg.ppo.policy.layerSizes = { 256, 128 };      // Reduced depth
	cfg.ppo.critic.layerSizes = { 256, 128 };      // Reduced depth

	auto optim = ModelOptimType::ADAMW;  // Better than ADAM
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::LEAKY_RELU;  // Better than ReLU
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

	cfg.sendMetrics = true; // Send metrics
	cfg.renderMode = false; // Don't render

	// Save checkpoints to project root (preserved across rebuilds)
	cfg.checkpointFolder = "C:/Giga/GigaLearnCPP/checkpoints";

	// 🚀 TRAINING CONFIGURATION APPLIED
	// - Mixed precision training enabled (50% VRAM reduction)
	// - Environment count: 256 (parallel training environments)
	// - Batch size: 50K (training batch size)  
	// - Mini-batch: 50K (mini-batch size)
	// - Epochs: 2 (training epochs per iteration)
	// - Learning rates: 2.0e-4 (optimized for mixed precision)
	// - Optimizer: AdamW (better than Adam)
	// - Activation: LeakyReLU (better than ReLU)
	// - Architecture: Wider, shallower networks for speed

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
