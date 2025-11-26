import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Updated {path}")

def optimize_learner_cpp():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\public\GigaLearnCPP\Learner.cpp"
    content = read_file(path)
    
    # Pre-allocate combinedTraj
    if "combinedTraj.states.reserve(config.ppo.tsPerItr);" not in content:
        print("Optimizing Learner.cpp: Pre-allocating combinedTraj vectors")
        # Find where combinedTraj is declared
        # auto combinedTraj = Trajectory();
        
        # We need to add reserve calls for all its members
        reserve_block = """
				// Only contains complete episodes
				auto combinedTraj = Trajectory();
				// Optimization: Pre-allocate memory to avoid reallocations during collection
				combinedTraj.states.reserve(config.ppo.tsPerItr * obsSize);
				combinedTraj.actions.reserve(config.ppo.tsPerItr);
				combinedTraj.logProbs.reserve(config.ppo.tsPerItr);
				combinedTraj.rewards.reserve(config.ppo.tsPerItr);
				combinedTraj.terminals.reserve(config.ppo.tsPerItr);
				combinedTraj.actionMasks.reserve(config.ppo.tsPerItr * numActions);
"""
        content = content.replace(
            "// Only contains complete episodes\n\t\t\t\tauto combinedTraj = Trajectory();",
            reserve_block
        )
        
    # Pre-allocate player trajectories
    if "trajectories[i].states.reserve(maxEpisodeLength * obsSize);" not in content:
        print("Optimizing Learner.cpp: Pre-allocating player trajectories")
        # auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
        # int maxEpisodeLength = ...
        
        # We want to iterate and reserve
        reserve_loop = """
		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));
		
		// Optimization: Pre-allocate player trajectories
		for (int i = 0; i < numPlayers; i++) {
			trajectories[i].states.reserve(maxEpisodeLength * obsSize);
			trajectories[i].actions.reserve(maxEpisodeLength);
			trajectories[i].logProbs.reserve(maxEpisodeLength);
			trajectories[i].rewards.reserve(maxEpisodeLength);
			trajectories[i].terminals.reserve(maxEpisodeLength);
			trajectories[i].actionMasks.reserve(maxEpisodeLength * numActions);
		}
"""
        content = content.replace(
            "auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});\n\t\tint maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));",
            reserve_loop
        )

    write_file(path, content)

def optimize_gae_cpp():
    path = r"c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\GAE.cpp"
    content = read_file(path)
    
    # Parallelize Reward Processing Loop
    if "#pragma omp parallel for" not in content:
        print("Optimizing GAE.cpp: Parallelizing loops")
        
        # Reward loop
        content = content.replace(
            "for (int i = 0; i < numReturns; i++) {",
            "#pragma omp parallel for\n\tfor (int i = 0; i < numReturns; i++) {"
        )
        
        # NextVals loop (careful with truncIdx)
        # truncIdx is incremented inside the loop. This makes it non-parallelizable easily without atomic or prefix sum.
        # "nextVals[i] = truncValPreds[truncIdx++].item<float>();"
        # So we SKIP parallelizing the nextVals loop unless we handle truncIdx.
        # Given it's a simple loop, maybe not worth the complexity.
        # But the reward loop is safe.
        
    write_file(path, content)

def main():
    print("Applying Iteration Speed Optimizations...")
    optimize_learner_cpp()
    optimize_gae_cpp()
    print("Done.")

if __name__ == "__main__":
    main()
