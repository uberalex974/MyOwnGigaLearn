 # GigaLearnCPP Reward System - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [File Organization and Compilation](#file-organization-and-compilation)
4. [Built-in Rewards Reference](#built-in-rewards-reference)
5. [Reward Implementation Deep Dive](#reward-implementation-deep-dive)
6. [Custom Reward Development](#custom-reward-development)
7. [Integration and Build System](#integration-and-build-system)
8. [Advanced Techniques](#advanced-techniques)
9. [Performance Optimization](#performance-optimization)
10. [Common Issues and Solutions](#common-issues-and-solutions)
11. [Configuration Examples](#configuration-examples)
12. [Mathematical Reference](#mathematical-reference)

---

## Overview

The GigaLearnCPP reward system is a sophisticated, enterprise-grade framework designed for training competitive Rocket League bots through reinforcement learning. It provides a comprehensive architecture for defining, combining, and optimizing reward functions that guide agent behavior toward desired objectives.

### Core Philosophy
The reward system follows a **polymorphic, header-first design** that enables:
- **Zero-compilation-overhead** through template metaprogramming
- **Automatic build system integration** via glob-based file discovery
- **Compile-time optimizations** for event-driven rewards
- **Runtime flexibility** through polymorphic interfaces

### Key Features
- **🎯 Modular Architecture**: Polymorphic base class with automatic extension
- **⚡ Event-Driven System**: 9 comprehensive player event tracking types
- **⚖️ Team Balance**: Sophisticated zero-sum reward wrapper
- **🚀 Performance Optimized**: Vectorized computation, caching, and template magic
- **📚 Rich Library**: 20+ production-tested reward functions
- **🔧 Build System Integration**: Automatic compilation without CMake modifications

---

## System Architecture

### Core Component Hierarchy

```
Reward (Abstract Base Class)
├── CommonRewards.h (20+ Built-in Rewards)
├── RewardWrapper.h (Wrapper Pattern)
├── ZeroSumReward.cpp/h (Team Balance)
└── PlayerReward.h (Template-Based Per-Player Rewards)
```

#### 1. Base Reward Interface (`Reward.h`)

The foundation of the reward system, implementing a clean polymorphic interface:

```cpp
namespace RLGC {
    class Reward {
    private:
        std::string _cachedName = {};  // Performance optimization
    
    public:
        // Lifecycle Management
        virtual void Reset(const GameState& initialState) {}
        virtual void PreStep(const GameState& state) {}
        
        // Core Reward Computation  
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal);
        virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal);
        
        // Identity and Debugging
        virtual std::string GetName();
        virtual ~Reward() {}
    };
}
```

**Key Design Principles:**
- **Lazy Name Caching**: Avoids repeated `typeid` lookups for performance
- **Virtual Interface**: Enables runtime polymorphism and wrapper patterns
- **Batch Processing**: `GetAllRewards()` provides vectorized computation
- **Optional Lifecycle**: `Reset()` and `PreStep()` are optional hooks

#### 2. Weighted Reward Composition (`Reward.h`)

The fundamental building block for combining multiple rewards:

```cpp
struct WeightedReward {
    Reward* reward;    // Owned pointer - managed by RAII
    float weight;      // Scaling factor for this reward component
    
    // Multiple constructor overloads for convenience
    WeightedReward(Reward* reward, float scale);
    WeightedReward(Reward* reward, int scale); // Implicit conversion
};
```

**Memory Management Strategy:**
- **Automatic Ownership**: `WeightedReward` does NOT take ownership
- **RAII Compliance**: Higher-level containers manage memory lifecycle
- **Zero-Copy Design**: Pointers are passed around without ownership transfer

#### 3. Template-Based Event Rewards (Compile-Time Optimization)

The system uses sophisticated template metaprogramming for zero-overhead event handling:

```cpp
template<bool PlayerEventState::* VAR, bool NEGATIVE>
class PlayerDataEventReward : public Reward {
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
        bool val = player.eventState.*VAR;
        return NEGATIVE ? -(float)val : (float)val;
    }
};
```

**Performance Benefits:**
- **Zero Runtime Overhead**: Event checking resolved at compile time
- **Code Generation**: Each event type gets optimized assembly
- **Type Safety**: Compile-time validation of event state access

#### 4. Event Tracking System

Comprehensive player event detection through `PlayerEventState`:

```cpp
struct PlayerEventState {
    bool goal, save, assist, shot, shotPass, 
         bump, bumped, demo, demoed;  // 9 event types
    
    PlayerEventState() {
        memset(this, 0, sizeof(*this));  // Zero initialization
    }
};
```

**Event Detection Pipeline:**
1. **Physics Engine**: Bullet3 collision detection triggers events
2. **GameEventTracker**: Processes and validates events
3. **Callback Functions**: Set player state in `EnvSet.cpp`
4. **Reward Access**: Available immediately in next reward computation

#### 5. Reward Pipeline Integration

The reward computation happens in a carefully orchestrated pipeline (`EnvSet.cpp` lines 190-239):

```cpp
// Step 1: Pre-processing 
for (auto& weighted : rewards[arenaIdx])
    weighted.reward->PreStep(gs);

// Step 2: Batch Reward Computation
FList allRewards = FList(gs.players.size(), 0);
for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
    auto& weightedReward = rewards[arenaIdx][rewardIdx];
    FList output = weightedReward.reward->GetAllRewards(gs, terminalType);
    
    // Vectorized weighted combination
    for (int i = 0; i < gs.players.size(); i++)
        allRewards[i] += output[i] * weightedReward.weight;
}

// Step 3: Final assignment to training state
for (int i = 0; i < gs.players.size(); i++)
    state.rewards[playerStartIdx + i] = allRewards[i];
```

**Performance Optimizations:**
- **Batch Processing**: Vectorized computation for all players
- **Memory Locality**: Contiguous memory layouts for cache efficiency
- **Minimal Synchronization**: Lock-free multi-environment execution

---

## Built-in Rewards

### Event-Based Rewards
| Reward | Purpose | Weight Range | Zero-Sum |
|--------|---------|-------------|----------|
| `PlayerGoalReward` | Individual goal scored | 150+ | Optional |
| `AssistReward` | Assist provided | 20-50 | Optional |
| `ShotReward` | Shot attempted | 10-30 | Optional |
| `SaveReward` | Save made | 25-75 | Optional |
| `BumpReward` | Opponent bumped | 20-40 | Recommended |
| `DemoReward` | Demolition performed | 80-120 | Recommended |
| `BumpedPenalty` | Was bumped (negative) | -10 to -30 | Recommended |
| `DemoedPenalty` | Was demoed (negative) | -50 to -100 | Recommended |

### Movement & Positioning Rewards
| Reward | Formula | Typical Weight | Notes |
|--------|---------|---------------|-------|
| `AirReward` | `!player.isOnGround` | 0.25 | Binary reward for aerial play |
| `VelocityReward` | `vel.Length() / CAR_MAX_SPEED` | 1-5 | Normalized velocity magnitude |
| `SpeedReward` | `vel.Length() / CAR_MAX_SPEED` | 1-3 | Simplified velocity reward |
| `FaceBallReward` | `forward.Dot(dirToBall)` | 0.25 | Alignment with ball direction |

### Player-Ball Interaction Rewards
| Reward | Formula | Typical Weight | Special Features |
|--------|---------|---------------|------------------|
| `VelocityPlayerToBallReward` | `dirToBall.Dot(normVel)` | 4-6 | Movement toward ball |
| `TouchBallReward` | `player.ballTouchedStep` | 10-25 | Binary touch reward |
| `StrongTouchReward(20,100)` | `min(1, hitForce/maxVel)` | 40-80 | Configurable speed range |
| `TouchAccelReward` | Speed increment reward | 30-60 | Ball acceleration focus |

### Ball-Goal Interaction Rewards
| Reward | Formula | Typical Weight | Team Balance |
|--------|---------|---------------|--------------|
| `VelocityBallToGoalReward` | `goalDir.Dot(ballVel/maxSpeed)` | 2-4 | Often zero-sum |
| `GoalReward` | Team goal scored | 150-300 | Zero-sum by default |

### Boost Management Rewards
| Reward | Formula | Typical Weight | Purpose |
|--------|---------|---------------|---------|
| `PickupBoostReward` | Boost increment reward | 8-15 | Encourage boost collection |
| `SaveBoostReward(0.5)` | `boost^exponent` | 0.2-0.5 | Encourage boost conservation |

---

## Reward Implementation Details

### Clipping and Normalization

The system uses standard clipping functions defined in `Framework.h`:
```cpp
#define RS_MAX(a, b) ((a > b) ? a : b)
#define RS_MIN(a, b) ((a < b) ? a : b)
#define RS_CLAMP(val, min, max) RS_MIN(RS_MAX(val, min), max)
```

**Common Normalization Patterns:**
```cpp
// Speed normalization
float speed = player.vel.Length() / CommonValues::CAR_MAX_SPEED;

// Distance normalization
float distNormalized = RS_CLAMP(distance / maxDistance, 0.0f, 1.0f);

// Boost level normalization
float boostNormalized = RS_CLAMP(player.boost / 100.0f, 0.0f, 1.0f);

// Position-based rewards
float posReward = RS_CLAMP((targetPos - currentPos).Length() / range, 0.0f, 1.0f);
```

### Reward Calculation Examples

**Strong Touch Reward Implementation:**
```cpp
virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
    if (!state.prev) return 0;

    if (player.ballTouchedStep) {
        float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
        if (hitForce < minRewardedVel)
            return 0;

        return RS_MIN(1, hitForce / maxRewardedVel);
    }
    return 0;
}
```

**Face Ball Reward Implementation:**
```cpp
virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
    Vec dirToBall = (state.ball.pos - player.pos).Normalized();
    return player.rotMat.forward.Dot(dirToBall); // Already in [-1, 1] range
}
```

---

## Custom Reward Development

### Basic Custom Reward Template

```cpp
// CustomRewards.h
#pragma once
#include "Reward.h"
#include "../Math.h"

namespace RLGC {
    class DistanceToBallReward : public Reward {
    public:
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            // Calculate distance to ball
            Vec distance = state.ball.pos - player.pos;
            float dist = distance.Length();
            
            // Normalize distance (closer = higher reward)
            float maxDist = 3000.0f; // Maximum relevant distance
            float normalizedDist = RS_CLAMP(1.0f - (dist / maxDist), 0.0f, 1.0f);
            
            return normalizedDist;
        }
    };

    class BoostManagementReward : public Reward {
    public:
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            // Reward having boost but not wasting it
            if (player.boost > 80.0f) {
                return 1.0f; // Good boost level
            } else if (player.boost < 20.0f) {
                return -1.0f; // Low boost penalty
            } else {
                return 0.0f; // Neutral boost level
            }
        }
    };
}
```

### Advanced Custom Reward Examples

#### Position Control Reward
```cpp
class PositionControlReward : public Reward {
private:
    Vec targetPosition;
    float maxDistance;

public:
    PositionControlReward(Vec target, float maxDist) 
        : targetPosition(target), maxDistance(maxDist) {}

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        float distance = (player.pos - targetPosition).Length();
        float distanceReward = RS_CLAMP(1.0f - (distance / maxDistance), 0.0f, 1.0f);
        
        // Add velocity alignment bonus
        Vec toTarget = (targetPosition - player.pos).Normalized();
        float velocityAlignment = RS_MAX(0, player.vel.Dot(toTarget));
        
        return distanceReward + 0.5f * velocityAlignment;
    }
};
```

#### Momentum Conservation Reward
```cpp
class MomentumConservationReward : public Reward {
public:
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!state.prev || !player.ballTouchedStep)
            return 0;

        // Calculate momentum conservation
        Vec ballMomentum = state.ball.vel * CommonValues::BALL_RADIUS;
        Vec carMomentum = player.vel * 1000; // Approximate car mass
        
        // Reward when ball and car momentum are aligned
        float momentumAlignment = ballMomentum.Normalized().Dot(carMomentum.Normalized());
        
        // Reward when total momentum is reasonable (not wasteful)
        float totalMomentum = ballMomentum.Length() + carMomentum.Length();
        float momentumEfficiency = RS_CLAMP(3000.0f / totalMomentum, 0.0f, 1.0f);
        
        return momentumAlignment * momentumEfficiency;
    }
};
```

### Using Custom Rewards in Training

```cpp
// In EnvCreateFunc
#include "CustomRewards.h"

EnvCreateResult EnvCreateFunc(int index) {
    std::vector<WeightedReward> rewards = {
        // Custom rewards
        { new DistanceToBallReward(), 2.0f },
        { new BoostManagementReward(), 1.5f },
        { new PositionControlReward(CommonValues::BLUE_GOAL_CENTER, 2000.0f), 1.0f },
        
        // Built-in rewards
        { new AirReward(), 0.25f },
        { new FaceBallReward(), 0.25f },
        { new VelocityPlayerToBallReward(), 4.0f },
        { new StrongTouchReward(20, 100), 60.0f },
        { new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
        { new PickupBoostReward(), 10.0f },
        { new SaveBoostReward(), 0.2f },
        { new ZeroSumReward(new BumpReward(), 0.5f), 20.0f },
        { new ZeroSumReward(new DemoReward(), 0.5f), 80.0f },
        { new GoalReward(), 150.0f }
    };

    // ... rest of environment setup
}
```

---

## Best Practices

### 1. Reward Value Ranges
- **Keep rewards between -1 and 1** for stability
- **Use smaller weights** (0.1-1.0) for subtle behaviors
- **Use larger weights** (50-200) for critical events like goals

### 2. Normalization Guidelines
```cpp
// Good: Properly normalized
float reward = RS_CLAMP(distance / maxDistance, 0.0f, 1.0f);

// Good: Speed-based normalization  
float reward = velocity.Length() / CAR_MAX_SPEED;

// Avoid: Unbounded rewards
float reward = someComplexCalculation(); // Could be very large

// Instead: Clamp or normalize
float reward = RS_CLAMP(someComplexCalculation(), -1.0f, 1.0f);
```

### 3. Performance Optimization
```cpp
// Early returns for efficiency
virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
    if (!player.ballTouchedStep) return 0; // Cheap check first
    
    // Expensive calculations only when needed
    Vec complexCalculation = expensiveOperation(player, state);
    return complexCalculation.Length() / normalizationFactor;
}
```

### 4. State Management
```cpp
class StatefulReward : public Reward {
private:
    float previousValue = 0;
    int stepCounter = 0;

public:
    virtual void Reset(const GameState& initialState) override {
        previousValue = 0;
        stepCounter = 0;
    }

    virtual void PreStep(const GameState& state) override {
        stepCounter++;
        // Prepare for reward calculation
    }

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        float currentValue = calculateValue(player, state);
        float deltaReward = currentValue - previousValue;
        previousValue = currentValue;
        return deltaReward;
    }
};
```

### 5. Debugging and Logging
```cpp
virtual std::string GetName() override {
    return "MyCustomReward"; // Use descriptive names
}

// In GetReward, consider adding debug output for development
virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
    float reward = calculateReward(player, state);
    
    // Debug output (remove in production)
    #ifdef DEBUG
    if (reward != 0) {
        std::cout << GetName() << ": " << reward << std::endl;
    }
    #endif
    
    return reward;
}
```

---

## Performance Considerations

### 1. Memory Management
- **Use stack allocation** for temporary calculations
- **Avoid dynamic allocation** in GetReward() calls
- **Reuse vectors and objects** across calculations

### 2. Computational Efficiency
```cpp
// Good: Pre-calculate expensive values
class EfficientReward : public Reward {
private:
    Vec goalPosition;
    float maxDistanceSquared;

public:
    EfficientReward() {
        goalPosition = CommonValues::BLUE_GOAL_CENTER;
        maxDistanceSquared = 2000.0f * 2000.0f;
    }

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        // Use squared distance to avoid sqrt
        float distSq = (player.pos - goalPosition).LengthSquared();
        if (distSq > maxDistanceSquared) return 0;
        
        return 1.0f - (sqrt(distSq) / 2000.0f);
    }
};
```

### 3. Vectorization Opportunities
- Use `GetAllRewards()` for batch processing
- Structure data for SIMD operations
- Minimize branching in reward calculations

### 4. Caching Strategies
```cpp
class CachedReward : public Reward {
private:
    std::unordered_map<uint64_t, float> cache;
    size_t maxCacheSize = 1000;

public:
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        uint64_t hash = calculateStateHash(player, state);
        
        auto it = cache.find(hash);
        if (it != cache.end()) {
            return it->second; // Cache hit
        }
        
        float reward = calculateExpensiveReward(player, state);
        
        // Cache management
        if (cache.size() >= maxCacheSize) {
            cache.clear(); // Simple cache eviction
        }
        cache[hash] = reward;
        
        return reward;
    }
};
```

---

## Common Issues and Solutions

### Issue 1: PlayerReward.h Compilation Errors

**Problem:** 
```cpp
// Lines with errors:
for (int i = 0; i < initialState.players.size())  // Missing )
for (auto inst : instances)                        // Wrong variable name
```

**Solution:**
```cpp
// Fixed version:
for (int i = 0; i < initialState.players.size(); i++)  // Add ;
for (auto inst : _instances)                            // Use _instances
```

### Issue 2: Reward Explosion

**Problem:** Unbounded rewards causing training instability

**Solution:**
```cpp
// Apply proper clipping
float reward = calculateReward();
reward = RS_CLAMP(reward, -1.0f, 1.0f);

// Or normalize to reasonable range
reward = tanh(reward / scaleFactor);
```

### Issue 3: Zero-Sum Configuration

**Problem:** Understanding when and how to apply zero-sum rewards

**Solution:**
```cpp
// Use zero-sum for competitive aspects
{ new ZeroSumReward(new BumpReward(), 0.5f), 20.0f },

// Don't use zero-sum for individual skills
{ new AirReward(), 0.25f }, // Individual skill

// Use zero-sum for team-relevant events
{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
```

### Issue 4: Reward Weight Tuning

**Problem:** Unclear how to set appropriate weights

**Solution:**
```cpp
// Start with small weights and adjust
{ new MyReward(), 0.1f },   // Small weight for experimentation

// Use relative scaling
{ new GoalReward(), 100.0f },      // Major events
{ new TouchBallReward(), 10.0f },  // Moderate events  
{ new AirReward(), 0.25f },        // Minor behaviors

// Test weights in isolation
// Comment out other rewards temporarily to test individual components
```

---

## Configuration Examples

### 1. Balanced Scoring Bot
```cpp
std::vector<WeightedReward> balancedRewards = {
    // Movement fundamentals
    { new AirReward(), 0.25f },
    { new FaceBallReward(), 0.25f },
    
    // Ball control
    { new VelocityPlayerToBallReward(), 4.0f },
    { new StrongTouchReward(20, 100), 60.0f },
    
    // Team objectives
    { new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
    { new GoalReward(), 150.0f },
    
    // Resource management
    { new PickupBoostReward(), 10.0f },
    { new SaveBoostReward(), 0.2f },
    
    // Aggressive plays
    { new ZeroSumReward(new BumpReward(), 0.5f), 20.0f },
    { new ZeroSumReward(new DemoReward(), 0.5f), 80.0f }
};
```

### 2. Defensive Specialist
```cpp
std::vector<WeightedReward> defensiveRewards = {
    // Positioning
    { new FaceBallReward(), 0.5f },           // Higher for defensive positioning
    { new DistanceToBallReward(), 3.0f },     // Custom reward
    
    // Defensive actions
    { new SaveReward(), 100.0f },             // High save reward
    { new ZeroSumReward(new BumpReward(), 0.5f), 30.0f },
    
    // Avoid unnecessary risks
    { new PickupBoostReward(), 15.0f },       // More boost management
    { new AirReward(), 0.1f },                // Less aerial play
    
    // Team coordination
    { new ZeroSumReward(new VelocityBallToGoalReward(), 1), 3.0f },
    { new GoalReward(), 150.0f }
};
```

### 3. Aerial Play Specialist
```cpp
std::vector<WeightedReward> aerialRewards = {
    // Aerial focus
    { new AirReward(), 1.0f },                // Much higher aerial reward
    { new FaceBallReward(), 0.5f },
    
    // Aerial ball control
    { new StrongTouchReward(10, 150), 80.0f }, // Wider speed range
    { new TouchAccelReward(), 50.0f },        // Focus on ball acceleration
    
    // Boost for aerial maneuvers
    { new PickupBoostReward(), 15.0f },
    
    // Reduced ground play
    { new VelocityPlayerToBallReward(), 2.0f }, // Less ground chasing
    
    // Scoring remains important
    { new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
    { new GoalReward(), 150.0f }
};
```

### 4. Debug Configuration (Single Reward)
```cpp
std::vector<WeightedReward> debugRewards = {
    // Test single reward in isolation
    { new MyCustomReward(), 1.0f },
    
    // Minimal other rewards for baseline
    { new GoalReward(), 1.0f }  // Just to have some signal
};
```

---

## Conclusion

The GigaLearnCPP reward system provides a powerful, flexible foundation for training Rocket League bots. By understanding the architecture, following best practices, and iterating on reward design, you can create sophisticated behaviors that lead to effective and competitive AI agents.

Key takeaways:
- **Start simple** with basic rewards and gradually add complexity
- **Test rewards individually** to understand their impact
- **Use proper normalization** to maintain training stability
- **Consider team dynamics** with zero-sum wrappers
- **Profile performance** for production deployments

Remember: Reward engineering is often more art than science. Experiment, measure, and iterate based on the specific behaviors you want to encourage in your bot.

---

*This guide covers the comprehensive reward system in GigaLearnCPP. For the latest updates and additional examples, refer to the source code and unit tests.*