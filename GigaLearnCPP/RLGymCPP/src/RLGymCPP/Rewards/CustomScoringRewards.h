#pragma once
#include "Reward.h"
#include "RewardWrapper.h"
#include "../CommonValues.h"
#include <cmath>

namespace RLGC {
    
    // 🎯 Goal Proximity Reward - Rewards being close to optimal shooting position
    class GoalProximityReward : public Reward {
    private:
        float optimalDistance; // Distance for best shooting angle
        
    public:
        GoalProximityReward(float optDist = 1500.0f) : optimalDistance(optDist) {}
        
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            Vec goalPos = (player.team == Team::BLUE) ? 
                CommonValues::ORANGE_GOAL_CENTER : 
                CommonValues::BLUE_GOAL_CENTER;
            
            float distanceToGoal = (player.position - goalPos).Length();
            float distanceToBall = (player.position - state.ball.position).Length();
            
            // Reward optimal positioning: close to ball, good shooting angle
            float optimalProximity = std::max(0.0f, 1.0f - std::abs(distanceToGoal - optimalDistance) / optimalDistance);
            float ballProximity = std::max(0.0f, 1.0f - distanceToBall / 1000.0f);
            
            return optimalProximity * ballProximity * 2.0f; // Scale up for significance
        }
        
        virtual std::string GetName() override {
            return "GoalProximityReward";
        }
    };
    
    // 🎯 Shooting Angle Reward - Rewards good shooting angles
    class ShootingAngleReward : public Reward {
    public:
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.ballTouchedStep) return 0.0f;
            
            Vec goalPos = (player.team == Team::BLUE) ? 
                CommonValues::ORANGE_GOAL_CENTER : 
                CommonValues::BLUE_GOAL_CENTER;
            
            // Calculate shooting angle quality
            Vec ballToGoal = (goalPos - state.ball.pos).Normalized();
            Vec ballVelocity = state.ball.vel.Normalized();
            
            float angleAlignment = ballToGoal.Dot(ballVelocity);
            return RS_MAX(0, angleAlignment); // Only positive alignment
        }
        
        virtual std::string GetName() override {
            return "ShootingAngleReward";
        }
    };
    
    // 🎯 Power Shot Reward - Rewards powerful, accurate shots
    class PowerShotReward : public Reward {
    private:
        float minPowerSpeed, maxPowerSpeed;
        
    public:
        PowerShotReward(float minSpeed = 80.0f, float maxSpeed = 150.0f) {
            minPowerSpeed = RLGC::Math::KPHToVel(minSpeed);
            maxPowerSpeed = RLGC::Math::KPHToVel(maxSpeed);
        }
        
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.ballTouchedStep || !state.prev) return 0.0f;
            
            float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
            if (hitForce < minPowerSpeed) return 0.0f;
            
            // Reward power and accuracy combination
            Vec goalPos = (player.team == Team::BLUE) ? 
                CommonValues::ORANGE_GOAL_CENTER : 
                CommonValues::BLUE_GOAL_CENTER;
            
            Vec shotDirection = (goalPos - state.ball.pos).Normalized();
            Vec shotVelocity = state.ball.vel.Normalized();
            float accuracy = shotDirection.Dot(shotVelocity);
            
            float powerReward = RS_MIN(1.0f, hitForce / maxPowerSpeed);
            float accuracyReward = RS_MAX(0, accuracy); // Only positive accuracy
            
            return powerReward * accuracyReward * 4.0f; // Scale up for significance
        }
        
        virtual std::string GetName() override {
            return "PowerShotReward";
        }
    };
    
    // 🎯 Ball Control for Scoring - Rewards maintaining ball possession in scoring positions
    class ScoringBallControlReward : public Reward {
    private:
        float controlRadius;
        
    public:
        ScoringBallControlReward(float radius = 300.0f) : controlRadius(radius) {}
        
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.ballTouchedStep) return 0.0f;
            
            float distanceToBall = (player.pos - state.ball.pos).Length();
            if (distanceToBall > controlRadius) return 0.0f;
            
            // Check if in good scoring position
            Vec goalPos = (player.team == Team::BLUE) ? 
                CommonValues::ORANGE_GOAL_CENTER : 
                CommonValues::BLUE_GOAL_CENTER;
            
            float distanceToGoal = (player.pos - goalPos).Length();
            Vec ballDirection = (state.ball.pos - player.pos).Normalized();
            Vec goalDirection = (goalPos - player.pos).Normalized();
            
            // Reward being close to ball AND having good angle to goal
            float proximityReward = RS_CLAMP(1.0f - distanceToBall / controlRadius, 0.0f, 1.0f);
            float angleReward = RS_MAX(0, ballDirection.Dot(goalDirection));
            
            // Bonus for being in scoring third of field
            float fieldPosition = abs(player.pos.y) / CommonValues::BACK_WALL_Y;
            float fieldBonus = RS_CLAMP(fieldPosition * 2.0f, 0.0f, 1.0f); // More reward further up field
            
            return (proximityReward * 0.4f + angleReward * 0.4f + fieldBonus * 0.2f) * 3.0f;
        }
        
        virtual std::string GetName() override {
            return "ScoringBallControlReward";
        }
    };
    
    // 🎯 Clipped Scoring Reward Wrapper - Prevents reward explosion
    class ClippedScoringReward : public RewardWrapper {
    private:
        float maxPositive, maxNegative;
        
    public:
        ClippedScoringReward(Reward* child, float maxPos = 5.0f, float maxNeg = -2.0f)
            : RewardWrapper(child), maxPositive(maxPos), maxNegative(maxNeg) {}
        
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            float reward = child->GetReward(player, state, isFinal);
            
            // Enhanced clipping for scoring rewards
            if (reward > maxPositive) return maxPositive;
            if (reward < maxNegative) return maxNegative;
            
            return reward;
        }
        
        virtual std::string GetName() override {
            return "Clipped(" + child->GetName() + ")";
        }
    };
}