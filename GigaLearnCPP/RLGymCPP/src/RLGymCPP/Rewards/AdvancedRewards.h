#pragma once
#include "Reward.h"
#include "../Math.h"
#include "../CommonValues.h"
#include <map>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <atomic>

namespace RLGC {

	// Global state for curriculum
	struct AdvancedRewardGlobals {
		static inline std::atomic<uint64_t> TotalSteps = 0;
	};

	// Curriculum Reward Wrapper
	class CurriculumReward : public Reward {
	private:
		Reward* _child;
		uint64_t _minSteps;
		bool _ownsChild;

	public:
		CurriculumReward(Reward* child, uint64_t minSteps, bool ownsChild = true) 
			: _child(child), _minSteps(minSteps), _ownsChild(ownsChild) {}

		virtual ~CurriculumReward() {
			if (_ownsChild && _child) delete _child;
		}

		virtual void Reset(const GameState& initialState) override {
			if (_child) _child->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) override {
			if (_child) _child->PreStep(state);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (AdvancedRewardGlobals::TotalSteps < _minSteps) return 0.0f;
			return _child->GetReward(player, state, isFinal);
		}

		virtual std::string GetName() override {
			return "Curriculum(" + (_child ? _child->GetName() : "NULL") + ")";
		}
	};

	// Reset Shot Reward: Rewards using a flip reset to touch the ball
	class ResetShotReward : public Reward {
	private:
		std::map<uint32_t, uint64_t> _tickCountWhenResetObtained;

	public:
		virtual void Reset(const GameState& initial_state) override {
			_tickCountWhenResetObtained.clear();
		}

		virtual void PreStep(const GameState& state) override {
			for (const auto& player : state.players) {
				if (state.prev == NULL || player.prev == NULL) continue;

				bool gotReset = !player.prev->isOnGround && player.HasFlipOrJump() && !player.prev->HasFlipOrJump();
				if (gotReset) {
					_tickCountWhenResetObtained[player.carId] = state.lastArena->tickCount;
				}
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (state.prev == NULL || player.prev == NULL) return 0.0f;

			if (_tickCountWhenResetObtained.count(player.carId)) {
				bool flipWasUsedForTouch = player.ballTouchedStep && !player.isOnGround && !player.hasJumped && player.prev->HasFlipOrJump() && !player.HasFlipOrJump();
				
				if (flipWasUsedForTouch) {
					float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
					float ballSpeed = state.ball.vel.Length();
					float baseReward = (hitForce + ballSpeed) / (CommonValues::CAR_MAX_SPEED + CommonValues::BALL_MAX_SPEED);
					
					uint64_t ticksSinceReset = state.lastArena->tickCount - _tickCountWhenResetObtained[player.carId];
					float timeSinceReset = ticksSinceReset * CommonValues::TICK_TIME;
					float timeBonus = 1.0f + std::log1p(timeSinceReset);

					_tickCountWhenResetObtained.erase(player.carId);
					return baseReward * timeBonus;
				}
			}
			if (player.isOnGround) _tickCountWhenResetObtained.erase(player.carId);
			return 0.0f;
		}
	};

	// Flip Reset Reward Giga: Rewards obtaining a flip reset
	class FlipResetRewardGiga : public Reward {
	private:
		std::unordered_map<uint32_t, bool> prevCanJump;
		std::unordered_map<uint32_t, bool> hasReset;
		float flipResetR;
		float holdFlipResetR;

	public:
		FlipResetRewardGiga(float flipResetR = 1.0f, float holdFlipResetR = 0.0f) 
			: flipResetR(flipResetR), holdFlipResetR(holdFlipResetR) {}

		virtual void Reset(const GameState& initialState) override {
			prevCanJump.clear();
			hasReset.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			uint32_t carId = player.carId;
			float reward = 0.0f;

			bool nearBall = (player.pos - state.ball.pos).Length() < 170.0f;
			bool heightCheck = (player.pos.z < 300.0f) || (player.pos.z > CommonValues::CEILING_Z - 300.0f);
			
			// Simple wall distance check (approximate)
			bool wallDisCheck = std::abs(player.pos.x) > 3500.0f || std::abs(player.pos.y) > 4500.0f;

			bool canJump = !player.hasJumped;
			bool hasFlipped = (player.isJumping && player.isFlipping);

			if (wallDisCheck || hasFlipped) {
				hasReset[carId] = false;
			}

			if (nearBall && !heightCheck && !wallDisCheck) {
				bool prev = prevCanJump.count(carId) ? prevCanJump[carId] : true;
				bool gotReset = (!prev && canJump);

				bool airborne = player.pos.z > 150.0f;
				Vec carUp(player.rotMat[0][2], player.rotMat[1][2], player.rotMat[2][2]);
				Vec carToBall = state.ball.pos - player.pos;
				bool undersideContact = carToBall.Dot(carUp) < -CommonValues::BALL_RADIUS * 0.5f;

				if (gotReset && airborne && undersideContact) {
					hasReset[carId] = true;
					reward = flipResetR;
				}
			}

			if (hasReset[carId]) {
				reward += holdFlipResetR;
			}

			prevCanJump[carId] = canJump;
			return reward;
		}
	};

	// Continuous Flip Reset Reward: Encourages positioning for flip resets
	class ContinuousFlipResetReward : public Reward {
	public:
		float minHeight;
		float maxDist;

		ContinuousFlipResetReward(float minHeight = 150.f, float maxDist = 300.f) 
			: minHeight(minHeight), maxDist(maxDist) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.isOnGround || player.HasFlipOrJump()) return 0.f;
			if (player.pos.z < minHeight) return 0.f;
			if (player.rotMat.up.z >= 0) return 0.f; // Must be upside down

			float distToBall = (player.pos - state.ball.pos).Length();
			if (distToBall > maxDist) return 0.f;

			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			Vec relVel = player.vel - state.ball.vel;
			float approachSpeed = relVel.Dot(dirToBall);
			if (approachSpeed <= 0) return 0.f;

			float normSpeed = RS_CLAMP(approachSpeed / CommonValues::CAR_MAX_SPEED, 0.f, 1.f);
			float normAlign = ((-player.rotMat.up).Dot(dirToBall) + 1.f) / 2.f;
			float normDist = 1.f - RS_CLAMP(distToBall / maxDist, 0.f, 1.f);

			return std::min({normSpeed, normAlign, normDist});
		}
	};

	// Mawkzy Flick Reward: Rewards executing a Mawkzy flick
	class MawkzyFlickReward : public Reward {
	public:
		const float MIN_DRIBBLE_HEIGHT = CommonValues::BALL_RADIUS + 20.0f;
		const float MAX_DRIBBLE_HEIGHT = CommonValues::BALL_RADIUS + 120.0f;
		const float VELOCITY_SYNC_THRESHOLD = 400.0f;
		const float MIN_BACKFLIP_COMPONENT = 0.6f;
		const float MIN_STALL_COMPONENT = 0.8f;
		const float MIN_ANGULAR_VEL_X = 3.5f;

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev || !state.prev) return 0.f;

			bool justFlipped = player.isFlipping && !player.prev->isFlipping;
			if (!justFlipped) return 0.f;

			if (!player.ballTouchedStep) return 0.f;
			if (!player.prev->isOnGround) return 0.f;

			float prev_ball_height = state.prev->ball.pos.z;
			if (prev_ball_height < MIN_DRIBBLE_HEIGHT || prev_ball_height > MAX_DRIBBLE_HEIGHT) return 0.f;

			Vec player_vel_2d = player.prev->vel; player_vel_2d.z = 0;
			Vec ball_vel_2d = state.prev->ball.vel; ball_vel_2d.z = 0;
			if (player_vel_2d.Dist(ball_vel_2d) > VELOCITY_SYNC_THRESHOLD) return 0.f;

			const Action& action = player.prevAction;
			bool isStallBackflip = action.pitch > MIN_BACKFLIP_COMPONENT && 
				std::abs(action.yaw) > MIN_STALL_COMPONENT && 
				std::abs(action.roll) > MIN_STALL_COMPONENT && 
				(RS_SGN(action.yaw) != RS_SGN(action.roll));

			if (!isStallBackflip) return 0.f;

			Vec localAngVel = player.rotMat.Dot(player.angVel);
			if (std::abs(localAngVel.x) < MIN_ANGULAR_VEL_X) return 0.f;

			return 1.0f;
		}
	};

	// Air Roll Reward: Rewards air rolling while airborne
	class AirRollReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.isOnGround) return 0.0f;
			return std::abs(player.rotMat.forward.z); // Reward being sideways/upside down
		}
	};

	// Double Tap Reward: Rewards double taps
	class DoubleTapReward : public Reward {
	private:
		int _candidateCarId = -1;
		bool _wallBounceDetected = false;
		bool _initiationWasAerial = false;
		float _rewardAmount;
		float _minHeight;
		float _wallThresholdY;
		float _minWallBounceSpeed;
		std::unordered_map<int, float> _currentStepRewards;

	public:
		DoubleTapReward(float rewardAmount = 2.0f, float minHeight = 300.0f) 
			: _rewardAmount(rewardAmount), _minHeight(minHeight) {
			_wallThresholdY = CommonValues::BACK_WALL_Y - 150.0f;
			_minWallBounceSpeed = 500.0f;
		}

		virtual void Reset(const GameState& initialState) override {
			_candidateCarId = -1;
			_wallBounceDetected = false;
			_currentStepRewards.clear();
		}

		virtual void PreStep(const GameState& state) override {
			_currentStepRewards.clear();
			if (!state.prev) return;

			bool touchDetected = false;
			int toucherId = -1;
			bool isToucherAirborne = false;

			for (const auto& player : state.players) {
				if (player.ballTouchedStep) {
					touchDetected = true;
					toucherId = player.carId;
					isToucherAirborne = !player.isOnGround;
					break;
				}
			}

			if (touchDetected) {
				if (toucherId == _candidateCarId && _wallBounceDetected && isToucherAirborne && state.ball.pos.z > _minHeight) {
					_currentStepRewards[toucherId] = _rewardAmount;
					_candidateCarId = -1;
					_wallBounceDetected = false;
				} else {
					_candidateCarId = toucherId;
					_initiationWasAerial = isToucherAirborne;
					_wallBounceDetected = false;
				}
			} else if (_candidateCarId != -1) {
				bool hitWall = (std::abs(state.ball.pos.y) > _wallThresholdY) && (std::abs(state.ball.vel.y) > _minWallBounceSpeed);
				if (hitWall) {
					_wallBounceDetected = true;
				}
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return _currentStepRewards.count(player.carId) ? _currentStepRewards[player.carId] : 0.0f;
		}
	};

	// Musty Continuous Positioning Reward
	class MustyContinuousPositionningReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Placeholder for now, returning 0
			return 0.0f;
		}
	};

	// Kickoff Proximity Reward 2v2: Teaches proper kickoff roles
	class KickoffProximityReward2v2 : public Reward {
	private:
		enum class PlayerRole { GOER, CHEATER };

		bool IsKickoffActive(const GameState& state) {
			float ballSpeed = state.ball.vel.Length();
			float ballHeight = state.ball.pos.z;
			Vec ballPos2D = state.ball.pos; ballPos2D.z = 0;
			
			return (ballSpeed < 2.f && ballHeight < 150.f && ballPos2D.Length() < 50.f);
		}

		PlayerRole DeterminePlayerRole(const Player& player, const Player* teammate, const GameState& state) {
			float playerDistToBall = (player.pos - state.ball.pos).Length();
			float teammateDistToBall = (teammate->pos - state.ball.pos).Length();
			return (playerDistToBall < teammateDistToBall) ? PlayerRole::GOER : PlayerRole::CHEATER;
		}

	public:
		float goerReward = 1.0f;
		float cheaterReward = 0.5f;

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!IsKickoffActive(state)) return 0.f;

			const Player* teammate = nullptr;
			for (const auto& p : state.players) {
				if (p.team == player.team && p.carId != player.carId) {
					teammate = &p;
					break;
				}
			}
			if (!teammate) return 0.f;

			PlayerRole role = DeterminePlayerRole(player, teammate, state);
			
			if (role == PlayerRole::GOER) {
				// Goer: Rewarded for VELOCITY towards ball (prevents stalling)
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				float velTowardsBall = player.vel.Dot(dirToBall);
				float normVel = velTowardsBall / CommonValues::CAR_MAX_SPEED;
				return RS_CLAMP(normVel, 0.f, 1.f) * goerReward;
			} else {
				// Cheater: Rewarded for staying back near goal
				float distToGoal = (player.pos - CommonValues::BLUE_GOAL_BACK).Length();
				if (player.team == Team::ORANGE) distToGoal = (player.pos - CommonValues::ORANGE_GOAL_BACK).Length();
				return (1.f - RS_CLAMP(distToGoal / 5500.f, 0.f, 1.f)) * cheaterReward;
			}
		}
	};

	// Kaiyo Energy Reward: Rewards energy management (altitude + velocity + boost)
	class KaiyoEnergyReward : public Reward {
	public:
		const double GRAVITY = 650;
		const double MASS = 180;

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const auto max_energy = (MASS * GRAVITY * (CommonValues::CEILING_Z - 17.)) + 
				(0.5 * MASS * (CommonValues::CAR_MAX_SPEED * CommonValues::CAR_MAX_SPEED));
			double energy = 0;

			if (player.HasFlipOrJump()) energy += 0.35 * MASS * 292. * 292.;
			if (player.HasFlipOrJump() && !player.isOnGround) energy += 0.35 * MASS * 550. * 550.;
			energy += MASS * GRAVITY * (player.pos.z - 17.) * 0.75;
			
			double velocity = player.vel.Length();
			energy += 0.5 * MASS * velocity * velocity;
			energy += 7.97e6 * player.boost;

			double norm_energy = player.isDemoed ? 0.0 : (energy / max_energy);
			return static_cast<float>(norm_energy);
		}
	};

	// Air Dribble Reward: Rewards air dribble positioning and control
	class AirdribbleRewardV1 : public Reward {
	private:
		std::unordered_map<uint32_t, float> lastPlayerZ;
		std::unordered_map<uint32_t, float> lastBallZ;

	public:
		virtual void Reset(const GameState& initialState) override {
			lastPlayerZ.clear();
			lastBallZ.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			uint32_t carId = player.carId;
			float reward = 0.0f;
			
			Vec posDiff = state.ball.pos - player.pos;
			float distToBall = posDiff.Length();
			Vec normPosDiff = posDiff.Normalized();
			float facingBall = player.rotMat.forward.Dot(normPosDiff);

			float BallY = RS_MAX(state.ball.pos.y, 0.0f);
			float NewY = 6000.0f - BallY;
			float BallX = std::abs(state.ball.pos.x) + 92.75f;
			float LargestX = NewY * 0.683f;

			if (!player.isOnGround && state.ball.pos.z > 250.0f && player.pos.z < state.ball.pos.z && 
				state.ball.pos.y > 1000.0f && BallX < LargestX) {
				
				float prevPlayerZ = lastPlayerZ.count(carId) ? lastPlayerZ[carId] : player.pos.z;
				float prevBallZ = lastBallZ.count(carId) ? lastBallZ[carId] : state.ball.pos.z;
				bool ascending = (player.pos.z > prevPlayerZ && state.ball.pos.z > prevBallZ);
				
				if (ascending && distToBall < 400.0f && facingBall > 0.74f) {
					if (player.pos.y < state.ball.pos.y) {
						if (player.ballTouchedStep) reward += 20.0f;
						reward += 2.5f;
					}
					reward += facingBall * 0.5f;
					reward += (1.0f - (distToBall / 400.0f)) * 3.0f;
				}
			}
			
			lastPlayerZ[carId] = player.pos.z;
			lastBallZ[carId] = state.ball.pos.z;
			return reward * 0.05f; // Heavily scaled down
		}
	};

} // namespace RLGC
