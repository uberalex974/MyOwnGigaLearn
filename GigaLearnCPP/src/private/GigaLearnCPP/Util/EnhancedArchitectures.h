#pragma once
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace GGL {

    // Enhanced Neural Network Architectures with Attention Mechanisms
    class AttentionModule : public torch::nn::Module {
    public:
        AttentionModule(int input_dim, int hidden_dim, int num_heads = 8)
            : input_dim_(input_dim), hidden_dim_(hidden_dim), num_heads_(num_heads) {
            
            // Multi-head attention layers
            query_linear_ = register_module("query_linear", torch::nn::Linear(input_dim, hidden_dim));
            key_linear_ = register_module("key_linear", torch::nn::Linear(input_dim, hidden_dim));
            value_linear_ = register_module("value_linear", torch::nn::Linear(input_dim, hidden_dim));
            
            // Output projection
            output_linear_ = register_module("output_linear", torch::nn::Linear(hidden_dim, input_dim));
            
            // Layer normalization
            norm1_ = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_dim})));
            norm2_ = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_dim})));
            
            // Feed-forward network
            ff_linear1_ = register_module("ff_linear1", torch::nn::Linear(input_dim, hidden_dim));
            ff_linear2_ = register_module("ff_linear2", torch::nn::Linear(hidden_dim, input_dim));
        }
        
        torch::Tensor forward(torch::Tensor x) {
            // Self-attention
            auto attn_output = self_attention(x);
            x = norm1_->forward(x + attn_output);
            
            // Feed-forward
            auto ff_output = feed_forward(x);
            x = norm2_->forward(x + ff_output);
            
            return x;
        }
        
    private:
        int input_dim_;
        int hidden_dim_;
        int num_heads_;
        
        torch::nn::Linear query_linear_{nullptr};
        torch::nn::Linear key_linear_{nullptr};
        torch::nn::Linear value_linear_{nullptr};
        torch::nn::Linear output_linear_{nullptr};
        torch::nn::LayerNorm norm1_{nullptr};
        torch::nn::LayerNorm norm2_{nullptr};
        torch::nn::Linear ff_linear1_{nullptr};
        torch::nn::Linear ff_linear2_{nullptr};
        
        torch::Tensor self_attention(torch::Tensor x) {
            auto batch_size = x.size(0);
            auto seq_len = x.size(1);
            
            // Linear projections
            auto q = query_linear_->forward(x).view({batch_size, seq_len, num_heads_, -1}).transpose(1, 2);
            auto k = key_linear_->forward(x).view({batch_size, seq_len, num_heads_, -1}).transpose(1, 2);
            auto v = value_linear_->forward(x).view({batch_size, seq_len, num_heads_, -1}).transpose(1, 2);
            
            // Scaled dot-product attention
            auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(hidden_dim_ / num_heads_);
            auto attention_weights = torch::softmax(scores, -1);
            auto context = torch::matmul(attention_weights, v);
            
            // Concatenate heads and project
            context = context.transpose(1, 2).contiguous().view({batch_size, seq_len, hidden_dim_});
            return output_linear_->forward(context);
        }
        
        torch::Tensor feed_forward(torch::Tensor x) {
            return ff_linear2_->forward(torch::relu(ff_linear1_->forward(x)));
        }
    };
    
    // Multi-Scale Feature Extraction Module
    class MultiScaleExtractor : public torch::nn::Module {
    public:
        MultiScaleExtractor(int input_dim, std::vector<int> scales = {1, 2, 4, 8})
            : input_dim_(input_dim), scales_(scales) {
            
            for (int i = 0; i < scales.size(); ++i) {
                auto scale = scales[i];
                auto scale_dim = input_dim / 4; // Reduced dimension for efficiency
                
                // Convolutional layers for each scale
                auto conv = torch::nn::Conv1d(
                    torch::nn::Conv1dOptions(input_dim, scale_dim, 3)
                        .padding(scale - 1)
                        .dilation(scale)
                );
                conv_layers_.push_back(conv);
                register_module("conv_" + std::to_string(i), conv);
                
                // Batch normalization
                auto bn = torch::nn::BatchNorm1d(scale_dim);
                bn_layers_.push_back(bn);
                register_module("bn_" + std::to_string(i), bn);
            }
            
            // Feature fusion layer
            auto total_scale_dim = input_dim / 4 * scales.size();
            fusion_linear_ = register_module("fusion_linear", torch::nn::Linear(total_scale_dim, input_dim));
        }
        
        torch::Tensor forward(torch::Tensor x) {
            std::vector<torch::Tensor> scale_features;
            
            for (int i = 0; i < scales_.size(); ++i) {
                auto scale = scales_[i];
                
                // Apply multi-scale convolution
                auto conv = conv_layers_[i];
                auto bn = bn_layers_[i];
                
                // Reshape for 1D convolution if needed
                auto x_conv = x.size(-1) > 1 ? x.transpose(-1, -2) : x;
                
                auto feature = torch::relu(bn->forward(conv->forward(x_conv)));
                
                // Adaptive pooling to fixed size
                feature = torch::adaptive_avg_pool1d(feature, 1);
                scale_features.push_back(feature.squeeze(-1));
            }
            
            // Concatenate multi-scale features
            auto multi_scale_features = torch::cat(scale_features, -1);
            
            // Fusion and residual connection
            auto fused = fusion_linear_->forward(multi_scale_features);
            return torch::relu(x + fused);
        }
        
    private:
        int input_dim_;
        std::vector<int> scales_;
        torch::nn::Linear fusion_linear_{nullptr};
        std::vector<torch::nn::Conv1d> conv_layers_;
        std::vector<torch::nn::BatchNorm1d> bn_layers_;
    };
    
    // Enhanced Policy Network with Attention
    class AttentionPolicyNetwork : public torch::nn::Module {
    public:
        AttentionPolicyNetwork(
            int input_dim, 
            int hidden_dim, 
            int output_dim,
            int num_heads = 8,
            int num_layers = 3
        ) : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim) {
            
            // Input projection
            input_projection_ = register_module("input_projection", torch::nn::Linear(input_dim, hidden_dim));
            
            // Multi-scale feature extraction
            multi_scale_ = std::make_shared<MultiScaleExtractor>(hidden_dim);
            
            // Attention layers
            for (int i = 0; i < num_layers; ++i) {
                auto attn = std::make_shared<AttentionModule>(hidden_dim, hidden_dim * 2, num_heads);
                register_module("attention_" + std::to_string(i), attn);
                attention_layers_.push_back(attn);
            }
            
            // Output layers
            policy_head_ = register_module("policy_head", torch::nn::Linear(hidden_dim, output_dim));
            
            // Dropout for regularization
            dropout_ = register_module("dropout", torch::nn::Dropout(0.1f));
        }
        
        torch::Tensor forward(torch::Tensor x) {
            // Input projection
            x = torch::relu(input_projection_->forward(x));
            
            // Multi-scale feature extraction
            x = multi_scale_->forward(x);
            
            // Apply attention layers
            for (auto& attn_module : attention_layers_) {
                x = attn_module->forward(x);
                x = dropout_->forward(x);
            }
            
            // Policy output
            auto policy_logits = policy_head_->forward(x);
            return policy_logits;
        }
        
    private:
        int input_dim_, hidden_dim_, output_dim_;
        torch::nn::Linear input_projection_{nullptr};
        std::shared_ptr<MultiScaleExtractor> multi_scale_;
        torch::nn::Linear policy_head_{nullptr};
        torch::nn::Dropout dropout_{nullptr};
        std::vector<std::shared_ptr<AttentionModule>> attention_layers_;
    };
    
    // Enhanced Value Network with Spatial Attention
    class SpatialValueNetwork : public torch::nn::Module {
    public:
        SpatialValueNetwork(
            int input_dim,
            int hidden_dim,
            int spatial_dims = 3  // x, y, z coordinates
        ) : input_dim_(input_dim), hidden_dim_(hidden_dim), spatial_dims_(spatial_dims) {
            
            // Feature extraction layers
            feature_layers_ = register_module("feature_layers", torch::nn::Sequential(
                torch::nn::Linear(input_dim, hidden_dim),
                torch::nn::ReLU(),
                torch::nn::Linear(hidden_dim, hidden_dim),
                torch::nn::ReLU()
            ));
            
            // Spatial attention for position-aware value estimation
            spatial_attention_ = std::make_shared<AttentionModule>(hidden_dim, hidden_dim, 4);
            
            // Value head
            value_head_ = register_module("value_head", torch::nn::Linear(hidden_dim, 1));
            
            // Layer normalization
            norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        }
        
        torch::Tensor forward(torch::Tensor x) {
            // Feature extraction
            auto features = feature_layers_->forward(x);
            
            // Add positional encoding for spatial awareness
            auto positional_features = add_positional_encoding(features);
            
            // Spatial attention
            auto attended_features = spatial_attention_->forward(positional_features);
            
            // Residual connection
            features = norm_->forward(features + attended_features);
            
            // Value estimation
            auto value = value_head_->forward(features);
            
            return value;
        }
        
    private:
        int input_dim_, hidden_dim_, spatial_dims_;
        torch::nn::Sequential feature_layers_{nullptr};
        std::shared_ptr<AttentionModule> spatial_attention_;
        torch::nn::Linear value_head_{nullptr};
        torch::nn::LayerNorm norm_{nullptr};
        
        torch::Tensor add_positional_encoding(torch::Tensor x) {
            auto seq_len = x.size(1);
            auto hidden_dim = x.size(2);
            
            // Create position indices [seq_len, hidden_dim]
            auto position = torch::arange(seq_len, x.device()).unsqueeze(1).to(x.dtype()).expand({seq_len, hidden_dim});
            
            // Create div_term for positional encoding [hidden_dim/2]
            auto div_term = torch::exp(torch::arange(0, hidden_dim, 2, x.device()).to(x.dtype()) * 
                                     (-std::log(10000.0) / static_cast<double>(hidden_dim)));
            
            // Create the complete positional encoding tensor
            auto positional_encoding = torch::zeros({seq_len, hidden_dim}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));
            
            // Manually assign values using a different approach
            // Create base tensors for even and odd positions
            auto pe_even = torch::zeros({seq_len, hidden_dim}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));
            auto pe_odd = torch::zeros({seq_len, hidden_dim}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));
            
            // Fill even positions with sin
            for (int64_t i = 0; i < hidden_dim; i += 2) {
                auto pos_slice = position.slice(1, i, i + 1);
                auto div_slice = div_term.slice(0, i / 2, i / 2 + 1);
                pe_even.slice(1, i, i + 1) = torch::sin(pos_slice * div_slice);
            }
            
            // Fill odd positions with cos
            for (int64_t i = 1; i < hidden_dim; i += 2) {
                auto pos_slice = position.slice(1, i, i + 1);
                auto div_slice = div_term.slice(0, i / 2, i / 2 + 1);
                pe_odd.slice(1, i, i + 1) = torch::cos(pos_slice * div_slice);
            }
            
            positional_encoding = pe_even + pe_odd;
            
            return x + positional_encoding.unsqueeze(0);
        }
    };
    
    // Progressive Training Manager for curriculum learning
    class ProgressiveTrainingManager : public torch::nn::Module {
    public:
        struct TrainingStage {
            std::string name;
            int difficulty_level;
            std::shared_ptr<torch::nn::Module> model;
            float learning_rate;
            bool is_active;
            
            TrainingStage(const std::string& n, int level, float lr) 
                : name(n), difficulty_level(level), learning_rate(lr), is_active(false) {}
        };
        
        ProgressiveTrainingManager() {
            stages_ = std::vector<TrainingStage>();
            current_stage_ = 0;
            stage_transition_threshold_ = 0.8f; // 80% performance threshold
        }
        
        // Add training stage
        void AddStage(const std::string& name, int difficulty_level, float learning_rate) {
            stages_.emplace_back(name, difficulty_level, learning_rate);
        }
        
        // Progress to next stage if criteria met
        bool ProgressToNextStage(float performance_metric) {
            if (current_stage_ < stages_.size() - 1 && 
                performance_metric >= stage_transition_threshold_) {
                
                stages_[current_stage_].is_active = false;
                current_stage_++;
                stages_[current_stage_].is_active = true;
                
                // Log progression (stub implementation)
                return true;
            }
            return false;
        }
        
        // Get current stage information
        const TrainingStage& GetCurrentStage() const {
            return stages_[current_stage_];
        }
        
        // Get all stages
        const std::vector<TrainingStage>& GetAllStages() const {
            return stages_;
        }
        
        // Dynamic difficulty adjustment
        void AdjustDifficulty(float performance_trend) {
            if (performance_trend < -0.1f) {
                // Performance decreasing, reduce difficulty
                if (current_stage_ > 0) {
                    stages_[current_stage_].is_active = false;
                    current_stage_--;
                    stages_[current_stage_].is_active = true;
                }
            } else if (performance_trend > 0.1f) {
                // Performance improving, consider advancing
                ProgressToNextStage(1.0f);
            }
        }
        
    private:
        std::vector<TrainingStage> stages_;
        int current_stage_;
        float stage_transition_threshold_;
    };
}