#pragma once

#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace GGL {

/// 🔥 OPTIMIZED: Performance Monitor for Real-time Training Metrics
/// Provides comprehensive monitoring of training performance and GPU utilization
class PerformanceMonitor {
public:
    struct PerformanceMetrics {
        // Training metrics
        float policy_loss;
        float value_loss;
        float entropy;
        float kl_divergence;
        float learning_rate;
        
        // Game performance metrics
        float win_rate;
        float goals_per_match;
        float saves_per_match;
        float shots_on_target_percentage;
        float possession_time_percentage;
        
        // Computational metrics
        float gpu_utilization;
        float memory_efficiency;
        float inference_latency;
        float training_throughput;
        
        // Temporal metrics
        float convergence_speed;
        float performance_stability;
        float adaptation_rate;
        
        // Timing information
        std::chrono::steady_clock::time_point timestamp;
    };

    struct BenchmarkResults {
        float baseline_performance;           // vs hand-coded bot
        float sota_comparison;                // vs other RL frameworks
        float hardware_scaling;               // scaling efficiency
        float training_efficiency;            // samples to convergence
        std::unordered_map<std::string, float> custom_metrics;
    };

    struct PerformanceAlert {
        enum class Severity { INFO, WARNING, CRITICAL };
        enum class Category { MEMORY, GPU, TRAINING, CONVERGENCE };
        
        Severity severity;
        Category category;
        std::string message;
        std::chrono::steady_clock::time_point timestamp;
        float threshold_value;
        float current_value;
    };

    PerformanceMonitor();
    ~PerformanceMonitor();

    // Metric collection
    void update_metrics(const PerformanceMetrics& metrics);
    void add_custom_metric(const std::string& name, float value);
    
    // Real-time monitoring
    void start_monitoring();
    void stop_monitoring();
    void setup_real_time_monitoring();
    void enable_performance_alerts();
    
    // Performance analysis
    PerformanceMetrics get_current_metrics() const;
    std::vector<PerformanceMetrics> get_historical_metrics(size_t count = 100) const;
    BenchmarkResults run_comprehensive_benchmark();
    
    // Alert system
    void set_alert_threshold(const std::string& metric_name, float threshold, PerformanceAlert::Severity severity);
    std::vector<PerformanceAlert> get_active_alerts() const;
    void clear_alerts();
    
    // Reporting
    void generate_performance_report() const;
    void export_metrics_to_file(const std::string& filepath) const;
    
    // Configuration
    void set_monitoring_interval(std::chrono::milliseconds interval);
    void set_gpu_monitoring_enabled(bool enabled);
    void set_memory_monitoring_enabled(bool enabled);

private:
    PerformanceMetrics current_metrics_;
    std::vector<PerformanceMetrics> historical_metrics_;
    std::vector<PerformanceAlert> active_alerts_;
    
    // Monitoring configuration
    bool monitoring_enabled_ = false;
    bool gpu_monitoring_enabled_ = true;
    bool memory_monitoring_enabled_ = true;
    std::chrono::milliseconds monitoring_interval_{1000}; // 1 second
    
    // Alert thresholds
    std::unordered_map<std::string, std::pair<float, PerformanceAlert::Severity>> alert_thresholds_;
    
    // Performance tracking
    size_t max_history_size_ = 1000;
    std::chrono::steady_clock::time_point start_time_;
    
    // Internal helper methods
    void check_alerts();
    void monitor_gpu_utilization();
    void monitor_memory_usage();
    void calculate_derived_metrics();
    void cleanup_old_metrics();
    
    // GPU monitoring (platform-specific)
    float get_gpu_utilization_percentage() const;
    size_t get_gpu_memory_usage() const;
    float get_gpu_temperature() const;
};

/// 🔥 OPTIMIZED: Training Efficiency Analyzer
/// Analyzes training efficiency and provides optimization recommendations
class TrainingEfficiencyAnalyzer {
public:
    struct EfficiencyReport {
        float samples_per_second;
        float steps_per_second;
        float convergence_rate;
        float gpu_utilization_efficiency;
        float memory_efficiency;
        float overall_efficiency_score;
        std::vector<std::string> optimization_recommendations;
        std::unordered_map<std::string, float> efficiency_metrics;
    };

    TrainingEfficiencyAnalyzer();
    ~TrainingEfficiencyAnalyzer();

    // Analysis methods
    EfficiencyReport analyze_efficiency(const std::vector<PerformanceMetrics>& metrics_history);
    void identify_bottlenecks(const PerformanceMetrics& current_metrics);
    std::vector<std::string> generate_optimization_suggestions();
    
    // Efficiency scoring
    float calculate_overall_efficiency_score(const PerformanceMetrics& metrics) const;
    float calculate_training_speed_score(const PerformanceMetrics& metrics) const;
    float calculate_resource_utilization_score(const PerformanceMetrics& metrics) const;
    
    // Benchmarking
    void set_baseline_performance(const PerformanceMetrics& baseline);
    EfficiencyReport compare_to_baseline(const PerformanceMetrics& current_metrics) const;
    
private:
    PerformanceMetrics baseline_metrics_;
    bool baseline_set_ = false;
    
    // Efficiency thresholds
    struct EfficiencyThresholds {
        float excellent_gpu_utilization = 0.9f;
        float good_memory_efficiency = 0.8f;
        float excellent_training_speed = 5000.0f; // steps per second
        float good_convergence_rate = 0.01f;      // improvement per 1000 steps
    };
    
    EfficiencyThresholds thresholds_;
    
    // Internal analysis methods
    void analyze_training_speed(const std::vector<PerformanceMetrics>& metrics, EfficiencyReport& report) const;
    void analyze_resource_utilization(const PerformanceMetrics& metrics, EfficiencyReport& report) const;
    void analyze_convergence_pattern(const std::vector<PerformanceMetrics>& metrics, EfficiencyReport& report) const;
    void generate_recommendations(const EfficiencyReport& report);
};

/// 🔥 OPTIMIZED: Memory Profiler for Deep Memory Analysis
/// Provides detailed memory profiling and optimization recommendations
class MemoryProfiler {
public:
    struct MemoryProfile {
        size_t total_allocated;               // Total allocated memory
        size_t peak_usage;                    // Peak memory usage
        size_t fragmentation;                 // Memory fragmentation
        std::unordered_map<std::string, size_t> allocation_by_type;
        std::vector<std::pair<std::string, size_t>> largest_allocations;
        float allocation_rate;                // MB/s allocation rate
        float deallocation_rate;              // MB/s deallocation rate
        std::chrono::steady_clock::time_point profile_timestamp;
    };

    struct MemoryOptimization {
        bool recommend_defragmentation;
        bool recommend_pool_resizing;
        bool detect_memory_leaks;
        std::vector<std::string> optimization_suggestions;
        float estimated_memory_savings;
    };

    MemoryProfiler();
    ~MemoryProfiler();

    // Profiling control
    void start_profiling();
    void stop_profiling();
    void enable_leak_detection();
    void setup_memory_alerts(size_t threshold_mb);
    
    // Memory analysis
    MemoryProfile get_current_profile() const;
    MemoryOptimization analyze_and_optimize();
    void generate_memory_report() const;
    
    // Memory tracking
    void track_allocation(const std::string& type, size_t size);
    void track_deallocation(const std::string& type, size_t size);
    void mark_tensor_lifecycle(const std::string& tensor_id, bool is_created);
    
    // Optimization
    void suggest_memory_optimizations();
    void optimize_tensor_layout();
    void enable_memory_pooling();

private:
    MemoryProfile current_profile_;
    std::vector<MemoryProfile> historical_profiles_;
    size_t alert_threshold_mb_;
    bool leak_detection_enabled_ = false;
    bool profiling_active_ = false;
    
    // Memory tracking
    std::unordered_map<std::string, size_t> allocation_history_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> tensor_lifecycles_;
    
    // Internal analysis methods
    void update_allocation_stats();
    void detect_memory_leaks();
    void calculate_fragmentation();
    void analyze_allocation_patterns();
    
    // Memory optimization helpers
    size_t estimate_optimal_pool_size(const std::string& allocation_type) const;
    void generate_optimization_suggestions(MemoryOptimization& optimization) const;
};

} // namespace GGL