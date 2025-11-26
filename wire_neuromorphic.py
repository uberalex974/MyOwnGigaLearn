"""
PHASE 10: WIRING NEUROMORPHIC
Ajouter et initialiser les membres Neuromorphic
"""

# Header addition
header_code = '''
		// Phase 10: Neuromorphic
		std::unique_ptr<GGL::Neuromorphic::SpikingNeuronLIF> spiking_neuron_;
		std::unique_ptr<GGL::Neuromorphic::EventProcessor> event_processor_;
		std::unique_ptr<GGL::Neuromorphic::STDP_Synapse> stdp_synapse_;
		std::unique_ptr<GGL::Neuromorphic::LiquidStateMachine> lsm_;
'''

# CPP initialization addition
init_code = '''
	// Phase 10
	spiking_neuron_ = std::make_unique<GGL::Neuromorphic::SpikingNeuronLIF>();
	event_processor_ = std::make_unique<GGL::Neuromorphic::EventProcessor>();
	stdp_synapse_ = std::make_unique<GGL::Neuromorphic::STDP_Synapse>(256, 256);
	lsm_ = std::make_unique<GGL::Neuromorphic::LiquidStateMachine>();
'''

# 1. Update Header
header_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h'
with open(header_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'std::unique_ptr<GGL::Neuromorphic::SpikingNeuronLIF> spiking_neuron_;' not in content:
    # Insert before "};"
    last_brace_idx = content.rfind('};')
    if last_brace_idx != -1:
        new_content = content[:last_brace_idx] + header_code + content[last_brace_idx:]
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Wired Neuromorphic in Header")

# 2. Update CPP
cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'
with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'spiking_neuron_ = std::make_unique' not in content:
    # Insert at end of constructor
    # Look for Phase 9 init code
    marker = 'task_embedder_ = std::make_unique<GGL::MetaLearning::TaskEmbedder>(10, 32);'
    idx = content.find(marker)
    if idx != -1:
        insert_pos = idx + len(marker)
        new_content = content[:insert_pos] + init_code + content[insert_pos:]
        with open(cpp_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Wired Neuromorphic in CPP")
    else:
        print("❌ Could not find Phase 9 init marker")
