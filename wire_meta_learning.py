"""
PHASE 9: WIRING META-LEARNING
Ajouter et initialiser les membres Meta-Learning
"""

# Header addition
header_code = '''
		// Phase 9: Meta-Learning
		std::unique_ptr<GGL::MetaLearning::MAML> maml_;
		std::unique_ptr<GGL::MetaLearning::Reptile> reptile_;
		std::unique_ptr<GGL::MetaLearning::MetaSGD> meta_sgd_;
		std::unique_ptr<GGL::MetaLearning::TaskEmbedder> task_embedder_;
'''

# CPP initialization addition
init_code = '''
	// Phase 9
	maml_ = std::make_unique<GGL::MetaLearning::MAML>();
	reptile_ = std::make_unique<GGL::MetaLearning::Reptile>();
	// meta_sgd_ requires params, skip for now or use placeholder
	task_embedder_ = std::make_unique<GGL::MetaLearning::TaskEmbedder>(10, 32);
'''

# 1. Update Header
header_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h'
with open(header_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'std::unique_ptr<GGL::MetaLearning::MAML> maml_;' not in content:
    # Insert before "};"
    last_brace_idx = content.rfind('};')
    if last_brace_idx != -1:
        new_content = content[:last_brace_idx] + header_code + content[last_brace_idx:]
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Wired Meta-Learning in Header")

# 2. Update CPP
cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'
with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'maml_ = std::make_unique' not in content:
    # Insert at end of constructor (we know where it is from previous script)
    # Look for the Phase 7 init code we added
    marker = 'level_replay_selector_ = std::make_unique<GGL::QuantumReady::LevelReplaySelector>();'
    idx = content.find(marker)
    if idx != -1:
        # Insert after this line
        insert_pos = idx + len(marker)
        new_content = content[:insert_pos] + init_code + content[insert_pos:]
        with open(cpp_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Wired Meta-Learning in CPP")
    else:
        print("❌ Could not find Phase 7 init marker")
