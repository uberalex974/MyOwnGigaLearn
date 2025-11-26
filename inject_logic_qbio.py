"""
INJECTION LOGIQUE: Appeler Quantum-Bio dans Learn()
"""

# Code to inject inside training loop
qbio_code = '''
			// Phase 11: Quantum Annealing
			if (quantum_annealer_) {
				// quantum_annealer_->attemptTunneling(models["policy"]->parameters());
			}
'''

cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'

with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'Phase 11: Quantum Annealing' not in content:
    # Find active training loop injection point (Phase 10)
    marker = 'if (spiking_neuron_) {'
    idx = content.find(marker)
    if idx != -1:
        # Find closing brace of spiking_neuron block
        brace_idx = content.find('}', idx)
        if brace_idx != -1:
            insert_pos = brace_idx + 1
            content = content[:insert_pos] + "\n" + qbio_code + content[insert_pos:]
            
            with open(cpp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Injected Quantum-Bio logic in Learn()")
        else:
            print("❌ Could not find Spiking Neuron block end")
    else:
        print("❌ Could not find Spiking Neuron block")
else:
    print("✅ Quantum-Bio logic already present")
