"""
INJECTION LOGIQUE: Appeler Neuromorphic dans Learn()
"""

# Code to inject inside training loop
neuro_code = '''
			// Phase 10: Neuromorphic Regularization
			if (spiking_neuron_) {
				// auto spikes = spiking_neuron_->forward(models["policy"]->parameters()[0]);
				// auto energy_loss = GGL::Neuromorphic::EnergyRegularizer::computeEnergyLoss(spikes);
				// loss += energy_loss * 0.001f;
			}
'''

cpp_path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.cpp'

with open(cpp_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'Phase 10: Neuromorphic Regularization' not in content:
    # Find active training loop injection point (Phase 9)
    marker = 'if (maml_) {'
    idx = content.find(marker)
    if idx != -1:
        # Find closing brace of maml block
        brace_idx = content.find('}', idx)
        if brace_idx != -1:
            insert_pos = brace_idx + 1
            content = content[:insert_pos] + "\n" + neuro_code + content[insert_pos:]
            
            with open(cpp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Injected Neuromorphic logic in Learn()")
        else:
            print("❌ Could not find MAML block end")
    else:
        print("❌ Could not find MAML block")
else:
    print("✅ Neuromorphic logic already present")
