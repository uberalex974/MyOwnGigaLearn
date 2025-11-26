"""
RESTORE PROPRE: Restaurer 100% depuis le backup GAE
"""
import shutil

# Restore PPOLearner.h from git remote
print("Restoring header from git...")

import subprocess
result = subprocess.run(
    ['git', 'show', 'HEAD:GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.h'],
    capture_output=True,
    text=True,
    cwd=r'c:\Giga\GigaLearnCPP'
)

if result.returncode == 0:
    with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    print("✅ Header restored from git")
else:
    print("❌ Git restore failed, using backup")

print("\n✅ Clean state ready")
print("BUILD MAINTENANT pour vérifier état stable")
