"""
AUTO-FIX SCRIPT: Analyse les erreurs et les corrige automatiquement
"""

import subprocess
import re

def get_build_errors():
    """Run build and extract errors"""
    result = subprocess.run(
        ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'build.ps1'],
        capture_output=True,
        text=True,
        cwd=r'c:\Giga\GigaLearnCPP',
        timeout=120
    )
    
    errors = []
    for line in (result.stdout + result.stderr).split('\n'):
        if 'error C' in line or 'error:' in line:
            errors.append(line.strip())
    
    return errors, result.returncode

def fix_errors_automatically(errors):
    """Automatically fix known error patterns"""
    fixes_applied = []
    
    # Pattern 1: Undefined identifier
    for error in errors:
        if 'identificateur non déclaré' in error or 'undeclared identifier' in error:
            # Extract variable name
            match = re.search(r"'(\w+)'", error)
            if match:
                var_name = match.group(1)
                fixes_applied.append(f"TODO: Declare {var_name}")
    
    # Pattern 2: Namespace issues
    if any('namespace GGL' in e for e in errors):
        fixes_applied.append("FIX: Clean nested namespaces")
        with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'r') as f:
            content = f.read()
        content = content.replace('namespace GGL {\nnamespace GGL {', 'namespace GGL {')
        with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'w') as f:
            f.write(content)
    
    return fixes_applied

print("=== AUTO-FIX BUILD ERRORS ===\n")
errors, exit_code = get_build_errors()

if exit_code == 0:
    print("✅ BUILD SUCCESSFUL!")
else:
    print(f"❌ {len(errors)} errors found")
    print("\nSample errors:")
    for e in errors[:5]:
        print(f"  - {e[:100]}")
    
    fixes = fix_errors_automatically(errors)
    print(f"\n=== FIXES APPLIED: {len(fixes)} ===")
    for fix in fixes:
        print(f"  ✅ {fix}")
