"""
FIX: RECONSTRUCTION PROPRE DE OPTIMIZATIONSHELPERS.H
"""

# We will read the file, extract the content of each namespace, and rewrite the file with clean structure.
# This avoids the "insert at end" trap.

import re

path = r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\OptimizationsHelpers.h'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Regex to capture content between "namespace X {" and the matching closing brace is hard.
# Instead, we know the namespaces we added.
# We can just ensure the file ends correctly.

# If the file has unbalanced braces (e.g. level > 0), we need to close them.
# But simply appending '}' might not be enough if we nested GGL inside GGL.

# Let's try to identify where "namespace GGL" starts and ends.
# In our scripts, we often added:
# namespace GGL {
# namespace PhaseX {
# ...
# } // namespace PhaseX
# } // namespace GGL

# If we inserted BEFORE the last closing brace of the previous block, we might have:
# namespace GGL {
#   namespace Phase1 { ... }
#   namespace GGL { namespace Phase2 { ... } }  <-- Nested GGL!
# }

# Let's look for "namespace GGL" occurrences.
ggl_count = content.count("namespace GGL")
print(f"Found 'namespace GGL' {ggl_count} times.")

# If > 1, we have a problem (unless they are sequential, but we intended one big file).
# Actually, our scripts usually did:
# namespace GGL { namespace PhaseX { ... } }
# And we appended this to the file.

# Wait, the previous scripts did:
# lines.insert(insert_idx, phase_code)
# where insert_idx = len(lines) - 3
# And phase_code contained "namespace GGL { ... }"

# So we effectively did:
# namespace GGL { ... } // Original
# namespace GGL { ... } // Phase 9
# namespace GGL { ... } // Phase 10
# ...
# } // End of original (maybe?)

# If the original file ended with "}" (closing GGL), and we inserted BEFORE it...
# Then we have:
# namespace GGL { ...
#    namespace GGL { ... }
# }

# This is valid C++ (nested namespaces), BUT `GGL::GGL::MetaLearning` is NOT `GGL::MetaLearning`.
# So `PPOLearner.h` looking for `GGL::MetaLearning` fails if it's actually `GGL::GGL::MetaLearning`.

# SOLUTION:
# Remove all "namespace GGL {" and corresponding closing braces from the *inserted* blocks,
# OR ensure they are NOT nested inside the outer GGL.

# Better yet: Flatten the file.
# 1. Keep the first "namespace GGL {"
# 2. Remove all internal "namespace GGL {" and their closing "}"
# 3. Ensure one final "}" at the end.

# Let's try to fix it by replacing "namespace GGL {" with nothing (except the first one)
# and removing the matching closing braces? That's risky.

# Safer approach:
# The file structure SHOULD be:
# #pragma once
# ... includes ...
# namespace GGL {
#    namespace Optimizations { ... }
#    namespace Additional { ... }
#    ...
# }

# Currently it probably looks like:
# namespace GGL {
#    namespace Optimizations { ... }
#    namespace GGL { namespace MetaLearning { ... } } // Inserted inside!
# }

# We need to change "namespace GGL { namespace MetaLearning" to just "namespace MetaLearning".
# And remove the extra "}" at the end of that block.

new_content = content

# Replace "namespace GGL {\nnamespace MetaLearning" with "namespace MetaLearning"
# We need to handle whitespace.

replacements = [
    (r"namespace GGL\s*\{\s*namespace MetaLearning", "namespace MetaLearning"),
    (r"namespace GGL\s*\{\s*namespace Neuromorphic", "namespace Neuromorphic"),
    (r"namespace GGL\s*\{\s*namespace QuantumBio", "namespace QuantumBio"),
    (r"namespace GGL\s*\{\s*namespace QuantumReady", "namespace QuantumReady"), # If this was also inserted wrong
]

for pattern, replacement in replacements:
    # We also need to remove one "}" at the end of these blocks.
    # This is tricky with regex.
    
    # Alternative:
    # Just replace "namespace GGL {" with "" (empty) inside the file, excluding the first one?
    pass

# Let's use a simpler logic.
# We know the specific blocks we added.
# Phase 9 code was:
# namespace GGL {
# namespace MetaLearning {
# ...
# } // namespace MetaLearning
# } // namespace GGL

# If this is inside the main GGL, we want:
# namespace MetaLearning {
# ...
# } // namespace MetaLearning

# So we just need to remove the wrapping "namespace GGL {" and "}".

def unwrap_ggl(text, ns_name):
    # Pattern: namespace GGL { namespace ns_name { ... } }
    # We want: namespace ns_name { ... }
    
    # Regex to find the start
    pattern_start = r"namespace\s+GGL\s*\{\s*namespace\s+" + ns_name
    match = re.search(pattern_start, text)
    if match:
        print(f"Found wrapped GGL for {ns_name}")
        # Replace start
        text = re.sub(pattern_start, f"namespace {ns_name}", text, count=1)
        
        # Now we need to remove the corresponding "}" at the end of this block.
        # The block ends with "} // namespace GGL" usually, if we copied the code exactly.
        # Let's look for "} // namespace GGL" that follows this block.
        
        # Actually, our inserted code had "} // namespace GGL" at the very end.
        # If we remove that line, we are good.
        
        # Let's try to remove the *last* occurrence of "} // namespace GGL" for each unwrapped block?
        # No, order matters.
        
        # Let's just replace "} // namespace GGL" with "" IF it is not the very last one in the file.
        pass
    return text

# Let's try a brute force fix for the specific phases we added recently.
# Phase 9, 10, 11.

# 1. Replace "namespace GGL { namespace MetaLearning" -> "namespace MetaLearning"
new_content = re.sub(r"namespace\s+GGL\s*\{\s*namespace\s+MetaLearning", "namespace MetaLearning", new_content)
# 2. Replace "namespace GGL { namespace Neuromorphic" -> "namespace Neuromorphic"
new_content = re.sub(r"namespace\s+GGL\s*\{\s*namespace\s+Neuromorphic", "namespace Neuromorphic", new_content)
# 3. Replace "namespace GGL { namespace QuantumBio" -> "namespace QuantumBio"
new_content = re.sub(r"namespace\s+GGL\s*\{\s*namespace\s+QuantumBio", "namespace QuantumBio", new_content)

# Now we have extra "}" characters (the ones that closed the inner GGLs).
# They are likely labeled "} // namespace GGL"
# We should remove them.
# But we must keep the FINAL one.

# Let's count how many "} // namespace GGL" are there.
closing_tags = new_content.count("} // namespace GGL")
print(f"Found {closing_tags} closing tags.")

# We want to keep ONLY ONE (the last one, or the one matching the first GGL).
# If we have nested GGLs that we just unwrapped, we have extra closing braces.
# We should replace "} // namespace GGL" with "" (empty) for all except the last one?
# Or better: replace "} // namespace GGL" with "}" if it was closing the file, but here they are extra.

# If we unwrapped the start, the closing brace "}" is now unbalanced (too many }).
# So we must remove the corresponding "}".

# Let's assume the inserted code was:
# ...
# } // namespace MetaLearning
# } // namespace GGL

# We want:
# ...
# } // namespace MetaLearning

# So we can replace "} // namespace MetaLearning\s*}\s*// namespace GGL" with "} // namespace MetaLearning"
new_content = re.sub(r"\}\s*//\s*namespace\s+MetaLearning\s*\}\s*//\s*namespace\s+GGL", "} // namespace MetaLearning", new_content)
new_content = re.sub(r"\}\s*//\s*namespace\s+Neuromorphic\s*\}\s*//\s*namespace\s+GGL", "} // namespace Neuromorphic", new_content)
new_content = re.sub(r"\}\s*//\s*namespace\s+QuantumBio\s*\}\s*//\s*namespace\s+GGL", "} // namespace QuantumBio", new_content)

# Also check QuantumReady if it was affected
new_content = re.sub(r"namespace\s+GGL\s*\{\s*namespace\s+QuantumReady", "namespace QuantumReady", new_content)
new_content = re.sub(r"\}\s*//\s*namespace\s+QuantumReady\s*\}\s*//\s*namespace\s+GGL", "} // namespace QuantumReady", new_content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Applied namespace flattening fix.")
