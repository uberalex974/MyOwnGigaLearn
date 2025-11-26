"""FIX: Nested GGL namespace issue"""

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix all nested namespace GGL issues
content = content.replace('namespace GGL {\nnamespace GGL {', 'namespace GGL {')
content = content.replace('} // namespace GGL\n} // namespace GGL', '} // namespace GGL')

# Remove duplicate namespace declarations
lines = content.split('\n')
cleaned_lines = []
in_ggl_namespace = False
ggl_count = 0

for line in lines:
    if 'namespace GGL {' in line and not in_ggl_namespace:
        cleaned_lines.append(line)
        in_ggl_namespace = True
        ggl_count += 1
    elif 'namespace GGL {' in line and in_ggl_namespace:
        # Skip duplicate opening
        continue
    elif '} // namespace GGL' in line:
        ggl_count -= 1
        if ggl_count == 0:
            cleaned_lines.append(line)
            in_ggl_namespace = False
        # else skip duplicate closing
    else:
        cleaned_lines.append(line)

content = '\n'.join(cleaned_lines)

with open(r'c:\Giga\GigaLearnCPP\GigaLearnCPP\src\private\GigaLearnCPP\PPO\PPOLearner.h', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed nested namespace GGL issue")
print("  - Removed duplicate namespace declarations")
print("  - Cleaned structure")
