import os
import re

# Directory to search
project_dir = "/Users/moyuanchen/Documents/thesis/collusion"

# Regex to match import statements
import_pattern = re.compile(r'^\s*(?:import|from)\s+([\w\.]+)')

# Collect all imports
all_imports = set()

for root, _, files in os.walk(project_dir):
    for file in files:
        if file.endswith(".py"):  # Only process Python files
            with open(os.path.join(root, file), "r") as f:
                for line in f:
                    match = import_pattern.match(line)
                    if match:
                        module = match.group(1).split('.')[0]  # Get the top-level module
                        all_imports.add(module)

# Compare with requirements.txt
requirements_path = os.path.join(project_dir, "requirements.txt")
with open(requirements_path, "r") as f:
    requirements = {line.split("==")[0].strip() for line in f if "==" in line}

# Print results
print("Modules in code but not in requirements.txt:")
print(all_imports - requirements)

print("\nModules in requirements.txt but not in code:")
print(requirements - all_imports)