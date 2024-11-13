#!/bin/bash

# Set the environment name
ENV_NAME="GeneEx2Conn"

# Default setting for removing problematic dependencies
REMOVE_DEPS=""

# Parse the input argument to specify problematic dependencies to exclude
# USAGE: bash create_env.sh remove_deps="tensorflow,enigma,tflow"
for arg in "$@"; do
    if [[ "$arg" == remove_deps=* ]]; then
        REMOVE_DEPS="${arg#*=}"
    fi
done

# Check if the specified Conda environment is active; if not, activate it
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Activating Conda environment: $ENV_NAME"
    conda activate "$ENV_NAME"
fi

# Step 1: Export only the explicitly used packages from the Conda environment to a temporary YAML file
# This file includes only packages installed by the user (without unnecessary dependencies)
conda_history_file="GeneEx2Conn_used_packages.yml"
echo "Exporting Conda environment history to $conda_history_file..."
conda env export --from-history > "$conda_history_file"

# Step 2: Create a pip requirements file containing only installed packages, without file-based references
# This step creates a clean list of pip-installed packages for the environment
pip_requirements_file="requirements_clean.txt"
echo "Generating pip requirements file to $pip_requirements_file..."
pip freeze | grep -v '@ file' > "$pip_requirements_file"

# Step 3: Combine the Conda and Pip files into a single comprehensive environment YAML
# The final output will be saved to `GeneEx2Conn_env_combined.yml`
combined_yml="GeneEx2Conn_env_combined.yml"
echo "Combining Conda and Pip dependencies into $combined_yml..."

# Prepare a sed exclusion pattern based on the problematic dependencies
exclude_pattern=""
IFS=',' read -ra DEPS <<< "$REMOVE_DEPS"
for dep in "${DEPS[@]}"; do
    exclude_pattern+="; /$dep/d"
done

# Copy Conda dependencies from the temporary YAML file, excluding:
# - Any existing `- pip:` section to avoid duplication
# - Any `prefix` line (this will be added at the end once)
# - Lines matching problematic dependencies specified in REMOVE_DEPS
if [[ -n "$exclude_pattern" ]]; then
    sed "/^ *- pip:/Q; /^prefix:/d${exclude_pattern}" "$conda_history_file" > "$combined_yml"
else
    sed '/^ *- pip:/Q; /^prefix:/d' "$conda_history_file" > "$combined_yml"
fi

# Add a pip section header to the final YAML file for including pip-specific dependencies
echo "  - pip:" >> "$combined_yml"

# Add each pip package from `requirements_clean.txt` with YAML indentation
# Skip lines containing any problematic dependencies specified in REMOVE_DEPS
while read -r line; do
    skip_line=false
    for dep in "${DEPS[@]}"; do
        if [[ "$line" == *"$dep"* ]]; then
            skip_line=true
            break
        fi
    done
    if [ "$skip_line" = true ]; then
        continue  # Skip this line if it contains any problematic dependency
    fi
    echo "      - $line" >> "$combined_yml"
done < "$pip_requirements_file"

# Append a single `prefix` line at the end of the combined YAML
# The prefix defines the location where the environment should be installed
echo "prefix: /ext3/miniconda3" >> "$combined_yml"

# Step 4: Clean up intermediate files used for combining dependencies
# Remove the temporary Conda YAML and pip requirements files to keep the workspace clean
echo "Cleaning up intermediate files..."
rm "$conda_history_file" "$pip_requirements_file"

echo "Final combined environment YAML created as $combined_yml"
