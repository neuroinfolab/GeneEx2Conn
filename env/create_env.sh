#!/bin/bash

# Set the environment name
ENV_NAME="GeneEx2Conn"

# Check if the Conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Activating Conda environment: $ENV_NAME"
    conda activate "$ENV_NAME"
fi

# Step 1: Export used packages from Conda history to a YAML file
conda_history_file="GeneEx2Conn_used_packages.yml"
echo "Exporting Conda environment history to $conda_history_file..."
conda env export --from-history > "$conda_history_file"

# Step 2: Create a Pip requirements file with installed packages
pip_requirements_file="requirements_clean.txt"
echo "Generating pip requirements file to $pip_requirements_file..."
pip freeze | grep -v '@ file' > "$pip_requirements_file"

# Step 3: Combine the Conda and Pip files into a single environment YAML
combined_yml="GeneEx2Conn_env_combined.yml"
echo "Combining Conda and Pip dependencies into $combined_yml..."

# Copy Conda part up to but not including any existing pip section
sed '/^ *- pip:/Q' "$conda_history_file" > "$combined_yml"

# Add the pip section header
echo "  - pip:" >> "$combined_yml"

# Add each pip package from requirements_clean.txt with YAML indentation
while read -r line; do
    echo "      - $line" >> "$combined_yml"
done < "$pip_requirements_file"

# Append the prefix line from the original Conda YAML file
grep "^prefix:" "$conda_history_file" >> "$combined_yml"

# Step 4: Clean up intermediate files
echo "Cleaning up intermediate files..."
rm "$conda_history_file" "$pip_requirements_file"

echo "Final combined environment YAML created as $combined_yml"