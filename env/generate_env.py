import os
import subprocess
import yaml

# ==== CONFIG ====
REPO_PATH = "/scratch/asr655/neuroinformatics/GeneEx2Conn"
IMPORT_FILE = os.path.join(REPO_PATH, "env", "imports.py")
OUTPUT_FILE = os.path.join(REPO_PATH, "env", "GeneEx2Conn.yml")
ENV_NAME = "GeneEx2Conn"
PYTHON_VERSION = "3.11"
CHANNELS = ["defaults"]  # Using conda defaults

# ==== STANDARD LIBRARY ====
STD_LIB = {
    'os', 'sys', 'time', 'math', 'random', 'itertools', 'collections',
    'inspect', 'gc', 're', 'ast', 'ssl', 'pickle', 'functools', 'subprocess',
    'importlib'
}

# ==== NAME MAPPING ====
CONDA_NAME_MAPPING = {
    'sklearn': 'scikit-learn',
    'GPUtil': 'gputil',
    'cv2': 'opencv',
    'PIL': 'pillow',
    'yaml': 'pyyaml',
    'torch': 'pytorch',
    'torchvision': 'torchvision',
    'umap': 'umap-learn',
    'skopt': 'scikit-optimize',
    'ipywidgets': 'ipywidgets',
    'imageio': 'imageio',
    'flash_attn': 'flash-attn',
    'mpl_toolkits': 'matplotlib',
    'matplotlib.lines': 'matplotlib',
    'matplotlib.axes': 'matplotlib',
}


# ==== Parse Imports ====
def parse_imports(file_path):
    packages = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('import '):
                line = line.replace('import ', '').split('#')[0]
                parts = [p.strip().split(' ')[0] for p in line.split(',')]
                packages.update(parts)
            elif line.startswith('from '):
                parts = line.split()
                if len(parts) >= 2:
                    packages.add(parts[1].strip())
    return sorted(p for p in packages if p and p not in STD_LIB)


# ==== Submodule Stripping ====
def strip_submodule(pkg):
    return pkg.split('.')[0]


# ==== Get Conda Version ====
def get_conda_version(package):
    try:
        result = subprocess.run(
            ["conda", "list", package],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        lines = result.stdout.splitlines()
        for line in lines:
            if line.startswith(package + ' '):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    except Exception:
        pass
    return None


# ==== Get Pip Version ====
def get_pip_version(package):
    try:
        result = subprocess.run(
            ["pip", "show", package],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ==== Check Conda ====
def is_conda_package(package):
    try:
        result = subprocess.run(
            ["conda", "search", package],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
        )
        return result.returncode == 0 and package.lower() in result.stdout.lower()
    except Exception:
        return False


# ==== Check Pip ====
def is_pip_package(package):
    try:
        result = subprocess.run(
            ["pip", "index", "versions", package],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
        )
        return "Available versions" in result.stdout
    except Exception:
        return False


# ==== Generate YAML ====
def build_environment_yaml(conda_packages, pip_packages):
    env = {
        'name': ENV_NAME,
        'channels': CHANNELS,
        'dependencies': [
            f'python={PYTHON_VERSION}'
        ] + sorted(set(conda_packages))
    }

    if pip_packages:
        env['dependencies'].append({'pip': sorted(set(pip_packages))})

    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(env, f, sort_keys=False)

    print(f"\n✅ GeneEx2Conn.yml created at {OUTPUT_FILE}")


# ==== MAIN ====
if __name__ == "__main__":
    print(f"🔍 Parsing imports from {IMPORT_FILE} ...")
    raw_packages = parse_imports(IMPORT_FILE)
    print(f"➡️ Raw packages parsed: {raw_packages}")

    stripped_packages = set(strip_submodule(pkg) for pkg in raw_packages)
    print(f"➡️ Stripped package names (deduplicated): {sorted(stripped_packages)}")

    conda_packages = []
    pip_packages = []
    missing_packages = []

    for pkg in sorted(stripped_packages):
        mapped_pkg = CONDA_NAME_MAPPING.get(pkg, pkg)
        print(f"➡️ Checking {pkg} (mapped as {mapped_pkg})...")

        if is_conda_package(mapped_pkg):
            version = get_conda_version(mapped_pkg)
            if version:
                print(f"   ✅ Found in conda defaults (version {version})")
                conda_packages.append(f"{mapped_pkg}={version}")
            else:
                print(f"   ✅ Found in conda defaults (version unknown)")
                conda_packages.append(mapped_pkg)
        elif is_pip_package(pkg):
            version = get_pip_version(pkg)
            if version:
                print(f"   ✅ Found in pip (version {version})")
                pip_packages.append(f"{pkg}=={version}")
            else:
                print(f"   ✅ Found in pip (version unknown)")
                pip_packages.append(pkg)
        else:
            print(
                f"\n❌ Package '{pkg}' not found in conda defaults or pip.\n"
                f"→ Please check the package name or install method.\n"
            )
            missing_packages.append(pkg)

    build_environment_yaml(conda_packages, pip_packages)

    if missing_packages:
        print("\n⚠️ The following packages were not found in conda defaults or pip:")
        for m in missing_packages:
            print(f"   - {m}")
        print("\n🛑 Please update CONDA_NAME_MAPPING or check package availability.\n")
    else:
        print("\n✅ All packages accounted for.\n")
