import subprocess
import sys
import os
import shutil
import importlib


def detect_package_manager():
    """Detect the available package manager"""
    package_managers = {
        "apt": "/usr/bin/apt",
        "yum": "/usr/bin/yum",
        "dnf": "/usr/bin/dnf",
        "pacman": "/usr/bin/pacman",
        "zypper": "/usr/bin/zypper",
    }

    for pm, path in package_managers.items():
        if os.path.exists(path):
            return pm
    return None


def check_package_installed(package_name, package_manager = None):
    """Check if a package is installed using the system package manager"""

    if package_manager is None:
        package_manager = detect_package_manager()

    if package_manager is None:
        print("Warning: Could not detect package manager")
        return None

    try:
        if package_manager == "apt":
            # Check with dpkg
            result = subprocess.run(
                ["dpkg", "-l", package_name], capture_output = True, text = True
            )
            return result.returncode == 0

        elif package_manager in ["yum", "dnf"]:
            # Check with rpm
            result = subprocess.run(
                ["rpm", "-q", package_name], capture_output = True, text = True
            )
            return result.returncode == 0

        elif package_manager == "pacman":
            result = subprocess.run(
                ["pacman", "-Q", package_name], capture_output = True, text = True
            )
            return result.returncode == 0

        elif package_manager == "zypper":
            result = subprocess.run(
                ["zypper", "se", "-i", package_name], capture_output = True, text = True
            )
            return package_name in result.stdout

    except Exception as e:
        print(f"Error checking package: {e}")
        return None


def require_package(package_name, executable_name = None):
    """Require a package to be installed, exit if not found"""

    # First check if executable is in PATH (most reliable)
    if executable_name:
        if shutil.which(executable_name):
            print(f"✓ {executable_name} is available")
            return

    # Then check with package manager
    pm = detect_package_manager()
    is_installed = check_package_installed(package_name, pm)

    if is_installed:
        print(f"✓ Package {package_name} is installed")
        return

    # Package not found - show installation instructions
    print(f"❌ Error: {package_name} is not installed")
    print(f"\nPlease install {package_name} using your system package manager:")

    install_commands = {
        "apt": f"sudo apt update && sudo apt install {package_name}",
        "yum": f"sudo yum install {package_name}",
        "dnf": f"sudo dnf install {package_name}",
        "pacman": f"sudo pacman -S {package_name}",
        "zypper": f"sudo zypper install {package_name}",
    }

    if pm and pm in install_commands:
        print(f"  {install_commands[pm]}")
    else:
        for pm_name, cmd in install_commands.items():
            print(f"  {pm_name}: {cmd}")

    print(f"\nAlternatively, install with conda:")
    print(f"  conda install -c conda-forge {package_name}")

    print(f"\nPlease install the required package and run the script again.")
    sys.exit(1)


# Usage
# require_package("ffmpeg", "ffmpeg")


def require_python_package(package_name, import_name = None, pip_name = None):
    """Require a Python package to be installed, exit if not found"""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name

    if importlib.util.find_spec(import_name) is None:
        print(f"❌ Error: Python package '{package_name}' is not installed")
        print(f"\nPlease install {package_name} using pip:")
        print(f"  pip install {pip_name}")
        print(f"  # or with conda:")
        print(f"  conda install {pip_name}")
        print(f"\nAfter installation, run this script again.")
        sys.exit(1)
    else:
        print(f"✓ Python package '{package_name}' is installed")
