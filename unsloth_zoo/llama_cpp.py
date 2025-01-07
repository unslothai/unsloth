# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "install_llama_cpp",
]

import subprocess
import sys
import os
import time
import psutil
import re
import requests

LLAMA_CPP_CONVERT_FILE = \
    "https://github.com/ggerganov/llama.cpp/raw/refs/heads/master/convert_hf_to_gguf.py"

COMMANDS_NOT_FOUND = (
    "command not found",
    "not found",
    "No such file or directory",
)

# llama.cpp specific targets - all takes 90s. Below takes 60s
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    "llama-export-lora",
    "llama-cli",
    "llama-llava-cli",
    "llama-gguf-split",
]

PIP_OPTIONS = [
    "pip",
    "pip3",
    "python3 -m pip", # Python standalone installation
    "py -m pip", # Windows
    "uv pip", # Astral's uv
    "poetry", # Poetry
]

BAD_OUTCOMES = {
    "undefined reference"        : "Please report this ASAP!",
    "Unknown argument"           : "Please report this ASAP!",
    "[FAIL]"                     : "Please report this ASAP!",
    "--break-system-packages"    : "You need to redo the command manually with elevated permissions.",
    "establish a new connection" : "You do not have internet connection!",
    "fatal: unable to access"    : "You do not have internet connection!",
    "failure resolving"          : "You do not have internet connection!",
    "fatal "                     : "",
    "Err:"                       : "",
    "Failed "                    : "",
    "is deprecated"              : "Command is deprecated!",
}

def install_package(package, sudo = False, print_output = False, print_outputs = None):
    # All Unsloth Zoo code licensed under LGPLv3
    x = f"{'sudo ' if sudo else ''}apt-get install {package} -y"
    print(f"Unsloth: Installing packages: {package}")
    with subprocess.Popen(x, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()

            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line or "fatal" in line:
                raise RuntimeError(f"[FAIL] Unsloth: Permission denied when installing package {package}")
            elif line.endswith(COMMANDS_NOT_FOUND):
                raise RuntimeError(f"[FAIL] Unsloth: apt-get does not exist when installing {package}? Is this NOT a Linux / Mac based computer?")
            elif "Unable to locate package" in line:
                raise RuntimeError(f"[FAIL] Unsloth: Could not install package {package} since it does not exist.")
            if print_output: print(line, flush = True, end = "")
            if print_outputs is not None: print_outputs.append(line)
        pass
    pass
pass


def do_we_need_sudo():
    # All Unsloth Zoo code licensed under LGPLv3
    # Check apt-get updating
    sudo = False
    print("Unsloth: Updating system package directories")

    x = "apt-get update -y"

    start_time = time.time()
    with subprocess.Popen(x, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()
            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line or "fatal" in line:
                sudo = True
                break
            elif line.endswith(COMMANDS_NOT_FOUND):
                raise RuntimeError("[FAIL] Unsloth: apt-get does not exist? Is this NOT a Linux / Mac based computer?")
            elif "failure resolving" in line or "Err:" in line:
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
            elif time.time() - start_time >= 180:
                # Failure if longer than 3 minutes
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
        pass
    pass

    # Update all package lists as well
    x = f"sudo apt-get update -y"

    start_time = time.time()
    with subprocess.Popen(x, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()
            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line or "fatal" in line:
                raise RuntimeError("[FAIL] Unsloth: Tried with sudo, but still failed?")
            elif "failure resolving" in line or "Err:" in line:
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
            elif time.time() - start_time >= 180:
                # Failure if longer than 3 minutes
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
        pass
    pass

    if sudo: print("Unsloth: All commands will now use admin permissions (sudo)")
    return sudo
pass


def check_pip():
    # All Unsloth Zoo code licensed under LGPLv3
    for pip in PIP_OPTIONS:
        final_pip = pip
        with subprocess.Popen(pip, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
            for line in sp.stdout:
                if line.decode("utf-8", errors = "replace").rstrip().endswith(COMMANDS_NOT_FOUND):
                    final_pip = None
                    break
            pass
        pass
        if final_pip is not None: return final_pip
    pass
    raise RuntimeError(f"[FAIL] Unsloth: Tried all of `{', '.join(PIP_OPTIONS)}` but failed.")
pass


def try_execute(command, sudo = False, print_output = False, print_outputs = None):
    # All Unsloth Zoo code licensed under LGPLv3
    need_to_install = False
    with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace")
            if line.rstrip().endswith(COMMANDS_NOT_FOUND):
                need_to_install = True

            error_msg = f"[FAIL] Unsloth: Failed executing command `[{command}]` with error `[{line}]`.\n"
            
            for key, value in BAD_OUTCOMES.items():
                if key in line:
                    raise RuntimeError(error_msg + value)
            pass

            if print_output: print(line, flush = True, end = "")
            if print_outputs is not None: print_outputs.append(line)
        pass
    pass
    if need_to_install:
        install_package(command.split(" ", 1)[0], sudo)
        try_execute(command, sudo, print_output)
    pass
pass


def check_llama_cpp(llama_cpp_folder = "llama.cpp"):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check PATH and main directory
    system_directories = [os.getcwd()] + list(os.environ.get("PATH").split(os.pathsep))

    partial_outputs = []

    # Check llama-quantize
    quantizer_location = None
    converter_location = None
    saved_error = None

    for directory in system_directories:
        quantizer_location = None
        converter_location = None
        try:
            # Check llama.cpp/llama-quantize binary file
            for quantizer in ["llama-quantize", "quantize"]:
                location = os.path.join(llama_cpp_folder, quantizer)
                if os.path.exists(location) and os.access(location, os.X_OK):
                    try:
                        try_execute(
                            f"./{location} --help",
                            sudo = False,
                            print_output = False,
                            print_outputs = partial_outputs,
                        )
                        quantizer_location = location
                        break
                    except: pass
                pass
            pass
            if quantizer_location is None:
                error_log = '\n'.join(partial_outputs)
                raise RuntimeError(
                    f"Unsloth: Failed to run `{quantizer}` - please re-compile llama.cpp!\n"\
                    f"Error log:\n{error_log}"
                )
            pass

            # Check convert_hf_to_gguf.py file
            for converter in ["convert-hf-to-gguf.py", "convert_hf_to_gguf.py"]:
                location = os.path.join(llama_cpp_folder, converter)
                if os.path.exists(location):
                    converter_location =  location
                    break
            pass
            if converter_location is None:
                raise RuntimeError(f"Unsloth: Failed to find `{converter}` - please re-compile llama.cpp!")
            pass
        except Exception as error:
            saved_error = str(error)
            pass

        if quantizer_location is not None and converter_location is not None:
            return quantizer_location, converter_location
    pass
    raise RuntimeError(saved_error)
pass


def get_latest_supported_models(llama_cpp_folder = "llama.cpp"):
    # All Unsloth Zoo code licensed under LGPLv3
    # Gets all model config names like LlamaForCasualLM that are supported by llama.cpp
    try:
        # Try getting llama.cpp folder
        quantizer, converter = check_llama_cpp(llama_cpp_folder = llama_cpp_folder)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            name = "llama_cpp_module",
            location = converter,
        )
        llama_cpp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llama_cpp_module)
        supported_types = frozenset(llama_cpp_module.Model._model_classes.keys())
    except:
        # Instead get it from the latest llama.cpp Github repo
        converter_latest = requests.get(LLAMA_CPP_CONVERT_FILE).content
        supported_types = re.findall(rb"@Model\.register\(([^)]{1,})\)", converter_latest)
        supported_types = b", ".join(supported_types).decode("utf-8")
        supported_types = re.findall(r"[\'\"]([^\'\"]{1,})[\'\"]", supported_types)
        supported_types = frozenset(supported_types)
    pass
    return supported_types
pass


def install_llama_cpp(
    llama_cpp_folder = "llama.cpp",
    llama_cpp_targets = LLAMA_CPP_TARGETS,
    print_output = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Installs llama.cpp
    quantizer = None
    converter = None

    if os.path.exists(llama_cpp_folder):
        try:
            quantizer, converter = check_llama_cpp(llama_cpp_folder = llama_cpp_folder)
            print(f"Unsloth: llama.cpp folder already exists - will use `{llama_cpp_folder}`")
        except: pass
    pass
    if quantizer is not None and converter is not None: return quantizer, converter

    print_outputs = []
    sudo = do_we_need_sudo()
    kwargs = {"sudo" : sudo, "print_output" : print_output, "print_outputs" : print_outputs,}
    cpu_count = psutil.cpu_count() + 2

    try:
        try_execute(f"git clone https://github.com/ggerganov/llama.cpp {llama_cpp_folder}", **kwargs)
        
        install_package("build-essential cmake curl libcurl4-openssl-dev", sudo)

        pip = check_pip()
        kwargs["sudo"] = False

        print("Unsloth: Install GGUF and other packages")
        try_execute(f"{pip} install gguf protobuf sentencepiece", **kwargs)

        print("Unsloth: Install llama.cpp and building - please wait 1 to 3 minutes")
        try:
            # Try using make first
            try_execute(f"make clean -C llama.cpp", **kwargs)
            try_execute(f"make all -j{cpu_count} -C llama.cpp", **kwargs)
        except:
            # Use cmake instead
            try_execute(
                f"cmake {llama_cpp_folder} -B {llama_cpp_folder}/build "\
                "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=ON",
                **kwargs
            )
            try_execute(
                f"cmake --build {llama_cpp_folder}/build --config Release "\
                f"-j{cpu_count} --clean-first --target "\
                f"{' '.join(llama_cpp_targets)}",
                **kwargs
            )
            # Move compiled objects to main folder
            try_execute(
                f"cp {llama_cpp_folder}/build/bin/llama-* "\
                f"{llama_cpp_folder}",
                **kwargs
            )
            # Remove build folder
            try_execute(f"rm -rf {llama_cpp_folder}/build", **kwargs)
        pass
            
    except Exception as error:
        print("="*30)
        print("=== Unsloth: FAILED installing llama.cpp ===")
        print(f"=== Main error = {str(error)} ===")
        print("=== Error log below: ===")
        print("".join(print_outputs))
    pass

    # Check if it installed correctly
    quantizer, converter = check_llama_cpp(llama_cpp_folder)
    return quantizer, converter
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
