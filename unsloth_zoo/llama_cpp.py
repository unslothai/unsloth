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

COMMANDS_NOT_FOUND = ("command not found", "not found", "No such file or directory",)


def install_package(package, sudo = False, print_output = False, print_outputs = None):
    # Code licensed under LGPL
    x = f"{'sudo ' if sudo else ''}apt-get install {package} -y"
    print(f"Unsloth: Installing packages: {package}")
    with subprocess.Popen(x, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()

            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line:
                raise RuntimeError(f"*** Unsloth: Permission denied when installing package {package}")
            elif line.endswith(COMMANDS_NOT_FOUND):
                raise RuntimeError(f"*** Unsloth: apt-get does not exist when installing {package}? Is this NOT a Linux / Mac based computer?")
            elif "Unable to locate package" in line:
                raise RuntimeError(f"*** Unsloth: Could not install package {package} since it does not exist.")
            if print_output: print(line, flush = True, end = "")
            if print_outputs is not None: print_outputs.append(line)
        pass
    pass
pass


def do_we_need_sudo():
    # Code licensed under LGPL
    # Check apt-get updating
    sudo = False
    x = "apt-get update -y"
    print("Unsloth: Updating system package directories")
    with subprocess.Popen(x, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()
            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line:
                sudo = True
                break
            elif line.endswith(COMMANDS_NOT_FOUND):
                raise RuntimeError("*** Unsloth: apt-get does not exist? Is this NOT a Linux / Mac based computer?")
            pass
        pass
    pass
    # Update all packages as well
    x = f"sudo apt-get update -y"
    with subprocess.Popen(x, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()
            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line:
                raise RuntimeError("*** Unsloth: Tried with sudo, but still failed?")
        pass
    pass
    if sudo: print("Unsloth: All commands will now use admin permissions (sudo)")
    return sudo
pass


def check_pip():
    # Code licensed under LGPL
    pip = "pip"
    with subprocess.Popen(pip, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            if line.decode("utf-8", errors = "replace").rstrip().endswith(COMMANDS_NOT_FOUND):
                pip = None
                break
        pass
    pass
    if pip is not None: return "pip"
    with subprocess.Popen("pip3", shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            if line.decode("utf-8", errors = "replace").rstrip().endswith(COMMANDS_NOT_FOUND):
                raise RuntimeError("*** Unsloth: pip or pip3 not found!")
                break
        pass
    pass
    return "pip3"
pass


def try_execute(command, sudo = False, print_output = False, print_outputs = None):
    # Code licensed under LGPL
    need_to_install = False
    with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace")
            if line.rstrip().endswith(COMMANDS_NOT_FOUND):
                need_to_install = True
            elif "undefined reference" in line or "Unknown argument" in line or "***" in line:
                raise RuntimeError(f"*** Unsloth: Failed executing command [{command}] with error [{line}]. Please report this ASAP!")
            if print_output: print(line, flush = True, end = "")
            if print_outputs is not None: print_outputs.append(line)
        pass
    pass
    if need_to_install:
        install_package(command.split(" ", 1)[0], sudo)
        try_execute(command, sudo, print_output)
    pass
pass


def install_llama_cpp(
    llama_cpp_folder = "llama.cpp",
    # llama.cpp specific targets - all takes 90s. Below takes 60s
    llama_cpp_targets = ["llama-quantize", "llama-export-lora", "llama-cli",],
    print_output = False,
):
    if os.path.exists(llama_cpp_folder):
        files = os.listdir()
        while llama_cpp_folder in files:
            llama_cpp_folder = llama_cpp_folder + "_"
        pass
    pass

    print_outputs = []
    sudo = do_we_need_sudo()
    try:
        try_execute(
            f"git clone https://github.com/ggerganov/llama.cpp {llama_cpp_folder}",
            sudo = sudo,
            print_output  = print_output,
            print_outputs = print_outputs,
        )
        install_package("build-essential cmake curl libcurl4-openssl-dev", sudo)
        try_execute(
            f"cmake {llama_cpp_folder} -B {llama_cpp_folder}/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=ON",
            sudo = sudo,
            print_output  = print_output,
            print_outputs = print_outputs,
        )
        pip = check_pip()
        try_execute(
            f"{pip} install gguf protobuf sentencepiece",
            sudo = False,
            print_output  = print_output,
            print_outputs = print_outputs,
        )
    except Exception as error:
        print("="*30)
        print("=== Unsloth: FAILED installing llama.cpp ===")
        print(f"=== Main error = {str(error)} ===")
        print("=== Error log below: ===")
        print("".join(print_outputs))
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
