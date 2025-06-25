# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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
import json
from tqdm.auto import tqdm as ProgressBar
from functools import lru_cache
import inspect
import contextlib
import os

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
                sp.terminate()
                raise RuntimeError(f"[FAIL] Unsloth: Permission denied when installing package {package}")
            elif line.endswith(COMMANDS_NOT_FOUND):
                sp.terminate()
                raise RuntimeError(f"[FAIL] Unsloth: apt-get does not exist when installing {package}? Is this NOT a Linux / Mac based computer?")
            elif "Unable to locate package" in line:
                sp.terminate()
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
                sp.terminate()
                sudo = True
                break
            elif line.endswith(COMMANDS_NOT_FOUND):
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: apt-get does not exist? Is this NOT a Linux / Mac based computer?")
            elif "failure resolving" in line or "Err:" in line:
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
            elif time.time() - start_time >= 180:
                # Failure if longer than 3 minutes
                sp.terminate()
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
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: Tried with sudo, but still failed?")
            elif "failure resolving" in line or "Err:" in line:
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
            elif time.time() - start_time >= 180:
                # Failure if longer than 3 minutes
                sp.terminate()
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
                    sp.terminate()
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
                    sp.terminate()
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


def install_llama_cpp(
    llama_cpp_folder = "llama.cpp",
    llama_cpp_targets = LLAMA_CPP_TARGETS,
    print_output = False,
    gpu_support = False,
    just_clone_repo = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Installs llama.cpp
    quantizer = None
    converter = None

    gpu_support = "ON" if gpu_support else "OFF"

    if os.path.exists(llama_cpp_folder):
        try:
            quantizer, converter = check_llama_cpp(llama_cpp_folder = llama_cpp_folder)
            print(f"Unsloth: llama.cpp folder already exists - will use `{llama_cpp_folder}`")
        except: pass
    pass
    if quantizer is not None and converter is not None: return quantizer, converter

    print_outputs = []
    sudo = do_we_need_sudo()
    sudo = False
    kwargs = {"sudo" : sudo, "print_output" : print_output, "print_outputs" : print_outputs,}

    try:
        try_execute(f"git clone https://github.com/ggml-org/llama.cpp {llama_cpp_folder}", **kwargs)

        pip = check_pip()
        kwargs["sudo"] = False

        print("Unsloth: Install GGUF and other packages")
        try_execute(f"{pip} install gguf protobuf sentencepiece", **kwargs)
        if just_clone_repo: return llama_cpp_folder

        install_package("build-essential cmake curl libcurl4-openssl-dev", sudo)

        print("Unsloth: Install llama.cpp and building - please wait 1 to 3 minutes")
        if gpu_support == "ON":
            print("Unsloth: Building llama.cpp with GPU support")
        try:
            # Try using make first
            try_execute(f"make clean -C llama.cpp", **kwargs)
            try_execute(f"make all -j -C llama.cpp", **kwargs)
        except:
            # Use cmake instead
            try_execute(
                f"cmake {llama_cpp_folder} -B {llama_cpp_folder}/build "\
                f"-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA={gpu_support} -DLLAMA_CURL=ON",
                **kwargs
            )
            try_execute(
                f"cmake --build {llama_cpp_folder}/build --config Release "\
                f"-j --clean-first --target "\
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


@lru_cache(1)
def _download_convert_hf_to_gguf(
    name = "unsloth_convert_hf_to_gguf",
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Downloads from llama.cpp's Github repo
    try:
        converter_latest = requests.get(LLAMA_CPP_CONVERT_FILE).content
    except:
        raise RuntimeError(
            f"Unsloth: Could not obtain `{LLAMA_CPP_CONVERT_FILE}`.\n"\
            f"Maybe you don't have internet ocnnection?"
        )

    # Get all supported models
    supported_types = re.findall(rb"@(?:Model|ModelBase)\.register\(\s*([^)]+?)\s*\)",converter_latest)
    supported_types = b", ".join(supported_types).decode("utf-8")
    supported_types = re.findall(r"[\'\"]([^\'\"]{1,})[\'\"]", supported_types)
    supported_types = frozenset(supported_types)

    # Sometimes gguf.x cannot be found!
    archs = list(set(re.findall(rb"[\n\s]gguf\.([\.A-Z\_0-9]{3,})[\n\s\,]", converter_latest)))
    archs = [x.decode("utf-8") for x in archs]
    all_edits = "\n\n".join(
        f"try: gguf.{x}\nexcept: gguf.{x} = None"
        for x in archs
    ).encode("utf-8")

    # Make main() become main(args)
    changes = [
        (b"import gguf", b"import gguf\n" + all_edits,),
        # (b"def main()",  b"def main(args)",),
        # (b"args = parse_args()", b"",),
    ]
    for old, new in changes:
        if old not in converter_latest:
            raise RuntimeError(
                f"Unsloth: Could not patch `{old}` - Report immediately as a bug - llama.cpp is broken!"
            )
        converter_latest = converter_latest.replace(old, new, 1)
    pass

    # Fix metadata
    converter_latest = re.sub(
        rb"(self\.metadata \= .+?\(.+?\)"\
        rb"[\n]{1,}([\s]{4,}))",
        rb"\1"\
        rb"if hasattr(self.metadata, 'quantized_by'): self.metadata.quantized_by = 'Unsloth'\n"\
        rb"\2if hasattr(self.metadata, 'repo_url'): self.metadata.repo_url = 'https://huggingface.co/unsloth'\n"\
        rb"\2if hasattr(self.metadata, 'tags'): self.metadata.tags = ['unsloth', 'llama.cpp']\n"\
        rb"\2",
        converter_latest,
    )

    # Write file
    with open(f"llama.cpp/{name}.py", "wb") as file:
        file.write(converter_latest)
    filename = f"llama.cpp/{name}.py"

    # Get all flags in parser
    flags = re.findall(
        rb"parser\.add_argument\([\s]{4,}[\"\']([^\"\']{1,})[\'\"]", converter_latest,
    )
    if len(flags) == 0:
        raise RuntimeError("Unsloth: Failed parsing convert_hf_to_gguf.py with no flags found.")

    # Get defaults
    defaults = re.findall(
        rb"parser\.add_argument\([\s]{4,}[\"\']([^\"\']{1,})[\'\"]"\
        rb"[^\)]{1,}(?:action|default)[\s\=]{1,}([^\s\,]{1,})", converter_latest,
    )
    all_flags = {}
    for flag, default in defaults:
        flag = flag.decode("utf-8")
        if flag.startswith("--"): flag = flag[2:]
        flag = flag.replace("-", "_")

        default = eval(default.decode("utf-8"))
        if   default == "store_true":  default = True
        elif default == "store_false": default = False
        all_flags[flag] = default
    pass

    # Rest of flags
    rest_flags = []
    for flag in flags:
        flag = flag.decode("utf-8")
        if flag.startswith("--"): flag = flag[2:]
        flag = flag.replace("-", "_")
        if flag not in all_flags:
            rest_flags.append(flag)
    pass

    for flag in ["outfile", "model"]:
        if flag not in rest_flags:
            raise RuntimeError(f"Unsloth: Failed parsing convert_hf_to_gguf.py with no `{flag}` found.")
        else: rest_flags = [x for x in rest_flags if x != flag]
    pass

    # Rest are just None
    for flag in rest_flags: all_flags[flag] = None

    # Check mandatory flags:
    for flag in ["outtype", "split_max_size", "dry_run"]:
        if flag not in all_flags:
            raise RuntimeError(f"Unsloth: Failed parsing convert_hf_to_gguf.py with no `{flag}` found.")
    pass
    return filename, supported_types
pass


def _split_str_to_n_bytes(split_str: str) -> int:
    # All Unsloth Zoo code licensed under LGPLv3
    # Converts 50G to bytes
    if split_str.endswith("K"):
        n = float(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = float(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = float(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = float(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n
pass


def _convert_to_gguf(command, output_filename, print_output = False, print_outputs = None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Filter warnings / errors with dates
    import datetime
    datetime = datetime.datetime.today().strftime("%Y-%m-%d")

    popen = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        universal_newlines = True,
        shell = True,
    )
    ProgressBar._instances.clear()

    progress_bar = None
    chat_template_line = 0
    stop_chat_template = False
    metadata = {}

    for line in iter(popen.stdout.readline, ""):
        if line.startswith("Writing:"):
            if progress_bar is None:
                progress_bar = ProgressBar(total = 100, position = 0, leave = True, desc = "Unsloth: GGUF conversion")

            desc = re.findall(r"([\d]{1,3})\%.+?([\d\.].+?\])", line)
            if len(desc) == 1 and len(desc[0]) == 2:
                percentage, info = desc[0]
                progress_bar.update(int(percentage) - progress_bar.n)
                info = re.findall(r"([\d\.]{1,}(?:K|M|G)\/[\d\.]{1,}(?:K|M|G))", info)
                if len(info) != 0: progress_bar.set_postfix_str(info[0])
                continue
            pass

        elif line.startswith("INFO:gguf.gguf_writer") and "total_size = " in line:
            # Get name of file as well
            name = re.findall(r"INFO:gguf\.gguf_writer:([^\:]{1,})\:", line)
            if len(name) == 1:
                name = name[0]
                # Save final size of model
                x = re.findall(r"total_size = ([\d\.]{1,}(?:K|M|G))", line)
                if len(x) == 1:
                    try:
                        total_size = _split_str_to_n_bytes(x[0])
                    except Exception as error:
                        popen.terminate()
                        raise RuntimeError(error)
                    metadata[name] = (total_size, x[0],)
                pass
            pass

        elif line.startswith((datetime, "WARNING:", "INFO:numexpr")):
            # Skip warnings / errors
            continue

        elif line.startswith("INFO:hf-to-gguf:blk"):
            # Skip showcasing conversions - unnecessary
            continue

        elif line.startswith("INFO:gguf.vocab:Setting chat_template"):
            # Do not print super long chat templates - allow 5 lines
            chat_template_line = 1

        if chat_template_line != 0: chat_template_line += 1

        if chat_template_line >= 10:
            # Restart if possible
            if line.startswith("INFO:hf-to-gguf:"):
                chat_template_line = 0
            else:
                if not stop_chat_template:
                    print("..... Chat template truncated .....\n")
                stop_chat_template = True
                continue
            pass
        pass

        # Fix up start of strings
        if line.startswith("INFO:"): line = "Unsloth GGUF:" + line[len("INFO:"):]

        if print_output: print(line, flush = True, end = "")
        if print_outputs is not None: print_outputs.append(line)
    pass

    if progress_bar is not None: progress_bar.close()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    pass

    # Check final size approximately
    if len(metadata) != 0:
        for output_filename, (total_size, x,) in metadata.items():
            actual_size = os.path.getsize(output_filename)

            ratio = actual_size / total_size
            if ratio <= 0.9 or ratio >= 1.1:
                raise RuntimeError(
                    "Unsloth: Failed converting to GGUF since we do not have enough disk space!\n"\
                    f"We need {total_size} bytes but we managed to find only {actual_size} bytes!"
                )
            pass

            line = f"Unsloth: Converted to {output_filename} with size = {x}\n"
            if print_output: print(line, flush = True, end = "")
            if print_outputs is not None: print_outputs.append(line)
        pass
    else:
        raise RuntimeError(
            "Unsloth: Failed converting to GGUF since we did not create an GGUF files?"
        )
    return list(metadata.keys())
pass


def check_quantization_type(quantization_type = "Q8_0"):
    # All Unsloth Zoo code licensed under LGPLv3
    # Gets quantization and multiplier
    assert(type(quantization_type) is str)
    quantization_type = quantization_type.lower()
    SUPPORTED_GGUF_TYPES = frozenset(("f32", "f16", "bf16", "q8_0"))
    if quantization_type not in SUPPORTED_GGUF_TYPES:
        raise RuntimeError(
            f"Unsloth: `{quantization_type}` quantization type is not supported.\n"\
            f"The following quantization types are supported: `{list(SUPPORTED_GGUF_TYPES)}`"
        )
    pass
    size_multiplier = {
        "q8_0" : 0.5,
        "f32"  : 2.0,
        "f16"  : 1.0,
        "bf16" : 1.0,
    }
    return quantization_type, size_multiplier[quantization_type]
pass


def check_max_shard_size(max_shard_size = "50GB"):
    # All Unsloth Zoo code licensed under LGPLv3
    assert(type(max_shard_size) is str)
    if max_shard_size.endswith("B"): max_shard_size = max_shard_size[:-1]
    try:
        _split_str_to_n_bytes(max_shard_size)
    except:
        raise TypeError(f"Unsloth: Shard size must be in GB, but `{max_shard_size}` is not")
    return max_shard_size
pass


def convert_to_gguf(
    input_folder,
    output_filename = None,
    quantization_type = "Q8_0",
    max_shard_size = "50GB",
    print_output = False,
    print_outputs = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Converts to GGUF using convert_hf_to_gguf.py directly!

    max_shard_size = check_max_shard_size(max_shard_size)
    quantization_type, _ = check_quantization_type(quantization_type)

    if not os.path.exists(input_folder):
        raise RuntimeError(f"Unsloth: `{input_folder}` does not exist?")

    config_file = os.path.join(input_folder, "config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"Unsloth: `config.json` does not exist inside `{input_folder}`.")

    # Load config.json
    with open(config_file, "r", encoding = "utf-8") as config_file:
        config_file = json.load(config_file)
    pass

    # Get latest llama.cpp conversion file
    conversion_filename, supported_types = _download_convert_hf_to_gguf()

    # Check if arch is supported
    assert("architectures") in config_file
    arch = config_file["architectures"][0]
    if arch not in supported_types:
        raise NotImplementedError(
            f"Unsloth: llama.cpp GGUF conversion does not yet support "\
            f"converting model types of `{arch}`."
        )
    pass

    # Get arguments
    if output_filename is None:
        output_filename = f"{input_folder}.{quantization_type.upper()}.gguf"
    else:
        assert(output_filename.endswith(".gguf"))

    args = {
        "--outfile"        : output_filename,
        "--outtype"        : quantization_type,
        "--split-max-size" : max_shard_size,
    }
    args = " ".join(f"{k} {v}" for k, v in args.items())
    metadata = None
    for python in ["python", "python3"]:
        try:
            command = f"{python} {conversion_filename} {args} {input_folder}"
            metadata = _convert_to_gguf(
                command,
                output_filename,
                print_output = print_output,
                print_outputs = print_outputs,
            )
            break
        except: continue
    pass

    if metadata is None:
        raise RuntimeError(f"Unsloth: Failed to convert {conversion_filename} to GGUF.")

    printed_metadata = "\n".join(metadata)
    if print_output: print(f"Unsloth: Successfully saved GGUF to:\n{printed_metadata}")

    return metadata
pass


def _assert_correct_gguf(model_name, model, tokenizer):
    # All Unsloth Zoo code licensed under LGPLv3
    # Verify if conversion is in fact correct by checking tokenizer and last tensor
    import gguf.gguf_reader
    from gguf.gguf_reader import GGUFReader

    # Stop until building tensors
    if not hasattr(GGUFReader, "__init__"):
        raise RuntimeError("Unsloth: Failed to verify GGUF: GGUFReader has no __init__")
    init_source = inspect.getsource(GGUFReader.__init__)
    text = "self._build_tensors(offs, tensors_fields"
    stop = init_source.find(text)
    if text not in init_source:
        raise RuntimeError(f"Unsloth: Failed to verify GGUF: Reader has no `{text}`")
    init_source = init_source.replace(text, text + "[-1:]")

    # Execute source and run partial GGUF reader
    source = f"class Partial_GGUFReader(GGUFReader):\n{init_source}"

    functions = dir(gguf.gguf_reader)
    functions = [x for x in functions if x in source]
    functions = f"from gguf.gguf_reader import ({','.join(functions)})"
    all_functions = {}
    exec(functions, all_functions)
    exec(source, all_functions)

    # Check if tokenizer is the same
    def check_gguf_tokenizer(tokenizer, reader):
        vocab = tokenizer.get_vocab()
        if not hasattr(reader, "fields"): return
        if not hasattr(reader.fields, "tokenizer.ggml.tokens"): return

        field = reader.fields["tokenizer.ggml.tokens"].data
        saved_vocab = [str(bytes(x), encoding = "utf-8") for x in field]

        vocab = [k for k, v in sorted(vocab.items(), key = lambda item: item[1])]
        if saved_vocab != vocab:
            raise RuntimeError("Unsloth: Failed converting to GGUF due to corrupted tokenizer.")
    pass

    # Get last tensor in file and check for exactness
    def check_gguf_last_tensor(model, reader):
        if not hasattr(reader, "tensors"): return

        last_tensor = reader.tensors[-1]
        last_tensor_data = torch.tensor(last_tensor.data)
        parameters = list(model.parameters())[-10:]

        distances = torch.ones(len(parameters), device = parameters[-1].device)
        found = False
        for k, param in enumerate(parameters):
            if param.shape[0] == last_tensor.shape[0]:
                x = torch.empty_like(param)
                x[:] = last_tensor_data[:]
                distances[k] = torch.dist(x, param)
                found = True
            pass
        pass
        if found:
            torch._assert(
                distances.min() == 0,
                "Unsloth: Failed converting to GGUF due to corrupted files."
            )
        pass
    pass

    reader = Partial_GGUFReader(model_name, "r")
    check_gguf_last_tensor(model, reader)
    check_gguf_tokenizer(tokenizer, reader)

    # Try parsing metadata
    try:
        from gguf.scripts.gguf_dump import dump_metadata_json
        class Arguments: pass
        args = Arguments()

        args.no_tensors = True
        args.model = model_name
        args.json_array = False

        # Stop prints
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            metadata = dump_metadata_json(reader, args)
        return
    except:
        pass
pass


def assert_correct_gguf(model_name, model, tokenizer):
    # All Unsloth Zoo code licensed under LGPLv3
    # Verify if conversion is in fact correct by checking tokenizer and last tensor
    if type(model_name) not in (list, tuple,):
        model_name = [model_name,]
    for name in model_name:
        _assert_correct_gguf(name, model, tokenizer)
    pass
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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
