# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a PEFT adapter directory to a GGUF LoRA. Lazy Unsloth / converter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from unforgettable.sidecar.peft import is_peft_adapter_dir

CONVERT_ENV = "UNFORGETTABLE_CONVERT_LORA_TO_GGUF"
MISSING_CONVERTER = (
    "convert_lora_to_gguf.py not found; install llama.cpp scripts via Unsloth "
    f"or set {CONVERT_ENV}"
)
NOT_PEFT_DIR = "not a PEFT adapter directory: {path}"
NO_GGUF_WRITTEN = "converter produced no GGUF at {path}"
DEFAULT_OUTTYPE = "f16"


def _basename(base_model: Optional[str]) -> str:
    text = (base_model or "adapter").strip().replace("\\", "/")
    return (text.rsplit("/", 1)[-1] or "adapter").replace(":", "_")


def find_converter() -> Optional[str]:
    env = (os.environ.get(CONVERT_ENV) or "").strip()
    if env and Path(env).is_file():
        return env
    try:
        from unsloth.save import LLAMA_CPP_DEFAULT_DIR
    except Exception:
        LLAMA_CPP_DEFAULT_DIR = "llama.cpp"
    candidates = [
        Path(LLAMA_CPP_DEFAULT_DIR) / "convert_lora_to_gguf.py",
        Path(LLAMA_CPP_DEFAULT_DIR).parent / "llama.cpp-source" / "convert_lora_to_gguf.py",
    ]
    try:
        from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR as zoo_dir
        candidates.insert(0, Path(zoo_dir) / "convert_lora_to_gguf.py")
    except Exception:
        pass
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def export_adapter_gguf(
    peft_dir,
    *,
    base_model: Optional[str] = None,
    outtype: str = DEFAULT_OUTTYPE,
) -> str:
    root = Path(peft_dir)
    if not is_peft_adapter_dir(root):
        raise ValueError(NOT_PEFT_DIR.format(path = root))
    converter = find_converter()
    if not converter:
        raise FileNotFoundError(MISSING_CONVERTER)
    outfile = root / f"{_basename(base_model)}-lora-{outtype}.gguf"
    cmd = [
        sys.executable,
        converter,
        str(root),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    if base_model and Path(base_model).is_dir():
        cmd += ["--base", str(base_model)]
    subprocess.run(cmd, check = True)
    if not outfile.is_file():
        raise RuntimeError(NO_GGUF_WRITTEN.format(path = outfile))
    return str(outfile)
