# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

"""Guarded imports and capability probes for the optional ExLlamaV3 dependency."""

from __future__ import annotations

import importlib
import importlib.util
import functools
import os

# Minimum exllamav3 version that exposes the EXL3 linear layer, the conversion
# entry points, and the transformers integration we rely on.
MIN_EXLLAMA_VERSION = "0.0.5"

EXLLAMA_IMPORT_ERROR = (
    "Unsloth: ExLlamaV3 is required for EXL3 quantization but is not installed "
    "(or failed to import).\n"
    "EXL3 is Unsloth's quantization backend (replacing bitsandbytes) and enables "
    "2/3/4/6/8-bit and fractional-bit quantization plus MoE support.\n"
    "Install it with:\n"
    "    pip install exllamav3\n"
    "or grab a prebuilt wheel matching your CUDA/PyTorch from\n"
    "    https://github.com/turboderp-org/exllamav3/releases\n"
    "ExLlamaV3 requires a CUDA GPU with PyTorch built for CUDA 12.4 or newer."
)

# Allow users / CI to hard-disable the backend without uninstalling it.
_DISABLED = os.environ.get("UNSLOTH_DISABLE_EXLLAMA", "0") == "1"


@functools.lru_cache(maxsize = 1)
def is_exllama_available() -> bool:
    """Return True if exllamav3 can be imported and is not disabled.

    Result is cached: the answer cannot change within a process.
    """
    if _DISABLED:
        return False
    if importlib.util.find_spec("exllamav3") is None:
        return False
    try:
        importlib.import_module("exllamav3")
    except Exception:
        return False
    # Enforce the minimum version that exposes the APIs we rely on (EXL3 linear
    # layer, converter entry points, transformers integration).
    version = exllama_version()
    if version is not None and not _version_at_least(version, MIN_EXLLAMA_VERSION):
        return False
    return True


def _version_at_least(have: str, want: str) -> bool:
    """Return True if version string ``have`` >= ``want`` (numeric, dotted)."""
    try:
        from packaging.version import Version
        return Version(str(have)) >= Version(str(want))
    except Exception:

        def _parts(v):
            return [
                int("".join(c for c in ch if c.isdigit()) or 0)
                for ch in str(v).split("+")[0].split(".")
            ]

        a, b = _parts(have), _parts(want)
        n = max(len(a), len(b))
        a += [0] * (n - len(a))
        b += [0] * (n - len(b))
        return a >= b


@functools.lru_cache(maxsize = 1)
def exllama_version() -> str | None:
    """Return the installed exllamav3 version string, or None if unavailable."""
    # NOTE: do not call is_exllama_available() here - it depends on this
    # function (version-gating), which would recurse. Probe the import directly.
    if _DISABLED or importlib.util.find_spec("exllamav3") is None:
        return None
    try:
        module = importlib.import_module("exllamav3.version")
        return getattr(module, "__version__", None)
    except Exception:
        try:
            module = importlib.import_module("exllamav3")
            return getattr(module, "__version__", None)
        except Exception:
            return None


def require_exllama():
    """Import and return the exllamav3 module, raising a clear error if missing.

    This is the single choke point for the backend; call it before touching any
    exllamav3 symbol so the failure mode is one actionable message rather than a
    raw ``ModuleNotFoundError`` deep in a call stack.
    """
    if not is_exllama_available():
        raise ImportError(EXLLAMA_IMPORT_ERROR)
    return importlib.import_module("exllamav3")
