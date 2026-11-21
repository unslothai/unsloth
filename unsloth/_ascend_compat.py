# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in cold-import bootstrap for eager Ascend on ARM64.

Torch 2.6's Inductor imports AttrsDescriptor, removed in Triton 3.5. Importing
its metadata through torch_npu therefore fails before NPU detection. In this
specific eager-only environment, initialize the vendor backend using Inductor's
no-Triton metadata fallback, then restore Triton for ordinary Unsloth imports.

Set UNSLOTH_ASCEND_EAGER_BOOTSTRAP=1, TORCH_COMPILE_DISABLE=1 and
UNSLOTH_COMPILE_DISABLE=1 before importing Unsloth. This does not enable NPU
compilation or provide a complete NPU training backend.
"""

import os
import platform
import sys
from importlib import import_module, util
from importlib.metadata import PackageNotFoundError, version


def bootstrap_ascend_eager():
    if os.environ.get("UNSLOTH_ASCEND_EAGER_BOOTSTRAP") != "1":
        return False
    if any(os.environ.get(k) != "1" for k in ("TORCH_COMPILE_DISABLE", "UNSLOTH_COMPILE_DISABLE")):
        return False
    if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        return False
    # Never change an initialized process's Inductor or backend registration.
    if "torch" in sys.modules or "torch_npu" in sys.modules:
        return False
    if util.find_spec("torch_npu") is None:
        return False
    try:
        torch_version = version("torch").split("+")[0].split(".")[:2]
        triton_version = version("triton").split("+")[0].split(".")[:2]
    except PackageNotFoundError:
        return False
    if torch_version != ["2", "6"] or triton_version != ["3", "5"]:
        return False
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    missing = object()
    previous = sys.modules.get("triton", missing)
    sys.modules["triton"] = None
    try:
        import_module("torch")
        import_module("torch_npu")
    finally:
        if previous is missing:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = previous
    return True
