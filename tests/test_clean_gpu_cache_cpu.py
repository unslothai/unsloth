# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
"""clean_gpu_cache on a CPU-only host, where the device module is torch.cpu (no empty_cache). No `import unsloth`."""

import ast
import pathlib
import types

import torch

_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "device_type.py"


def _clean_gpu_cache(device_module):
    tree = ast.parse(_SOURCE.read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clean_gpu_cache"
    )
    namespace = {"_DEVICE_MODULE": device_module}
    exec(compile(ast.Module([node], []), str(_SOURCE), "exec"), namespace)
    return namespace["clean_gpu_cache"]


def test_cpu_device_module_is_a_no_op():
    _clean_gpu_cache(torch.cpu)()


def test_accelerator_cache_is_still_emptied():
    calls = []
    _clean_gpu_cache(types.SimpleNamespace(empty_cache = lambda: calls.append(1)))()
    assert calls == [1]


def test_mlx_runtime_without_a_device_module():
    _clean_gpu_cache(None)()
