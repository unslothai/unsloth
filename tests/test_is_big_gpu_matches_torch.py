# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""is_big_gpu matches torch._inductor.utils.is_big_gpu's default index and SM thresholds, read from source."""

import ast
import functools
import os
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unsloth_is_big_gpu(device_type, sms):
    path = os.path.join(REPO, "unsloth", "models", "_utils.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_big_gpu")
    seen = []

    def create(device):
        seen.append(device)
        return types.SimpleNamespace(multi_processor_count = sms)

    ns = {
        "functools": functools,
        "torch": torch,
        "DEVICE_TYPE": device_type,
        "DeviceProperties": types.SimpleNamespace(create = create),
    }
    exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), ns)
    return ns["is_big_gpu"], seen


def _torch_is_big_gpu(monkeypatch, sms):
    import torch._inductor.utils as utils

    if torch.version.hip:
        pytest.skip("torch's ROCm branch does not read the SM count")
    original = getattr(utils.is_big_gpu, "__wrapped__", utils.is_big_gpu)
    monkeypatch.setattr(
        utils.DeviceProperties,
        "create",
        staticmethod(lambda device: types.SimpleNamespace(multi_processor_count = sms)),
    )
    return original(torch.device("cuda", 0))


@pytest.mark.parametrize("sms", [58, 67, 68, 80, 132])
def test_cuda_threshold_and_default_index_match_torch(monkeypatch, sms):
    fn, seen = _unsloth_is_big_gpu("cuda", sms)
    assert fn() == _torch_is_big_gpu(monkeypatch, sms)
    assert seen == [torch.device("cuda", 0)]


def test_xpu_threshold():
    assert _unsloth_is_big_gpu("xpu", 16)[0]() and not _unsloth_is_big_gpu("xpu", 15)[0]()
