# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""torchrun ranks import Unsloth with cuda:0 current, so the first CUDA call during import opened a context on
GPU 0 in every rank. Unsloth makes cuda:LOCAL_RANK current first. Loaded by path: CPU only, no unsloth import."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HELPER = Path(__file__).resolve().parents[1] / "unsloth" / "parallel_env.py"


@pytest.fixture
def bind(monkeypatch):
    spec = importlib.util.spec_from_file_location("_unsloth_parallel_env_bind", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    return module.bind_local_cuda_device, calls


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"WORLD_SIZE": "2", "LOCAL_RANK": "1"}, 1),
        ({"WORLD_SIZE": "2", "LOCAL_RANK": "0"}, None),
        ({"WORLD_SIZE": "1", "LOCAL_RANK": "1"}, None),
        ({"WORLD_SIZE": "4", "LOCAL_RANK": "3"}, None),
        ({"WORLD_SIZE": "2"}, None),
        ({"WORLD_SIZE": "2", "LOCAL_RANK": "1", "UNSLOTH_FORCE_CPU": "1"}, None),
    ],
)
def test_rank_binds_its_own_card(bind, env, expected):
    fn, calls = bind
    assert fn(env) == expected
    assert calls == ([] if expected is None else [expected])


def test_init_binds_before_transformers_is_imported():
    source = (HELPER.parent / "__init__.py").read_text(encoding = "utf-8")
    assert "_bind_local_cuda_device()" in source
    assert source.index("_bind_local_cuda_device()") < source.index(
        'if "transformers" not in sys.modules'
    )
