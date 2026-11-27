# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A sink race that finds torch's attention faster hands every strided processor its stock one back: the status and
the log then name the attention that runs, not the INT8 kernel the install logged (CPU, fakes only)."""

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_int8_attention as I8  # noqa: E402


class _Stock:
    _attention_backend = None


class _Strided:
    _unsloth_int8 = False


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 24
        self.processor = _Stock()


def _engaged_dit(n = 3):
    dit = torch.nn.Sequential(*[_Attn() for _ in range(n)])
    for m in dit:
        proc = _Strided()
        proc._unsloth_int8 = True
        proc._unsloth_stock = m.processor
        m.processor = proc
        I8.register_route(0, 128, I8._raise_processor_floor(proc, m))
    return dit


@pytest.fixture(autouse = True)
def _clean_routes():
    I8.forget_plans()
    yield
    I8.forget_plans()


def test_lost_sink_race_reports_the_incumbent(caplog):
    dit = _engaged_dit()
    pipe = SimpleNamespace(transformer = dit)
    setattr(pipe, I8.INCUMBENT_ATTR, None)
    assert I8.int8_attention_count(dit) == 3
    assert I8.live_backend(pipe, I8.BACKEND) == I8.BACKEND
    with caplog.at_level(logging.INFO, logger = I8.__name__):
        I8._drop_routes({"shape": (1, 4176, 4176, 24, 128), "index": 0})
    assert all(isinstance(m.processor, _Stock) for m in dit)
    assert I8.int8_attention_count(dit) == 0
    # native (None) runs now, not the kernel the install logged
    assert I8.live_backend(pipe, I8.BACKEND) is None
    text = caplog.text
    assert (
        "torch attention beat the sink path" in text
        and "3 attention modules back on their stock processor" in text
    )


def test_routes_on_another_head_dim_stay_reported():
    dit = _engaged_dit()
    pipe = SimpleNamespace(transformer = dit)
    setattr(pipe, I8.INCUMBENT_ATTR, "_native_cudnn")
    I8._drop_routes({"shape": (1, 4176, 4176, 24, 64), "index": 0})
    assert I8.int8_attention_count(dit) == 3
    assert I8.live_backend(pipe, I8.BACKEND) == I8.BACKEND


def test_other_backends_and_unrecorded_loads_pass_through():
    dit = _engaged_dit()
    I8._drop_routes({"shape": (1, 4176, 4176, 24, 128), "index": 0})
    # no recorded incumbent (another install path): the load-time value stays
    assert I8.live_backend(SimpleNamespace(transformer = dit), I8.BACKEND) == I8.BACKEND
    pipe = SimpleNamespace(transformer = dit)
    setattr(pipe, I8.INCUMBENT_ATTR, None)
    assert I8.live_backend(pipe, "_native_cudnn") == "_native_cudnn"
    assert I8.live_backend(None, I8.BACKEND) == I8.BACKEND
