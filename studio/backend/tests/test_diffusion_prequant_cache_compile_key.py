# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An int8 read-back keys its compile cache like the runtime quantize; fp8 read-backs still install the dispatcher."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_diffusion_prequant_cache as T  # noqa: E402
from core.inference import diffusion_compile_cache as CC  # noqa: E402
from core.inference import diffusion_fp8_unit_scale as U  # noqa: E402
from core.inference import diffusion_prequant as P  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402

env = T.env


@pytest.fixture
def ada(monkeypatch):
    cc = pytest.importorskip("torch.compiler.config")
    if not isinstance(getattr(cc, "cache_key_tag", None), str):
        pytest.skip("torch without compiler cache_key_tag")
    before = cc.cache_key_tag
    U.uninstall()
    monkeypatch.setenv(U.UNIT_SCALE_ENV, "1")
    yield cc
    U.uninstall()
    cc.cache_key_tag = before


def _key(transformer, quant) -> str:
    model_fp = CC.model_fingerprint(
        family = T.FAMILY,
        transformer = transformer,
        dtype = torch.bfloat16,
        quant = quant,
        attention_backend = None,
        compile_kwargs = {"mode": "default", "fullgraph": True, "dynamic": None},
    )
    return CC.cache_key(CC.environment_fingerprint(), model_fp)


def _fresh_then_readback(tmp_path, monkeypatch, scheme_name):
    from core.inference import diffusion as D

    base, model = T._write_base(tmp_path)
    dense = T.Tiny().to(torch.bfloat16)
    dense.load_state_dict(model.state_dict())
    fam = T._fam(base)
    T._quantize(model, scheme_name)
    fresh_key = _key(model, scheme_name)
    assert isinstance(C.save_runtime_quant(model, fam, base_repo = str(base)), str)
    T._patch_scheme(monkeypatch, D, scheme_name)
    pipe = types.SimpleNamespace(transformer = dense)
    assert T._pipeline_read(D, pipe, fam, base, scheme_name) == scheme_name
    return fresh_key, _key(dense, scheme_name)


def _int8_cpu_quantizable_and_flattenable() -> bool:
    import core.inference.prequant_safetensors as PS
    try:
        from torchao.quantization import quantize_

        m = torch.nn.Sequential(torch.nn.Linear(512, 512)).to(torch.bfloat16)
        quantize_(m, T._make_quant_config("int8"))
        helpers = PS._torchao_helpers()
        if helpers is None:
            return False
        helpers[0](m.state_dict())
        return True
    except Exception:  # noqa: BLE001
        return False


def test_an_int8_read_back_keys_like_the_runtime_quantize_it_copies(
    env, tmp_path, ada, monkeypatch
):
    if not _int8_cpu_quantizable_and_flattenable():
        pytest.skip("int8 is not CPU-quantizable + flattenable on this torchao")
    # Probed apart, so a cache save / read-back regression fails here instead of skipping.
    T._SCHEME = ("int8", True)
    try:
        fresh_key, read_key = _fresh_then_readback(tmp_path, monkeypatch, "int8")
    finally:
        T._SCHEME = None
    assert not U._STATE["installed"]
    assert U._CACHE_TAG_PREFIX not in ada.cache_key_tag
    assert read_key == fresh_key


def test_an_fp8_or_unknown_read_back_still_installs_the_dispatcher(tmp_path, monkeypatch):
    import core.inference.prequant_safetensors as PS

    calls = []
    monkeypatch.setattr(P, "_install_fp8_unit_scale", lambda **_: calls.append(1))
    monkeypatch.setattr(PS, "is_safetensors_checkpoint", lambda path: True)
    fp8 = type("Float8Tensor", (), {})()
    int8 = type("Int8Tensor", (), {})()
    for state_dict, installs in (
        ({"w": fp8, "b": torch.zeros(1)}, True),
        ({"w": int8, "v": fp8}, True),
        ({"w": int8, "b": torch.zeros(1)}, False),
    ):
        calls.clear()
        monkeypatch.setattr(
            PS,
            "load_prequant_safetensors",
            lambda path, mmap = False, sd = state_dict: {"state_dict": sd},
        )
        assert (
            P._load_prequant_checkpoint(str(tmp_path / "x.safetensors"))["state_dict"] is state_dict
        )
        assert bool(calls) is installs


def test_only_an_all_int8_checkpoint_skips_the_install():
    int8 = type("Int8Tensor", (), {})()
    fp8 = type("Float8Tensor", (), {})()
    legacy = type("AffineQuantizedTensor", (), {})()
    plain = torch.zeros(1)
    assert P._int8_only({"state_dict": {"a": int8, "b": plain}})
    assert not P._int8_only({"state_dict": {"a": int8, "b": fp8}})
    assert not P._int8_only({"state_dict": {"a": legacy}})
    assert not P._int8_only({"state_dict": {"b": plain}})
    assert not P._int8_only({"format": "x"})
    assert not P._int8_only(None)
