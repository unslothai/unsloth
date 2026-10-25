# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image native int8 builds are cached in the background and read back instead of re-quantizing."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
pytest.importorskip("accelerate")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion as D  # noqa: E402
from core.inference import diffusion_native_cache as NC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

FAMILY = "flux.1"
CUDA = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def _fam():
    return detect_family("black-forest-labs/FLUX.1-schnell", override = FAMILY)


@pytest.fixture
def base_dir(tmp_path):
    root = tmp_path / "FLUX.1-schnell"
    torch.manual_seed(0)
    # 2 x 256 = 512 wide: every block Linear passes the 512 filter and g256 ConvRot divides it.
    model = diffusers.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 16,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 256,
        num_attention_heads = 2,
        joint_attention_dim = 512,
        pooled_projection_dim = 64,
        axes_dims_rope = (32, 112, 112),
    )
    model.save_pretrained(str(root / "transformer"), safe_serialization = True)
    return str(root)


@pytest.fixture(autouse = True)
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.delenv(C.ENV_ASYNC, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    return root


def _fresh_build(base_dir, act_int8 = True):
    model = diffusers.FluxTransformer2DModel.from_pretrained(
        base_dir, subfolder = "transformer", torch_dtype = torch.bfloat16
    )
    view = types.SimpleNamespace(transformer = model)
    assert (
        tq._quantize_native(
            view,
            "int8",
            family = FAMILY,
            min_features = tq.DEFAULT_MIN_LINEAR_FEATURES,
            act_int8 = act_int8,
            logger = None,
        )
        == "int8"
    )
    return model


def _assert_same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]), k


def _write(
    base_dir,
    model,
    fam = None,
):
    pending = NC.save_deferred(
        model, fam or _fam(), base_repo = "black-forest-labs/FLUX.1-schnell", source_base = base_dir
    )
    assert isinstance(pending, NC.NativePendingWrite)
    assert pending in C._BACKGROUND
    assert pending.start()
    assert pending.wait(120)
    assert pending not in C._BACKGROUND
    return pending.result


@pytest.mark.parametrize("act_int8", [True, False])
def test_background_write_round_trips_bit_identical(base_dir, act_int8):
    built = _fresh_build(base_dir, act_int8)
    assert _write(base_dir, built)
    spec = NC.native_spec("int8", FAMILY, act_int8 = act_int8)
    assert NC.build_spec(built, FAMILY) == spec
    read = NC.read_for_load(
        _fam(),
        spec,
        diffusers.FluxTransformer2DModel,
        base_repo = "black-forest-labs/FLUX.1-schnell",
        source_base = base_dir,
    )
    assert read is not None
    assert (
        read._unsloth_runtime_quant_info["native"]
        and read._unsloth_runtime_quant_info["cache_path"]
    )
    _assert_same(built, read)


def test_key_tracks_the_recipe(base_dir, monkeypatch):
    built = _fresh_build(base_dir, True)
    assert _write(base_dir, built)
    lookup = lambda spec: NC.lookup(  # noqa: E731
        _fam(), spec, base_repo = "black-forest-labs/FLUX.1-schnell", source_base = base_dir
    )
    assert lookup(NC.native_spec("int8", FAMILY, act_int8 = True))
    # weight-only, another rotation group, another family's exclusions, another base: all misses
    assert lookup(NC.native_spec("int8", FAMILY, act_int8 = False)) is None
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ROT", "0")
    assert lookup(NC.native_spec("int8", FAMILY, act_int8 = True)) is None
    monkeypatch.delenv("UNSLOTH_NATIVE_INT8_ROT")
    assert (
        NC.lookup(
            _fam(),
            NC.native_spec("int8", FAMILY, act_int8 = True),
            base_repo = "black-forest-labs/FLUX.1-dev",
            source_base = base_dir,
        )
        is None
    )
    # a torchao lookup never serves the native entry
    assert C.lookup(_fam(), "int8", base_repo = "black-forest-labs/FLUX.1-schnell") is None


def test_discarded_write_writes_nothing(base_dir):
    built = _fresh_build(base_dir, True)
    pending = NC.save_deferred(
        built, _fam(), base_repo = "black-forest-labs/FLUX.1-schnell", source_base = base_dir
    )
    pending.discard()
    assert pending.done() and pending not in C._BACKGROUND
    assert (
        NC.lookup(
            _fam(),
            NC.native_spec("int8", FAMILY, act_int8 = True),
            base_repo = "black-forest-labs/FLUX.1-schnell",
            source_base = base_dir,
        )
        is None
    )


def test_finish_background_writes_waits_for_a_native_write(base_dir):
    built = _fresh_build(base_dir, True)
    pending = NC.save_deferred(
        built, _fam(), base_repo = "black-forest-labs/FLUX.1-schnell", source_base = base_dir
    )
    assert pending.start()
    assert C.finish_background_writes(timeout = 120)
    assert pending.done() and pending.result


def test_sync_write_when_background_writes_are_off(base_dir, monkeypatch):
    monkeypatch.setenv(C.ENV_ASYNC, "0")
    built = _fresh_build(base_dir, True)
    out = NC.save_deferred(
        built, _fam(), base_repo = "black-forest-labs/FLUX.1-schnell", source_base = base_dir
    )
    assert isinstance(out, str) and Path(out).is_file()


def test_backend_save_routes_native_builds_to_the_native_cache(base_dir):
    built = _fresh_build(base_dir, True)
    pipe = types.SimpleNamespace(transformer = built)
    pending = D.DiffusionBackend._save_pipeline_quant_to_cache(
        pipe,
        _fam(),
        [("transformer", built)],
        loras = None,
        base = "black-forest-labs/FLUX.1-schnell",
        source_base = base_dir,
        defer = True,
        native_cache = True,
    )
    assert isinstance(pending, NC.NativePendingWrite)
    pending.start()
    assert pending.wait(120) and pending.result


def test_backend_save_skips_a_native_build_the_load_cannot_read_back(base_dir):
    # MPS / ROCm builds never hit the read path, so writing them would only cost disk and an unload wait.
    built = _fresh_build(base_dir, True)
    pipe = types.SimpleNamespace(transformer = built)
    pending = D.DiffusionBackend._save_pipeline_quant_to_cache(
        pipe,
        _fam(),
        [("transformer", built)],
        loras = None,
        base = "black-forest-labs/FLUX.1-schnell",
        source_base = base_dir,
        defer = True,
    )
    assert pending is None
    assert (
        NC.lookup(
            _fam(),
            NC.build_spec(built, _fam().name),
            base_repo = "black-forest-labs/FLUX.1-schnell",
            source_base = base_dir,
        )
        is None
    )


def test_backend_read_hits_after_a_write_and_misses_before(base_dir):
    spec_args = dict(base = "black-forest-labs/FLUX.1-schnell", source_base = base_dir)
    read = lambda target: D.DiffusionBackend._native_quant_from_cache(  # noqa: E731
        _fam(), diffusers.FluxTransformer2DModel, "int8", target, **spec_args
    )
    assert read(CUDA) is None
    built = _fresh_build(base_dir, D.native_int8_act(CUDA))
    assert _write(base_dir, built)
    hit = read(CUDA)
    assert hit is not None
    _assert_same(built, hit)
    # CPU / MPS targets never read (their weight-only scale dtype is not part of the key)
    assert read(types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)) is None
    assert (
        read(CUDA) is not None
        and D.DiffusionBackend._native_quant_from_cache(
            _fam(), diffusers.FluxTransformer2DModel, None, CUDA, **spec_args
        )
        is None
    )


def test_cache_disabled_reads_and_writes_nothing(base_dir, monkeypatch):
    monkeypatch.setenv(C.ENV_MODE, "0")
    built = _fresh_build(base_dir, True)
    assert (
        NC.save_deferred(
            built, _fam(), base_repo = "black-forest-labs/FLUX.1-schnell", source_base = base_dir
        )
        is None
    )
