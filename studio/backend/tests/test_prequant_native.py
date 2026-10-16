# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safetensors pre-quant checkpoints rebuilt without torchao's deserializer, the local mirror, and the converter."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

import core.inference.diffusion_prequant as pq
import core.inference.prequant_native as pn
import core.inference.prequant_safetensors as ps

torch = pytest.importorskip("torch")
torchao = pytest.importorskip("torchao")

_HAVE_HELPERS = ps.safetensors_prequant_supported()
needs_helpers = pytest.mark.skipif(
    not _HAVE_HELPERS, reason = "needs torchao >= 0.16 flatten helpers"
)
_V1 = pn._v1_int8_api() is not None
_API = pn._torchao_api() or {}
needs_int8tensor = pytest.mark.skipif("Int8Tensor" not in _API, reason = "needs torchao Int8Tensor")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _int8_tensor(
    n = 8,
    k = 32,
    seed = 0,
):
    from core.inference.prequant_legacy_int8 import _int8_tensor_api, _to_int8_tensor

    g = torch.Generator().manual_seed(seed)
    qdata = torch.randint(-127, 128, (n, k), dtype = torch.int8, generator = g)
    scale = (torch.rand(n, generator = g) / 50 + 1e-3).to(torch.bfloat16)
    return _to_int8_tensor(qdata, scale, torch.bfloat16, _int8_tensor_api())


def _fp8_tensor(
    n = 8,
    k = 32,
    seed = 0,
):
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization import Float8Tensor, PerRow
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        QuantizeTensorToFloat8Kwargs,
    )

    g = torch.Generator().manual_seed(seed)
    qdata = (torch.randn(n, k, generator = g) * 50).to(torch.float8_e4m3fn)
    scale = torch.rand(n, 1, generator = g) / 100 + 1e-4
    act = QuantizeTensorToFloat8Kwargs(
        float8_dtype = torch.float8_e4m3fn,
        granularity = PerRow(),
        mm_config = None,
        hp_value_lb = 1e-12,
        hp_value_ub = None,
        kernel_preference = KernelPreference.AUTO,
    )
    return Float8Tensor(
        qdata,
        scale,
        block_size = [1, k],
        mm_config = Float8MMConfig(use_fast_accum = True),
        act_quant_kwargs = act,
        kernel_preference = KernelPreference.AUTO,
        dtype = torch.bfloat16,
    )


def _write(
    path,
    state_dict,
    *,
    layout = None,
    scheme = "int8",
):
    from safetensors.torch import save_file

    flatten, _ = ps._torchao_helpers()
    roots = ps._root_level_keys(state_dict)
    flat, meta = flatten({k: v for k, v in state_dict.items() if k not in roots})
    header = dict(meta)
    header[ps.UNSLOTH_FORMAT_KEY] = pq.PREQUANT_FORMAT
    header[ps.UNSLOTH_METADATA_KEY] = json.dumps({"scheme": scheme})
    if layout is not None:
        header[pn.QUANT_LAYOUT_KEY] = json.dumps(layout)
    flat = dict(flat)
    if roots:
        for k in roots:
            flat[ps.UNSLOTH_ROOT_PREFIX + k] = state_dict[k]
        header[ps.UNSLOTH_ROOT_KEYS_KEY] = json.dumps(roots)
    save_file({k: v.contiguous() for k, v in flat.items()}, str(path), metadata = header)
    return path


def _canon(value):
    if type(value) is torch.Tensor:
        return ("Tensor", value)
    names, ctx = value.__tensor_flatten__()
    return (type(value).__name__, {n: _canon(getattr(value, n)) for n in names}, repr(ctx))


def _same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict = True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def _v1_layout():
    return {"version": 1, "int8_source": pn.INT8_SOURCE_V1, "int8_v1": dict(pn.INT8_V1_FACTS)}


@needs_helpers
@needs_int8tensor
def test_native_rebuild_matches_torchao_unflatten(tmp_path, monkeypatch):
    sd = {
        "blk.q.weight": _int8_tensor(),
        "blk.f.weight": _fp8_tensor(),
        "blk.norm.weight": torch.ones(32, dtype = torch.bfloat16),
        "pad_token": torch.zeros(1, 32, dtype = torch.bfloat16),
    }
    path = _write(tmp_path / "M-INT8.safetensors", sd)
    native = ps.load_prequant_safetensors(str(path))
    monkeypatch.setenv(pn.NATIVE_REBUILD_ENV, "0")
    stock = ps.load_prequant_safetensors(str(path))
    assert native["reader"] == "native" and stock["reader"] == "torchao"
    assert native["state_dict"].keys() == stock["state_dict"].keys() == sd.keys()
    for key in sd:
        assert _same(_canon(native["state_dict"][key]), _canon(stock["state_dict"][key])), key
        assert _same(_canon(native["state_dict"][key]), _canon(sd[key])), key


@needs_helpers
@needs_int8tensor
def test_mapped_read_is_the_same_tensors_as_a_copied_read(tmp_path):
    sd = {"blk.q.weight": _int8_tensor(), "blk.f.weight": _fp8_tensor(), "x": torch.arange(4.0)}
    path = _write(tmp_path / "M-INT8.safetensors", sd)
    mapped = ps.load_prequant_safetensors(str(path), mmap = True)["state_dict"]
    copied = ps.load_prequant_safetensors(str(path))["state_dict"]
    for key in sd:
        assert _same(_canon(mapped[key]), _canon(copied[key])), key


def test_a_tensor_off_its_dtype_alignment_is_still_read(tmp_path):
    a = torch.tensor([1, -2, 3], dtype = torch.int8)
    b = torch.tensor([1.5, -0.25], dtype = torch.bfloat16)
    header = {
        "a": {"dtype": "I8", "shape": [3], "data_offsets": [0, 3]},
        "b": {"dtype": "BF16", "shape": [2], "data_offsets": [3, 7]},
    }
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path = tmp_path / "odd.safetensors"
    path.write_bytes(
        len(raw).to_bytes(8, "little")
        + raw
        + a.numpy().tobytes()
        + b.view(torch.int16).numpy().tobytes()
    )
    _, tensors = ps._mapped_tensors(str(path))
    assert torch.equal(tensors["a"], a) and torch.equal(tensors["b"], b)


def test_a_truncated_header_is_refused(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes((10_000).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError):
        ps._mapped_tensors(str(path))


@needs_helpers
@needs_int8tensor
def test_v1_converted_int8_rebuilds_as_v1_only_where_torchao_ships_it(tmp_path):
    sd = {"blk.q.weight": _int8_tensor(), "blk.norm.weight": torch.ones(32, dtype = torch.bfloat16)}
    path = _write(tmp_path / "M-INT8.safetensors", sd, layout = _v1_layout())
    w = ps.load_prequant_safetensors(str(path))["state_dict"]["blk.q.weight"]
    if _V1:
        assert type(w).__name__ == "LinearActivationQuantizedTensor"
        impl = w.original_weight_tensor.tensor_impl
        assert torch.equal(impl.int_data, sd["blk.q.weight"].qdata)
        assert torch.equal(impl.scale, sd["blk.q.weight"].scale.reshape(-1))
        assert impl.zero_point is None
        assert w.input_quant_func.__name__ == "_int8_symm_per_token_reduced_range_quant"
    else:
        assert type(w).__name__ == "Int8Tensor"
    # Without the v1 declaration the artifact keeps the class it was written as, on every torchao.
    plain = _write(tmp_path / "P-INT8.safetensors", sd)
    assert (
        type(ps.load_prequant_safetensors(str(plain))["state_dict"]["blk.q.weight"]).__name__
        == "Int8Tensor"
    )


@pytest.mark.skipif(not _V1, reason = "needs torchao <= 0.17 (v1 int8 classes)")
@needs_helpers
def test_v1_rebuild_is_bit_identical_to_a_real_v1_quantize(tmp_path):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    from core.inference.prequant_legacy_int8 import convert_legacy_int8_weights

    model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias = False)).to(torch.bfloat16)
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    v1 = model[0].weight
    assert type(v1).__name__ == "LinearActivationQuantizedTensor"
    reference = dict(model.state_dict())
    converted = torch.nn.Sequential(torch.nn.Linear(64, 32, bias = False)).to(torch.bfloat16)
    converted.load_state_dict(reference, assign = True)
    assert convert_legacy_int8_weights(converted) == 1
    path = _write(
        tmp_path / "M-INT8.safetensors", dict(converted.state_dict()), layout = _v1_layout()
    )
    back = ps.load_prequant_safetensors(str(path))["state_dict"]
    assert _same(_canon(back["0.weight"]), _canon(v1))
    x = torch.randn(17, 64, dtype = torch.bfloat16)
    assert torch.equal(
        torch.nn.functional.linear(x, back["0.weight"]), torch.nn.functional.linear(x, v1)
    )


@needs_helpers
@needs_int8tensor
def test_unknown_entries_fall_back_to_torchao_and_leave_the_tensors_alone(tmp_path):
    sd = {"blk.q.weight": _int8_tensor()}
    flatten, _ = ps._torchao_helpers()
    flat, meta = flatten(sd)
    raw = dict(meta)
    entry = json.loads(raw["blk.q.weight"])
    entry["_type"] = "SomeFutureTensor"
    raw["blk.q.weight"] = json.dumps(entry)
    before = dict(flat)
    assert pn.native_unflatten(flat, raw) is None
    assert flat.keys() == before.keys()


@needs_helpers
@needs_int8tensor
def test_a_missing_scale_or_a_stray_tensor_is_named(tmp_path):
    flatten, _ = ps._torchao_helpers()
    flat, meta = flatten({"blk.q.weight": _int8_tensor()})
    no_scale = {k: v for k, v in flat.items() if not k.endswith("_scale")}
    with pytest.raises(ValueError, match = "missing"):
        pn.native_unflatten(no_scale, dict(meta), path = "x")
    stray = dict(flat)
    stray["blk.other._weight_qdata"] = torch.zeros(2, 2, dtype = torch.int8)
    with pytest.raises(ValueError, match = "does not account for"):
        pn.native_unflatten(stray, dict(meta), path = "x")


@needs_helpers
@needs_int8tensor
def test_a_live_field_this_torchao_lacks_is_refused_and_an_inert_one_is_dropped():
    flatten, _ = ps._torchao_helpers()
    flat, meta = flatten({"blk.f.weight": _fp8_tensor()})
    entry = json.loads(meta["blk.f.weight"])
    entry["_data"]["mm_config"]["_data"]["some_new_knob"] = False
    inert = dict(meta, **{"blk.f.weight": json.dumps(entry)})
    out = pn.native_unflatten(dict(flat), inert)
    assert type(out["blk.f.weight"]).__name__ == "Float8Tensor"
    entry["_data"]["mm_config"]["_data"]["some_new_knob"] = 3
    live = dict(meta, **{"blk.f.weight": json.dumps(entry)})
    with pytest.raises(ValueError, match = "some_new_knob"):
        pn.native_unflatten(dict(flat), live)


@needs_helpers
@needs_int8tensor
def test_layouts_the_native_reader_does_not_model_go_to_torchao():
    from torchao.quantization import Float8Tensor, PerRow, PerTensor
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
        Int8Tensor,
        QuantizeTensorToInt8Kwargs,
    )

    flatten, _ = ps._torchao_helpers()
    per_tensor = Int8Tensor.from_hp(
        torch.randn(16, 32, dtype = torch.bfloat16),
        PerTensor(),
        act_quant_kwargs = QuantizeTensorToInt8Kwargs(granularity = PerRow()),
    )
    experts = Float8Tensor.from_hp(
        torch.randn(2, 16, 32, dtype = torch.bfloat16), granularity = PerRow()
    )
    for weight in (per_tensor, experts):
        flat, meta = flatten({"blk.w.weight": weight})
        assert pn.native_unflatten(dict(flat), dict(meta)) is None
    flat, meta = flatten({"blk.w.weight": _int8_tensor()})
    entry = json.loads(meta["blk.w.weight"])
    entry["_data"]["future_live_field"] = "per_block_128"
    assert (
        pn.native_unflatten(dict(flat), dict(meta, **{"blk.w.weight": json.dumps(entry)})) is None
    )


@needs_helpers
@needs_int8tensor
def test_duplicate_names_and_unknown_granularity_fields_go_to_torchao():
    flatten, _ = ps._torchao_helpers()
    flat, meta = flatten({"blk.f.weight": _fp8_tensor()})
    dup = dict(meta, tensor_names = json.dumps(["blk.f.weight", "blk.f.weight"]))
    assert pn.native_unflatten(dict(flat), dup) is None
    entry = json.loads(meta["blk.f.weight"])
    text = json.dumps(entry)
    assert '"_type": "PerRow", "_data": {"dim": -1}' in text
    odd = dict(
        meta,
        **{
            "blk.f.weight": text.replace(
                '"_type": "PerRow", "_data": {"dim": -1}',
                '"_type": "PerRow", "_data": {"dim": -1, "block": 128}',
            )
        },
    )
    assert pn.native_unflatten(dict(flat), odd) is None


def test_overlapping_tensor_bytes_are_refused(tmp_path):
    header = {
        "x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "y": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
    }
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path = tmp_path / "overlap.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(16))
    with pytest.raises(ValueError):
        ps._mapped_tensors(str(path))


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason = "needs float8_e8m0fnu")
def test_the_mapped_read_takes_mxfp8_scale_dtypes(tmp_path):
    from safetensors.torch import save_file

    t = torch.ones(8, 4).to(torch.float8_e8m0fnu)
    save_file({"a.b": t}, str(tmp_path / "e.safetensors"))
    _, tensors = ps._mapped_tensors(str(tmp_path / "e.safetensors"))
    assert tensors["a.b"].dtype == torch.float8_e8m0fnu and torch.equal(
        tensors["a.b"].view(torch.uint8), t.view(torch.uint8)
    )


def _mirror(tmp_path, monkeypatch, *files):
    root = tmp_path / "mirror"
    out = []
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        out.append(path)
    monkeypatch.setenv(pq.PREQUANT_MIRROR_ENV, str(root))
    return out


def test_a_mirrored_sibling_this_host_cannot_read_is_not_served(tmp_path, monkeypatch):
    _mirror(tmp_path, monkeypatch, "unsloth/Model-INT8/Model-INT8.safetensors")
    monkeypatch.setattr(ps, "safetensors_prequant_supported", lambda: False)
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: True)
    monkeypatch.setattr(pq, "_download_checkpoint_name", lambda *a, **k: "/hub/Model-INT8.pt")
    src = pq.PrequantSource(
        kind = "repo",
        location = "unsloth/Model-INT8",
        filename = "Model-INT8.pt",
        declared_filenames = ("Model-INT8.pt",),
    )
    assert pq._resolve_checkpoint_path(src, None, None, scheme = "int8") == "/hub/Model-INT8.pt"
    cached = pq.cached_checkpoint_path(src)
    assert cached is None or not ps.is_safetensors_checkpoint(cached)


def test_any_mirrored_name_wins_before_the_hub_is_asked(tmp_path, monkeypatch):
    (pt,) = _mirror(tmp_path, monkeypatch, "unsloth/Model-INT8/Model-INT8.pt")
    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )
    calls = []
    monkeypatch.setattr(
        pq,
        "_download_checkpoint_name",
        lambda source, name, *a, **k: calls.append(name) or f"/hub/{name}",
    )
    src = pq.PrequantSource(
        kind = "repo",
        location = "unsloth/Model-INT8",
        filename = "Model-INT8.safetensors",
        fallback_filenames = ("Model-INT8.pt",),
    )
    assert (
        pq._resolve_checkpoint_path(src, None, None, scheme = "int8") == str(pt.resolve())
        and calls == []
    )


def test_the_text_encoder_mirror_is_used_when_the_hub_is_unreachable(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    import core.inference.diffusion_te_prequant as te

    (pt,) = _mirror(tmp_path, monkeypatch, "unsloth/Model-FP8/Model-text_encoder-FP8.pt")

    def offline(*a, **k):
        raise LocalEntryNotFoundError("connection error")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    src = te.TePrequantSource(
        kind = "repo",
        location = "unsloth/Model-FP8",
        filename = "Model-text_encoder-FP8.safetensors",
        fallback_filenames = ("Model-text_encoder-FP8.pt",),
    )
    assert te._resolve_checkpoint_path(src, None, cache_dir = str(tmp_path / "hub")) == str(
        pt.resolve()
    )


def test_the_mirror_is_read_before_the_hub_and_cannot_escape_its_root(tmp_path, monkeypatch):
    root = tmp_path / "mirror"
    target = root / "unsloth" / "Model-FP8" / "Model-INT8.safetensors"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"x")
    outside = tmp_path / "secret.safetensors"
    outside.write_bytes(b"x")
    monkeypatch.setenv(pq.PREQUANT_MIRROR_ENV, str(root))
    assert pq.prequant_mirror_path("unsloth/Model-FP8", "Model-INT8.safetensors") == str(
        target.resolve()
    )
    assert pq.prequant_mirror_path("unsloth/Model-FP8", "Model-INT8.pt") == str(target.resolve())
    assert pq.prequant_mirror_path("unsloth/Model-FP8", "Model-FP8.pt") is None
    assert pq.prequant_mirror_path("unsloth/..", "secret.safetensors") is None
    assert pq.prequant_mirror_path("unsloth/Model-FP8", "../../../secret.safetensors") is None
    monkeypatch.delenv(pq.PREQUANT_MIRROR_ENV)
    assert pq.prequant_mirror_path("unsloth/Model-FP8", "Model-INT8.safetensors") is None


def test_the_resolver_takes_the_mirrored_safetensors_without_asking_the_hub(tmp_path, monkeypatch):
    root = tmp_path / "mirror"
    target = root / "unsloth" / "Model-FP8" / "Model-INT8.safetensors"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"x")
    monkeypatch.setenv(pq.PREQUANT_MIRROR_ENV, str(root))
    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )

    def no_hub(*args, **kwargs):
        raise AssertionError("the Hub was asked for a mirrored file")

    monkeypatch.setattr(pq, "_download_checkpoint_name", no_hub)
    src = pq.PrequantSource(
        kind = "repo",
        location = "unsloth/Model-FP8",
        filename = "Model-INT8.safetensors",
        fallback_filenames = ("Model-INT8.pt",),
    )
    assert pq._resolve_checkpoint_path(src, None, None, scheme = "int8") == str(target.resolve())
    assert pq.cached_checkpoint_path(src) == str(target.resolve())


def test_the_text_encoder_resolver_reads_the_mirror_too(tmp_path, monkeypatch):
    import core.inference.diffusion_te_prequant as te

    root = tmp_path / "mirror"
    target = root / "unsloth" / "Model-FP8" / "Model-text_encoder-FP8.safetensors"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"x")
    monkeypatch.setenv(pq.PREQUANT_MIRROR_ENV, str(root))
    import huggingface_hub

    def no_hub(*args, **kwargs):
        raise AssertionError("the Hub was asked for a mirrored file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_hub)
    src = te.TePrequantSource(
        kind = "repo",
        location = "unsloth/Model-FP8",
        filename = "Model-text_encoder-FP8.safetensors",
        fallback_filenames = ("Model-text_encoder-FP8.pt",),
    )
    assert te._resolve_checkpoint_path(src, None, cache_dir = str(tmp_path / "hub")) == str(
        target.resolve()
    )


def test_the_dense_fast_path_reason_names_a_load_time_failure(monkeypatch):
    import core.inference.diffusion as diffusion

    monkeypatch.setattr(diffusion, "prequant_unreadable_reason", lambda *a, **k: None)
    reason = diffusion._dense_fast_path_reason(
        object(), "int8", "org/base", "pipeline", None, None, failure = "ValueError: bad header"
    )
    assert "did not load (ValueError: bad header)" in reason and "quantized instead" in reason
    assert (
        diffusion._dense_fast_path_reason(object(), "int8", "org/base", "pipeline", None, None)
        == "engaged on the dense fast path"
    )


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "/srv/hf/hub/models--unsloth--X/snapshots/abc/X-INT8.safetensors: the data section does not match its header",
            "ValueError: X-INT8.safetensors: the data section does not match its header",
        ),
        (
            r"C:\Users\me\mirror\unsloth\X\X-FP8.safetensors is truncated",
            "ValueError: X-FP8.safetensors is truncated",
        ),
        (
            "unsloth/X-FP8 has no X-FP8.pt (https://huggingface.co/unsloth/X-FP8)",
            "ValueError: unsloth/X-FP8 has no X-FP8.pt (https://huggingface.co/unsloth/X-FP8)",
        ),
    ],
)
def test_the_status_failure_note_carries_no_server_directory(message, expected):
    import core.inference.diffusion_prequant as dp
    dp._warn(None, "load", ValueError(message))
    assert dp.last_prequant_failure() == expected


def test_every_load_starts_without_the_previous_loads_failure_note():
    import core.inference.diffusion as diffusion

    src = inspect.getsource(diffusion.DiffusionBackend.load_pipeline)
    reset = src.find("self._prequant_fallback_note = None")
    read = src.find('getattr(self, "_prequant_fallback_note", None)')
    assert 0 <= reset < read


def _converter():
    spec = importlib.util.spec_from_file_location(
        "convert_prequant_to_safetensors",
        REPO_ROOT / "scripts" / "convert_prequant_to_safetensors.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_writer():
    spec = importlib.util.spec_from_file_location(
        "_legacy_int8_fixture", Path(__file__).with_name("test_prequant_legacy_int8.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_foreign_legacy_registrations():
    """Other test files register placeholders under the v1 int8 names process-wide (see test_prequant_legacy_int8)."""
    import core.inference.prequant_legacy_int8 as li

    saved = torch.serialization.get_safe_globals()
    legacy = {*li.LEGACY_INT8_CLASS_NAMES, li._ACT_QUANT}
    torch.serialization.clear_safe_globals()
    torch.serialization.add_safe_globals(
        [g for g in saved if not (isinstance(g, tuple) and g[1] in legacy)]
    )
    yield
    torch.serialization.clear_safe_globals()
    torch.serialization.add_safe_globals(saved)


@needs_helpers
@needs_int8tensor
def test_the_converter_writes_a_bit_identical_v1_int8_artifact(
    tmp_path, monkeypatch, no_foreign_legacy_registrations
):
    src = tmp_path / "Model-INT8.pt"
    if _V1:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias = False)).to(torch.bfloat16)
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        sd = {
            "blk.proj.weight": model[0].weight,
            "blk.norm.weight": torch.ones(64, dtype = torch.bfloat16),
        }
        torch.save(
            {"format": pq.PREQUANT_FORMAT, "metadata": {"scheme": "int8"}, "state_dict": sd},
            str(src),
        )
    else:
        import core.inference.prequant_legacy_int8 as li
        if not li.legacy_int8_decode_supported():
            pytest.skip("no v1 classes and no legacy decoder")
        _legacy_writer()._write_legacy_checkpoint(monkeypatch, src)
    conv = _converter()
    rec = conv.convert(
        str(src), str(tmp_path / "out"), repo = "unsloth/Model-FP8", revision = "abc", sha = None
    )
    assert rec["verified_bit_identical"] and rec["int8_source"] == pn.INT8_SOURCE_V1
    dst = tmp_path / "out" / "unsloth" / "Model-FP8" / "Model-INT8.safetensors"
    assert rec["dst"] == str(dst) and dst.is_file()
    header = ps.read_prequant_header(str(dst))
    assert header["format"] == pq.PREQUANT_FORMAT and header["metadata"] == {"scheme": "int8"}
    from safetensors import safe_open

    with safe_open(str(dst), framework = "pt") as handle:
        raw = handle.metadata()
    assert json.loads(raw[pn.SOURCE_KEY])["sha256"] == conv.sha256_of(str(src))
    assert json.loads(raw[pn.QUANT_LAYOUT_KEY])["int8_v1"] == pn.INT8_V1_FACTS
    a = pq._load_prequant_checkpoint(str(src), map_location = "cpu")["state_dict"]
    b = pq._load_prequant_checkpoint(str(dst), map_location = "cpu", mmap = True)["state_dict"]
    assert conv.compare_state_dicts(a, b) == []


def test_the_converter_writes_a_text_encoder_with_tied_weights(tmp_path):
    from core.inference.diffusion_te_prequant import TE_PREQUANT_FORMAT

    embed = torch.randn(16, 8).to(torch.float8_e4m3fn)
    sd = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": embed,
        "model.norm.weight": torch.ones(8),
    }
    src = tmp_path / "Model-text_encoder-FP8.pt"
    torch.save(
        {
            "format": TE_PREQUANT_FORMAT,
            "metadata": {"scheme": "fp8", "component": "text_encoder"},
            "state_dict": sd,
        },
        str(src),
    )
    rec = _converter().convert(str(src), str(tmp_path / "out"), repo = None, revision = None, sha = None)
    assert rec["kind"] == "text_encoder" and rec["verified_bit_identical"]
    assert rec["dst"].endswith("Model-text_encoder-FP8.safetensors")
    pth = tmp_path / "Other-text_encoder-FP8.pth"
    pth.write_bytes(src.read_bytes())
    assert (
        _converter()
        .convert(str(pth), str(tmp_path / "out"), repo = None, revision = None, sha = None)["dst"]
        .endswith("Other-text_encoder-FP8.safetensors")
    )
    back = ps.load_plain_prequant_safetensors(rec["dst"])
    assert back["format"] == TE_PREQUANT_FORMAT
    for key in sd:
        assert back["state_dict"][key].dtype == sd[key].dtype and torch.equal(
            back["state_dict"][key].float(), sd[key].float()
        )


def test_the_converters_comparison_sees_every_kind_of_difference():
    conv = _converter()
    w = _int8_tensor()
    assert (
        conv.compare_state_dicts({"a.w": w, "b": torch.zeros(2)}, {"a.w": w, "b": torch.zeros(2)})
        == []
    )
    assert conv.compare_state_dicts({"a.w": w}, {"a.w": _int8_tensor(seed = 1)}) == ["a.w"]
    assert conv.compare_state_dicts({"b": torch.zeros(2)}, {"b": -torch.zeros(2)}) == ["b"]
    assert conv.compare_state_dicts(
        {"b": torch.zeros(2)}, {"b": torch.zeros(2, dtype = torch.float64)}
    ) == ["b"]
    assert conv.compare_state_dicts({"b": torch.zeros(2)}, {"c": torch.zeros(2)}) == ["b", "c"]
    nan = torch.tensor([float("nan")])
    assert conv.compare_state_dicts({"b": nan}, {"b": nan.clone()}) == []


def test_the_converter_refuses_v1_weights_off_the_supported_layout_and_restores_the_decoder(
    tmp_path, monkeypatch, no_foreign_legacy_registrations
):
    import core.inference.prequant_legacy_int8 as li

    conv = _converter()
    original = li._rebuild_weight
    restore = conv._record_v1_facts([])
    assert li._rebuild_weight is not original
    restore()
    assert li._rebuild_weight is original

    def odd_facts(w, standins = None):
        return dict(pn.INT8_V1_FACTS, quant_kwargs = {"reduce_range": False})

    monkeypatch.setattr(conv, "_v1_facts_of", odd_facts)
    src = tmp_path / "Model-INT8.pt"
    if _V1:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias = False)).to(torch.bfloat16)
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        torch.save(
            {
                "format": pq.PREQUANT_FORMAT,
                "metadata": {"scheme": "int8"},
                "state_dict": {"blk.p.weight": model[0].weight},
            },
            str(src),
        )
    elif li.legacy_int8_decode_supported():
        _legacy_writer()._write_legacy_checkpoint(monkeypatch, src)
    else:
        pytest.skip("no v1 classes and no legacy decoder")
    with pytest.raises(ValueError, match = "supported v1 layout"):
        conv.convert(str(src), str(tmp_path / "out"), repo = None, revision = None, sha = None)
    assert li._rebuild_weight is original
