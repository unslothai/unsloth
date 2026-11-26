# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""float16 cards: an automatic pick takes Qwen-Image-2.1's hosted int8 ConvRot encoder only when the dense float16
load cannot be held in host RAM, priced by the small-host route's own resolver; explicit picks and the switch win."""

from __future__ import annotations

import json
import struct
import types

import pytest

torch = pytest.importorskip("torch")

# imported before any fixture patches diffusion_small_host: diffusion.py binds its names at import
import core.inference.diffusion as D  # noqa: E402
import core.inference.diffusion_small_host as sh  # noqa: E402
import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.inference.diffusion_precision import AutoTeQuant  # noqa: E402

BASE = "Qwen/Qwen-Image-2.1"
INT8_NAME = "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors"
CUDA_FP16 = types.SimpleNamespace(device = "cuda", dtype = torch.float16)
SWITCH = "UNSLOTH_TE_INT8_FP16"
GIB = 1024
# host readings (total, available) MiB: a 51 GB Colab high-RAM T4 (as measured: 49760 MiB available before the load)
# and a 31 GB Kaggle T4 host
HOST_51 = (52 * GIB, 49760)
HOST_31 = (31 * GIB, 28 * GIB)


def _fam():
    return detect_family("", override = "qwen-image-2.1")


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(SWITCH, raising = False)
    monkeypatch.delenv(sh.HOST_RAM_LIMIT_ENV, raising = False)
    monkeypatch.delenv(sh.SMALL_HOST_ENV, raising = False)
    monkeypatch.setattr(sh, "host_commit_mib", lambda: None)
    # no local snapshot: the family's bf16 table prices the load
    monkeypatch.setattr(sh, "resolve_snapshot_dir", lambda repo, cache_dir = None: None)


def _host(monkeypatch, reading):
    monkeypatch.setattr(sh, "host_ram_mib", lambda: reading)


def _sources(mode):
    src = tpq.te_prequant_sources_for_base(_fam(), BASE, te_quant_mode = mode, target = CUDA_FP16)
    return src.get("text_encoder")


def test_auto_on_a_51_gb_host_keeps_the_dense_encoder(monkeypatch):
    _host(monkeypatch, HOST_51)
    assert _sources(AutoTeQuant("int8")) is None
    fits, why = tpq.dense_fp16_encoder_fits_host(_fam(), BASE)
    assert fits is True and "MiB" in why


def test_auto_on_a_31_gb_host_takes_the_int8_encoder(monkeypatch):
    _host(monkeypatch, HOST_31)
    src = _sources(AutoTeQuant("int8"))
    assert src is not None and src.filename == INT8_NAME
    assert tpq.te_candidate_filenames(src) == (INT8_NAME,)


def test_auto_inside_an_unset_request_flips_the_same_way(monkeypatch):
    for reading, int8 in ((HOST_51, False), (HOST_31, True)):
        _host(monkeypatch, reading)
        with tpq.auto_te_request(True):
            assert (_sources("int8") is not None) is int8


def test_an_explicit_int8_request_is_honoured_on_a_large_host(monkeypatch):
    _host(monkeypatch, HOST_51)
    src = _sources("int8")
    assert src is not None and src.filename == INT8_NAME


def test_the_switch_wins_both_ways(monkeypatch):
    _host(monkeypatch, HOST_51)
    monkeypatch.setenv(SWITCH, "1")
    assert _sources(AutoTeQuant("int8")) is not None
    _host(monkeypatch, HOST_31)
    monkeypatch.setenv(SWITCH, "0")
    assert _sources(AutoTeQuant("int8")) is None
    assert _sources("int8") is None


def test_unreadable_host_ram_keeps_the_int8_encoder(monkeypatch):
    _host(monkeypatch, (None, None))
    assert _sources(AutoTeQuant("int8")) is not None


def test_bf16_cards_are_unchanged(monkeypatch):
    _host(monkeypatch, HOST_51)
    bf16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    assert not tpq.int8_convrot_te_runs_fp16(_fam(), bf16, auto = True, base = BASE)


def _write_component(
    root,
    name,
    mib,
    dtype = "BF16",
    library = "transformers",
):
    """Header only: ``stored_components`` sizes a component from its safetensors header's ``data_offsets``, so the
    body is never written (no multi-GB allocation, which NTFS cannot keep sparse)."""
    sub = root / name
    sub.mkdir()
    numel = mib * 1024 * 1024 // 2
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [numel], "data_offsets": [0, numel * 2]}}
    ).encode()
    with open(sub / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)
    return library


def _assert_small(root):
    """The fixture snapshot stays tiny on disk (under 1 MB), whatever sizes its headers claim."""
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    assert total < 1024 * 1024, total


@pytest.mark.parametrize("reading", [HOST_51, HOST_31, (24 * GIB, 20 * GIB), (64 * GIB, 60 * GIB)])
def test_the_pick_agrees_with_the_small_host_decision_on_the_same_snapshot(
    tmp_path, monkeypatch, reading
):
    """Same components, same host reading: int8 exactly when the dense load would take the low-RAM route."""
    index = {"_class_name": "QwenImage21Pipeline"}
    for name, mib in (("transformer", 13540), ("text_encoder", 16700), ("vae", 650)):
        index[name] = [_write_component(tmp_path, name, mib), "X"]
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    _assert_small(tmp_path)
    assert sh.stored_components(tmp_path)["text_encoder"].mib == 16700
    monkeypatch.setattr(sh, "resolve_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    _host(monkeypatch, reading)
    decision = sh.decide_small_host(
        sh.stored_components(tmp_path),
        torch.float16,
        device = "cuda",
        host_total_mib = reading[0],
        host_available_mib = reading[1],
        host_commit_mib = None,
        pread_libraries = ("transformers",),
    )
    int8 = _sources(AutoTeQuant("int8")) is not None
    assert int8 is decision.engaged
    fits, dense = sh.dense_load_fits(
        sh.stored_components(tmp_path),
        torch.float16,
        host_total_mib = reading[0],
        host_available_mib = reading[1],
    )
    assert dense == decision.dense_host_mib and fits is (not decision.engaged)


# ------------------------------------------------------------------------------- one pick per load, carried to assembly

# the re-check repro: 48 GiB host, available RAM rises from 35,000 to 45,000 MiB between staging and assembly
HOST_PLAN = (48 * GIB, 35000)
HOST_ASSEMBLY = (48 * GIB, 45000)


def test_a_staged_int8_pick_survives_a_ram_rise_before_assembly(monkeypatch):
    _host(monkeypatch, HOST_PLAN)
    pick = tpq.decide_fp16_te_int8(_fam(), BASE)
    assert pick is True
    with tpq.fp16_te_pick_scope(pick):
        staged = _sources(AutoTeQuant("int8"))
    assert staged is not None and staged.filename == INT8_NAME
    _host(monkeypatch, HOST_ASSEMBLY)
    # a fresh reading alone would now pick dense: exactly the flip that left assembly without the skipped dense shards
    assert tpq.decide_fp16_te_int8(_fam(), BASE) is False
    with tpq.fp16_te_pick_scope(pick):
        assembled = _sources(AutoTeQuant("int8"))
    assert assembled is not None and assembled.filename == staged.filename


def test_a_staged_dense_pick_survives_a_ram_drop_before_assembly(monkeypatch):
    _host(monkeypatch, HOST_ASSEMBLY)
    pick = tpq.decide_fp16_te_int8(_fam(), BASE)
    assert pick is False
    _host(monkeypatch, HOST_PLAN)
    with tpq.fp16_te_pick_scope(pick):
        assert _sources(AutoTeQuant("int8")) is None


def test_load_pipeline_wrapper_carries_the_staged_pick(monkeypatch):
    """_run_load passes ``_te_fp16_int8`` into load_pipeline; the wrapper pops it and the whole load runs under it."""
    seen = []

    def fake_load(self, repo_id, **kwargs):
        assert "_te_fp16_int8" not in kwargs
        seen.append(_sources(AutoTeQuant("int8")))
        _host(monkeypatch, HOST_PLAN)
        seen.append(_sources(AutoTeQuant("int8")))
        return {}

    _host(monkeypatch, HOST_ASSEMBLY)
    D._with_supplied_components(fake_load)(None, BASE, text_encoder_quant = None, _te_fp16_int8 = True)
    assert all(s is not None and s.filename == INT8_NAME for s in seen)


def test_a_direct_load_decides_once_and_keeps_it(monkeypatch):
    """No staged pick (a direct load_pipeline call): the first resolver call decides, later ones reuse it."""
    _host(monkeypatch, HOST_ASSEMBLY)
    with tpq.fp16_te_pick_scope(None):
        first = _sources(AutoTeQuant("int8"))
        _host(monkeypatch, HOST_PLAN)
        second = _sources(AutoTeQuant("int8"))
    assert first is None and second is None


# --------------------------------------------------------------- the selector prices the loader's own component set


def _snapshot(root, present):
    index = {"_class_name": "QwenImage21Pipeline"}
    for name, mib in (("transformer", 13540), ("text_encoder", 16700), ("vae", 650)):
        index[name] = ["diffusers" if name != "text_encoder" else "transformers", "X"]
        if name in present:
            _write_component(root, name, mib)
    (root / "model_index.json").write_text(json.dumps(index))
    _assert_small(root)
    return root


def _full(tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    return _snapshot(full, ("transformer", "text_encoder", "vae"))


def _loader_engaged(snapshot, supplied, reading):
    return sh.decide_small_host(
        sh.small_host_components(sh.stored_components(snapshot), supplied),
        torch.float16,
        device = "cuda",
        host_total_mib = reading[0],
        host_available_mib = reading[1],
        host_commit_mib = None,
        pread_libraries = ("transformers",),
    ).engaged


def test_a_partial_cache_is_priced_like_the_full_load(tmp_path, monkeypatch):
    """Only the encoder cached: the selector must not price the encoder alone (the loader downloads the rest first and
    prices all of it). 28 GiB available: the full dense load does not fit, so int8, as the loader would route it."""
    partial = tmp_path / "partial"
    partial.mkdir()
    _snapshot(partial, ("text_encoder",))
    reading = (31 * GIB, 28 * GIB)
    monkeypatch.setattr(sh, "resolve_snapshot_dir", lambda repo, cache_dir = None: partial)
    _host(monkeypatch, reading)
    fits, why = tpq.dense_fp16_encoder_fits_host(_fam(), BASE)
    assert fits is False and "family table" in why
    assert (
        tpq.decide_fp16_te_int8(_fam(), BASE)
        is _loader_engaged(_full(tmp_path), (), reading)
        is True
    )


def test_a_supplied_gguf_denoiser_is_left_out_like_the_loader_does(tmp_path, monkeypatch):
    """Every header cached and a GGUF supplies the denoiser: the loader prices encoder + VAE (~19,952 MiB), which fits
    28 GiB, so the dense encoder; counting the dense DiT would have picked int8."""
    full = _full(tmp_path)
    reading = (31 * GIB, 28 * GIB)
    monkeypatch.setattr(sh, "resolve_snapshot_dir", lambda repo, cache_dir = None: full)
    _host(monkeypatch, reading)
    fits, why = tpq.dense_fp16_encoder_fits_host(_fam(), BASE, ("transformer",))
    assert (
        fits is True
        and "~19952 MiB" in why
        and "transformer" not in why.split("(")[1].split(";")[0]
    )
    assert tpq.decide_fp16_te_int8(_fam(), BASE, ("transformer",)) is False
    assert _loader_engaged(full, ("transformer",), reading) is False
    # without the supplied set it would disagree with the loader
    assert tpq.decide_fp16_te_int8(_fam(), BASE) is True
    with tpq.fp16_te_pick_scope(None, ("transformer",)):
        assert _sources(AutoTeQuant("int8")) is None


@pytest.mark.parametrize(
    "kind, seeded, hosted_vae, replaced, want",
    [
        ("gguf", None, False, (), ("transformer",)),
        ("single_file", None, True, (), ("transformer", "vae")),
        ("pipeline", None, False, (), ()),
        ("pipeline", "int8", False, (), ("transformer",)),
        ("pipeline", None, False, ("vae", "text_encoder"), ("vae",)),
    ],
)
def test_the_planner_derives_the_loader_supplied_set(
    monkeypatch, kind, seeded, hosted_vae, replaced, want
):
    monkeypatch.setattr(D, "resolve_diffusion_device_target", lambda ordinal = None: CUDA_FP16)
    decided = []
    monkeypatch.setattr(
        tpq, "decide_fp16_te_int8", lambda fam, base, supplied: decided.append(supplied) or False
    )
    stub = types.SimpleNamespace(_pipeline_planned_denoiser_scheme = lambda fam, **k: seeded)
    pick, supplied = D.DiffusionBackend._te_fp16_pick(
        stub, _fam(), BASE, kind, {"repo_id": BASE}, AutoTeQuant("int8"), replaced, hosted_vae
    )
    assert pick is False and supplied == want and decided == [want]


def test_the_planner_leaves_explicit_and_switched_requests_alone(monkeypatch):
    monkeypatch.setattr(D, "resolve_diffusion_device_target", lambda ordinal = None: CUDA_FP16)
    stub = types.SimpleNamespace(_pipeline_planned_denoiser_scheme = lambda fam, **k: None)
    args = (_fam(), BASE, "pipeline", {"repo_id": BASE})
    assert D.DiffusionBackend._te_fp16_pick(stub, *args, "int8", (), False) == (None, ())
    monkeypatch.setenv(SWITCH, "1")
    assert D.DiffusionBackend._te_fp16_pick(stub, *args, AutoTeQuant("int8"), (), False) == (
        None,
        (),
    )


def _write_shard(
    sub,
    filename,
    mib,
    dtype = "BF16",
):
    numel = mib * 1024 * 1024 // 2
    header = json.dumps(
        {f"w_{filename}": {"dtype": dtype, "shape": [numel], "data_offsets": [0, numel * 2]}}
    ).encode()
    with open(sub / filename, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)


def test_an_interrupted_encoder_download_is_not_priced_from_its_cached_shard(tmp_path, monkeypatch):
    """One of the encoder's two shards cached (an interrupted download): ``stored_components`` would see a 4,000 MiB
    encoder and the dense load would 'fit' a 31 GiB host. The shard index lists a missing shard, so the snapshot is
    incomplete and the family table prices the load: int8, as the loader routes the complete download."""
    root = tmp_path / "interrupted"
    root.mkdir()
    index = {"_class_name": "QwenImage21Pipeline"}
    for name, mib in (("transformer", 13540), ("vae", 650)):
        index[name] = [_write_component(root, name, mib), "X"]
    index["text_encoder"] = ["transformers", "X"]
    te = root / "text_encoder"
    te.mkdir()
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    (te / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}})
    )
    _write_shard(te, shards[0], 4000)
    (root / "model_index.json").write_text(json.dumps(index))
    _assert_small(root)
    assert sh.stored_components(root)["text_encoder"].mib == 4000
    assert not sh.snapshot_weights_complete(root)
    reading = (31 * GIB, 28 * GIB)
    monkeypatch.setattr(sh, "resolve_snapshot_dir", lambda repo, cache_dir = None: root)
    _host(monkeypatch, reading)
    # the cached shard alone would fit: the bug the completeness gate prevents
    assert (
        sh.dense_load_fits(
            sh.stored_components(root),
            torch.float16,
            host_total_mib = reading[0],
            host_available_mib = reading[1],
        )[0]
        is True
    )
    fits, why = tpq.dense_fp16_encoder_fits_host(_fam(), BASE)
    assert fits is False and "family table" in why
    assert tpq.decide_fp16_te_int8(_fam(), BASE) is True
    src = _sources(AutoTeQuant("int8"))
    assert src is not None and src.filename == INT8_NAME
    # once the second shard lands the snapshot is complete and priced from its headers
    _write_shard(te, shards[1], 12700)
    assert sh.snapshot_weights_complete(root)
    fits, why = tpq.dense_fp16_encoder_fits_host(_fam(), BASE)
    assert fits is False and "snapshot" in why
