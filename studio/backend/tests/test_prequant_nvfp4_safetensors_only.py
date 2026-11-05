# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVFP4 diffusion checkpoints load from .safetensors only: no .pt / .pth is resolved, preferred from the
cache, accepted as a local override or unpickled. int8 / fp8 keep their pickle containers and fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.inference.diffusion_prequant as pq
from core.inference.diffusion_families import detect_family
from core.inference.video_families import detect_video_family

# The files the hosted unsloth/*-NVFP4 diffusion repos hold (Hub listing), keyed by (base, task).
HOSTED = {
    ("image", "Tongyi-MAI/Z-Image-Turbo", None): "Z-Image-Turbo-NVFP4.safetensors",
    ("image", "Qwen/Qwen-Image-2.1", None): "Qwen-Image-2.1-NVFP4.safetensors",
    ("image", "black-forest-labs/FLUX.1-schnell", None): "FLUX.1-schnell-NVFP4.safetensors",
    ("image", "Qwen/Qwen-Image-2512", None): "Qwen-Image-2512-NVFP4.safetensors",
    ("video", "Wan-AI/Wan2.2-TI2V-5B-Diffusers", None): "Wan2.2-TI2V-5B-NVFP4.safetensors",
    ("video", "Wan-AI/Wan2.2-T2V-A14B-Diffusers", None): "Wan2.2-T2V-A14B-NVFP4.safetensors",
    (
        "video",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "transformer_2",
    ): "Wan2.2-T2V-A14B-transformer_2-NVFP4.safetensors",
    (
        "video",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        None,
    ): "HunyuanVideo-1.5-Diffusers-480p_t2v-NVFP4.safetensors",
    (
        "video",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        None,
    ): "HunyuanVideo-1.5-Diffusers-720p_t2v-NVFP4.safetensors",
}
PICKLES = (".pt", ".pth")


@pytest.fixture(autouse = True)
def _nvfp4_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    for name in (pq.SAFETENSORS_TWINS_ENV, pq.PREFER_SAFETENSORS_ENV):
        monkeypatch.delenv(name, raising = False)


def _family(kind, base):
    fam = detect_family(base) if kind == "image" else detect_video_family(base)
    assert fam is not None, base
    return fam


@pytest.mark.parametrize("twins", [None, "0"], ids = ["twins_on", "twins_off"])
@pytest.mark.parametrize("key", sorted(HOSTED, key = str), ids = lambda k: f"{k[1]}-{k[2]}")
def test_every_hosted_nvfp4_chain_leads_with_the_hosted_safetensors_and_has_no_pickle(
    key, twins, monkeypatch
):
    if twins is not None:
        monkeypatch.setenv(pq.SAFETENSORS_TWINS_ENV, twins)
    kind, base, task = key
    src = pq.resolve_prequant_source(_family(kind, base), "nvfp4", base_repo = base, task = task)
    assert src is not None and src.kind == "repo"
    chain = pq.candidate_filenames_of(src)
    assert chain[0] == HOSTED[key]
    assert not [n for n in chain if n.lower().endswith(PICKLES)], chain
    # declared, so planning bets on it without needing the file already cached
    assert HOSTED[key] in src.declared_filenames


def test_the_hosted_repo_set_is_covered():
    repos = {
        pq.resolve_prequant_source(_family(k, b), "nvfp4", base_repo = b, task = t).location.lower()
        for k, b, t in HOSTED
    }
    assert repos == set(pq.hosted_nvfp4_repo_ids())


def test_nvfp4_derives_only_the_safetensors_name():
    assert pq.derived_prequant_filenames("unsloth/Model-NVFP4", "nvfp4") == (
        "Model-NVFP4.safetensors",
    )
    assert pq.derived_prequant_filenames(
        "unsloth/Model-NVFP4", "nvfp4", component = "transformer_2"
    ) == ("Model-transformer_2-NVFP4.safetensors",)


def test_a_family_still_declaring_an_nvfp4_pickle_resolves_only_its_twin():
    """A stale or third-party table row naming a .pt: the twin is requested, never the pickle."""
    fam = _family("video", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    stale = type(fam).__new__(type(fam))
    stale.__dict__.update(fam.__dict__)
    object.__setattr__(
        stale,
        "prequant_filenames",
        (
            ("nvfp4", "Wan2.2-T2V-A14B-NVFP4.pt"),
            ("nvfp4", "transformer_2", "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"),
        ),
    )
    for task, want in (
        (None, "Wan2.2-T2V-A14B-NVFP4.safetensors"),
        ("transformer_2", "Wan2.2-T2V-A14B-transformer_2-NVFP4.safetensors"),
    ):
        src = pq.resolve_prequant_source(stale, "nvfp4", task = task)
        chain = pq.candidate_filenames_of(src)
        assert chain[0] == want
        assert not [n for n in chain if n.lower().endswith(PICKLES)], chain
        assert src.declared_filenames == (want,)


@pytest.mark.parametrize(
    "base,scheme,pickle",
    [
        ("Tongyi-MAI/Z-Image-Turbo", "int8", "Z-Image-Turbo-INT8.pt"),
        ("Tongyi-MAI/Z-Image-Turbo", "fp8", "Z-Image-Turbo-FP8.pt"),
    ],
)
def test_int8_and_fp8_chains_keep_their_pickle_fallbacks(base, scheme, pickle):
    src = pq.resolve_prequant_source(_family("image", base), scheme, base_repo = base)
    chain = pq.candidate_filenames_of(src)
    assert pickle in chain
    assert f"transformer_{scheme}.pt" in chain


def test_wan_int8_second_expert_keeps_its_declared_pickle_behind_its_twin():
    fam = _family("video", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    src = pq.resolve_prequant_source(fam, "int8", task = "transformer_2")
    assert pq.candidate_filenames_of(src) == (
        "Wan2.2-T2V-A14B-INT8-2.safetensors",
        "Wan2.2-T2V-A14B-INT8-2.pt",
    )


def test_only_nvfp4_pickles_are_unreadable():
    for name in ("x-NVFP4.pt", "x-NVFP4.pth", "transformer_nvfp4.pt"):
        assert pq.restricted_prequant_load_supported("nvfp4", name) is False
    assert pq.nvfp4_pickle_name("nvfp4", "x.PT") is True
    assert pq.nvfp4_pickle_name("int8", "x.pt") is False
    assert pq.nvfp4_pickle_name("nvfp4", "x.safetensors") is False
    assert pq.without_nvfp4_pickles("int8", ["a.safetensors", "a.pt"]) == ["a.safetensors", "a.pt"]
    assert pq.without_nvfp4_pickles("nvfp4", ["a.safetensors", "a.pt", "b.pth"]) == [
        "a.safetensors"
    ]


def test_the_pickle_allowlist_names_no_nvfp4_class():
    assert not any("nvfp4" in module.lower() for module, _name in pq._PREQUANT_SAFE_GLOBALS)
    assert "nvfp4" not in pq._SCHEME_REQUIRED_GLOBALS
    assert {"int8", "fp8"} <= set(pq._SCHEME_REQUIRED_GLOBALS)


def _pickle(path: Path, scheme: str) -> Path:
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "format": pq.PREQUANT_FORMAT,
            "metadata": {"scheme": scheme, "component": "transformer"},
            "state_dict": {"blk.proj.weight": torch.zeros(4, 4)},
        },
        str(path),
    )
    return path


def test_the_loader_refuses_an_nvfp4_pickle(tmp_path):
    path = _pickle(tmp_path / "Model-NVFP4.pt", "nvfp4")
    # asked for nvfp4: refused on the extension, before any unpickling
    with pytest.raises(ValueError, match = "safetensors only"):
        pq._load_prequant_checkpoint(str(path), scheme = "nvfp4", map_location = "cpu")
    # not told the scheme: the recorded nvfp4 metadata refuses it
    with pytest.raises(ValueError, match = "safetensors only"):
        pq._load_prequant_checkpoint(str(path), map_location = "cpu")
    with pytest.raises(ValueError, match = "safetensors only"):
        pq._read_prequant_for(str(path), "cpu", scheme = "nvfp4")


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_the_loader_still_reads_an_int8_or_fp8_pickle(tmp_path, scheme):
    path = _pickle(tmp_path / f"Model-{scheme.upper()}.pt", scheme)
    ckpt = pq._load_prequant_checkpoint(str(path), map_location = "cpu")
    assert ckpt["metadata"]["scheme"] == scheme
    assert pq.local_prequant_scheme(str(path)) == scheme
    assert pq.restricted_prequant_load_supported(scheme, path.name) is True


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_naming_the_scheme_still_reads_an_int8_or_fp8_pickle(tmp_path, scheme):
    path = _pickle(tmp_path / f"Model-{scheme.upper()}.pt", scheme)
    ckpt = pq._load_prequant_checkpoint(str(path), scheme = scheme, map_location = "cpu")
    assert ckpt["metadata"]["scheme"] == scheme


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_an_int8_or_fp8_pickle_override_is_still_a_source(tmp_path, monkeypatch, scheme):
    fam = _family("image", "Tongyi-MAI/Z-Image-Turbo")
    path = _pickle(tmp_path / f"Model-{scheme.upper()}.pt", scheme)
    got = pq.resolve_prequant_source(fam, scheme, path_override = str(path))
    assert got is not None and got.kind == "path"
    assert pq._resolve_checkpoint_path(got, None, scheme = scheme) == str(path)


def test_a_local_nvfp4_pickle_never_reads_as_nvfp4(tmp_path):
    path = _pickle(tmp_path / "Model-NVFP4.pt", "nvfp4")
    assert pq.local_prequant_scheme(str(path)) is None


def test_a_local_nvfp4_pickle_override_is_refused(tmp_path, monkeypatch):
    fam = _family("image", "Tongyi-MAI/Z-Image-Turbo")
    path = _pickle(tmp_path / "Model-NVFP4.pt", "nvfp4")
    assert pq.resolve_prequant_source(fam, "nvfp4", path_override = str(path)) is None
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    assert pq.usable_prequant_source(fam, "nvfp4", path_override = str(path)) is None
    # a hand-built source still never reaches the unpickler
    src = pq.PrequantSource(kind = "path", location = str(path))
    with pytest.raises(ValueError, match = "safetensors only"):
        pq._resolve_checkpoint_path(src, None, scheme = "nvfp4")


def test_the_full_loader_reports_the_refused_nvfp4_pickle(tmp_path, monkeypatch):
    path = _pickle(tmp_path / "Model-NVFP4.pt", "nvfp4")
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    opened = []
    monkeypatch.setattr(pq, "_torch_load_prequant", lambda *a, **k: opened.append(a))
    got = pq.load_prequantized_transformer(
        object,
        "Tongyi-MAI/Z-Image-Turbo",
        pq.PrequantSource(kind = "path", location = str(path)),
        device = "cpu",
        dtype = "bfloat16",
        hf_token = None,
        scheme = "nvfp4",
    )
    assert got is None and opened == []
    assert "safetensors only" in (pq.last_prequant_failure() or "")


def test_nvfp4_detection_never_unpickles(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("an nvfp4 probe unpickled a checkpoint")

    _pickle(tmp_path / "Model-NVFP4.pt", "nvfp4")
    monkeypatch.setattr(pq, "_torch_load_prequant", _boom)
    folder = tmp_path
    assert pq._artifacts_declare_nvfp4(str(folder)) is False
    # a single pickle file is judged by its name, never opened
    assert pq.declares_nvfp4_checkpoint(str(tmp_path / "Model-NVFP4.pt")) is True
    other = tmp_path / "Model-INT8.pt"
    other.write_bytes(b"x")
    assert pq.declares_nvfp4_checkpoint(str(other)) is False


def test_the_gate_records_name_the_hosted_safetensors():
    record = json.loads(
        (Path(pq.__file__).parent / "nvfp4_gate_record.json").read_text(encoding = "utf-8")
    )
    entries = record.get("records") or record.get("entries") or []
    assert entries
    for entry in entries:
        assert entry["filename"].endswith(".safetensors"), entry["filename"]
        assert entry["filename"] in HOSTED.values()
        assert len(entry["checkpoint_sha256"]) == 64
