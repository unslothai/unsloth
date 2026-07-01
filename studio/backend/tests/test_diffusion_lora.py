"""Tests for diffusion LoRA support: the shared helpers, request-model validation, the
native prompt-tag/dir wiring, and the diffusers set_adapters manager."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from core.inference import diffusion_lora as dl


# ── Pure helpers ────────────────────────────────────────────────────────────


def test_sanitize_alias_strips_path_ext_and_unsafe_chars():
    assert dl.sanitize_alias("My Cool/LoRA v2.safetensors") == "LoRA_v2"
    assert dl.sanitize_alias("owner/repo-name") == "repo-name"
    assert dl.sanitize_alias("weird:<>chars.gguf") == "weird_chars"
    assert dl.sanitize_alias("") == "lora"
    # Internal dots (version tags like "V1.0") must be replaced: the alias becomes a
    # diffusers PEFT adapter name and PEFT rejects "." in module/adapter names.
    assert (
        dl.sanitize_alias("Qwen-Image-2512-Lightning-8steps-V1.0-bf16")
        == "Qwen-Image-2512-Lightning-8steps-V1_0-bf16"
    )
    assert "." not in dl.sanitize_alias("model.v1.0.safetensors")


def test_inject_prompt_tags_appends_with_spacing():
    r = dl.ResolvedLora("id", "style", "/p.safetensors", "safetensors", 0.8)
    assert dl.inject_prompt_tags("a cat", [r]) == "a cat <lora:style:0.8>"
    # weight formatting: 1.0 -> "1", trailing zeros trimmed
    r1 = dl.ResolvedLora("id", "s", "/p", "safetensors", 1.0)
    assert dl.inject_prompt_tags("x", [r1]) == "x <lora:s:1>"


def test_inject_prompt_tags_dedupes_user_typed_tag():
    r = dl.ResolvedLora("id", "style", "/p", "safetensors", 0.8)
    # user already wrote a tag for the same alias -> not duplicated
    assert dl.inject_prompt_tags("a cat <lora:style:1>", [r]) == "a cat <lora:style:1>"


def test_inject_prompt_tags_empty_returns_prompt():
    assert dl.inject_prompt_tags("hello", []) == "hello"


def test_supports_lora_matrix():
    # native: flux/z-image yes, qwen no
    assert dl.supports_lora(
        engine = "sd_cpp", family = "flux.1", model_kind = "gguf", transformer_quant = None
    )
    assert dl.supports_lora(
        engine = "sd_cpp", family = "z-image", model_kind = "gguf", transformer_quant = None
    )
    assert not dl.supports_lora(
        engine = "sd_cpp", family = "qwen-image", model_kind = "gguf", transformer_quant = None
    )
    # diffusers: bf16 yes, fp8/int8 dense no, gguf-diffusers no
    assert dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "pipeline", transformer_quant = None
    )
    assert dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = None
    )
    assert not dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = "fp8"
    )
    assert not dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = "int8"
    )
    assert not dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "gguf", transformer_quant = None
    )


def test_materialize_native_dir_symlinks_and_breaks_collisions(tmp_path):
    a = tmp_path / "a.safetensors"
    a.write_bytes(b"x")
    b = tmp_path / "sub"
    b.mkdir()
    b2 = b / "a.safetensors"  # same stem as `a` -> alias collision
    b2.write_bytes(b"y")
    resolved = [
        dl.ResolvedLora("a", "a", str(a), "safetensors", 1.0),
        dl.ResolvedLora("a2", "a", str(b2), "safetensors", 0.5),
    ]
    dest = tmp_path / "managed"
    out = dl.materialize_native_dir(resolved, dest)
    aliases = [r.alias for r in out]
    assert aliases == ["a", "a_2"]  # collision broken
    for r in out:
        assert os.path.exists(r.path)
        assert Path(r.path).parent == dest


def test_list_loras_scans_local(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "mystyle.safetensors").write_bytes(b"x")
    (d / "other.gguf").write_bytes(b"y")
    (d / "ignore.txt").write_bytes(b"z")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    ids = {e.id for e in dl.list_loras()}
    assert ids == {"mystyle", "other"}
    fmts = {e.id: e.fmt for e in dl.list_loras()}
    assert fmts["other"] == "gguf" and fmts["mystyle"] == "safetensors"


def test_resolve_one_local_and_unknown(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "mystyle.safetensors").write_bytes(b"x")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    r = dl.resolve_one("mystyle", 0.7)
    assert r.path.endswith("mystyle.safetensors") and r.weight == 0.7
    with pytest.raises(FileNotFoundError):
        dl.resolve_one("does-not-exist", 1.0)


def test_resolve_specs_drops_zero_weight(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    out = dl.resolve_specs([("a", 0.0), ("a", 1.0)])
    assert len(out) == 1 and out[0].weight == 1.0


# ── Request-model validation ────────────────────────────────────────────────


def test_lora_spec_and_request_validation():
    from models.inference import DiffusionGenerateRequest, LoraSpec

    # empty / missing loras -> unchanged behaviour
    assert DiffusionGenerateRequest(prompt = "x").loras is None
    req = DiffusionGenerateRequest(
        prompt = "x", loras = [{"id": "a", "weight": 0.5}, {"id": "b", "weight": 1.0}]
    )
    assert [l.id for l in req.loras] == ["a", "b"]
    # weight bounds enforced
    with pytest.raises(Exception):
        LoraSpec(id = "a", weight = 3.0)
    with pytest.raises(Exception):
        LoraSpec(id = "a", weight = -0.1)
    # default weight
    assert LoraSpec(id = "a").weight == 1.0


# ── Diffusers apply manager ─────────────────────────────────────────────────


class _FakePipe:
    def __init__(self):
        self.loaded: list[tuple[str, str]] = []
        self.active = None
        self.unloaded = 0

    def load_lora_weights(
        self,
        path,
        adapter_name = None,
    ):
        self.loaded.append((path, adapter_name))

    def set_adapters(
        self,
        names,
        adapter_weights = None,
    ):
        self.active = (list(names), list(adapter_weights) if adapter_weights else None)

    def unload_lora_weights(self):
        self.unloaded += 1
        self.loaded = []
        self.active = None


def _fake_state(
    pipe,
    *,
    kind = "pipeline",
    quant = None,
):
    fam = types.SimpleNamespace(name = "flux.1")
    return types.SimpleNamespace(
        pipe = pipe, family = fam, kind = kind, transformer_quant = quant, hf_token = None
    )


def _backend():
    from core.inference.diffusion import DiffusionBackend
    return DiffusionBackend()


def test_diffusers_apply_loads_and_sets_adapters(monkeypatch):
    import threading

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w)
            for i, w in specs
        ],
    )
    pipe = _FakePipe()
    _backend()._apply_loras(
        _fake_state(pipe), [("styleA", 0.8), ("styleB", 1.0)], threading.Event()
    )
    assert [n for _p, n in pipe.loaded] == ["styleA", "styleB"]
    assert pipe.active == (["styleA", "styleB"], [0.8, 1.0])
    assert getattr(pipe, "_unsloth_loras")  # marker recorded


def test_diffusers_apply_noop_when_unchanged(monkeypatch):
    import threading

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w)
            for i, w in specs
        ],
    )
    pipe = _FakePipe()
    b = _backend()
    b._apply_loras(_fake_state(pipe), [("styleA", 0.8)], threading.Event())
    first_loaded = list(pipe.loaded)
    b._apply_loras(_fake_state(pipe), [("styleA", 0.8)], threading.Event())
    assert pipe.loaded == first_loaded  # not reloaded
    assert pipe.unloaded == 0


def test_diffusers_apply_clears_when_empty(monkeypatch):
    import threading

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w)
            for i, w in specs
        ],
    )
    pipe = _FakePipe()
    b = _backend()
    b._apply_loras(_fake_state(pipe), [("styleA", 0.8)], threading.Event())
    b._apply_loras(_fake_state(pipe), [], threading.Event())
    assert pipe.unloaded == 1
    assert pipe._unsloth_loras == ()


def test_diffusers_apply_rejects_unsupported_quant():
    import threading
    pipe = _FakePipe()
    with pytest.raises(ValueError, match = "not supported"):
        _backend()._apply_loras(
            _fake_state(pipe, kind = "single_file", quant = "fp8"),
            [("styleA", 1.0)],
            threading.Event(),
        )
