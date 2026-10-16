# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for diffusion LoRA support: the shared helpers, request-model validation, the
native prompt-tag/dir wiring, and the diffusers set_adapters manager."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from core.inference import diffusion_lora as dl


def test_sanitize_alias_strips_path_ext_and_unsafe_chars():
    assert dl.sanitize_alias("My Cool/LoRA v2.safetensors") == "LoRA_v2"
    assert dl.sanitize_alias("owner/repo-name") == "repo-name"
    assert dl.sanitize_alias("weird:<>chars.gguf") == "weird_chars"
    assert dl.sanitize_alias("") == "lora"
    # The alias becomes a PEFT adapter name and PEFT rejects ".".
    assert (
        dl.sanitize_alias("Qwen-Image-2512-Lightning-8steps-V1.0-bf16")
        == "Qwen-Image-2512-Lightning-8steps-V1_0-bf16"
    )
    assert "." not in dl.sanitize_alias("model.v1.0.safetensors")


def test_inject_prompt_tags_appends_with_spacing():
    r = dl.ResolvedLora("id", "style", "/p.safetensors", "safetensors", 0.8)
    assert dl.inject_prompt_tags("a cat", [r]) == "a cat <lora:style:0.8>"
    r1 = dl.ResolvedLora("id", "s", "/p", "safetensors", 1.0)
    assert dl.inject_prompt_tags("x", [r1]) == "x <lora:s:1>"


def test_inject_prompt_tags_validated_weight_overrides_user_typed():
    r = dl.ResolvedLora("id", "style", "/p", "safetensors", 0.8)
    assert dl.inject_prompt_tags("a cat <lora:style:1>", [r]) == "a cat <lora:style:0.8>"


def test_inject_prompt_tags_strips_unselected_user_tags():
    r = dl.ResolvedLora("id", "style", "/p", "safetensors", 0.8)
    # Only selected adapters are materialized, so sd-cli would drop an unselected tag anyway.
    out = dl.inject_prompt_tags("a cat <lora:other:0.5>", [r])
    assert "<lora:other:0.5>" not in out
    assert out == "a cat <lora:style:0.8>"


def test_inject_prompt_tags_empty_returns_prompt():
    assert dl.inject_prompt_tags("hello", []) == "hello"


def test_supports_lora_matrix():
    assert dl.supports_lora(
        engine = "sd_cpp", family = "flux.1", model_kind = "gguf", transformer_quant = None
    )
    assert dl.supports_lora(
        engine = "sd_cpp", family = "z-image", model_kind = "gguf", transformer_quant = None
    )
    assert not dl.supports_lora(
        engine = "sd_cpp", family = "qwen-image", model_kind = "gguf", transformer_quant = None
    )
    assert dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "pipeline", transformer_quant = None
    )
    assert dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = None
    )
    assert dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = "fp8"
    )
    assert dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = "int8"
    )
    # The picker kind stays "gguf" on the quant fast path, so the quant check must run before the gguf check;
    # the bake precedes compilation, so compiled does not gate quant builds.
    assert dl.supports_lora(
        engine = "diffusers",
        family = "z-image",
        model_kind = "gguf",
        transformer_quant = "int8",
        compiled = True,
    )
    assert not dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = "nvfp4"
    )
    assert not dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "single_file", transformer_quant = "mxfp8"
    )
    assert not dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "gguf", transformer_quant = None
    )
    # diffusers needs a non-hotswap adapter loaded before compilation.
    assert not dl.supports_lora(
        engine = "diffusers",
        family = "flux.1",
        model_kind = "pipeline",
        transformer_quant = None,
        compiled = True,
    )
    assert dl.supports_lora(
        engine = "sd_cpp",
        family = "flux.1",
        model_kind = "gguf",
        transformer_quant = None,
        compiled = True,
    )


def test_resolve_specs_maps_cancelled_to_diffusion_sentinel(tmp_path, monkeypatch):
    # A cancelled Hub download raises RuntimeError("Cancelled"); it maps to the sentinel so the route returns 409.
    def _boom(spec_id, weight, **kw):
        raise RuntimeError("Cancelled")

    monkeypatch.setattr(dl, "resolve_one", _boom)
    with pytest.raises(RuntimeError) as ei:
        dl.resolve_specs([("a", 1.0)])
    assert str(ei.value) == dl.DIFFUSION_CANCELLED_MSG

    def _other(spec_id, weight, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(dl, "resolve_one", _other)
    with pytest.raises(RuntimeError) as ei2:
        dl.resolve_specs([("a", 1.0)])
    assert str(ei2.value) == "disk full"


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
    assert aliases == ["a", "a_2"]
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
    local = {e.id: e for e in dl.list_loras() if e.source == "local"}
    assert set(local) == {"mystyle", "other"}
    assert local["other"].fmt == "gguf" and local["mystyle"].fmt == "safetensors"


def test_resolve_one_local_and_unknown(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "mystyle.safetensors").write_bytes(b"x")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    r = dl.resolve_one("mystyle", 0.7)
    assert r.path.endswith("mystyle.safetensors") and r.weight == 0.7
    with pytest.raises(FileNotFoundError):
        dl.resolve_one("does-not-exist", 1.0)


def test_resolve_one_rejects_cross_family_catalog_entry(tmp_path, monkeypatch):
    # Enforced in the resolver, not just the UI picker, so direct API clients cannot bypass it.
    d = tmp_path / "loras"
    d.mkdir()
    (d / "krea-style.safetensors").write_bytes(b"x")
    (d / "krea-style.json").write_text('{"families": ["krea-2"]}', encoding = "utf-8")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    r = dl.resolve_one("krea-style", 0.7, family = "krea-2")
    assert r.path.endswith("krea-style.safetensors")
    with pytest.raises(ValueError):
        dl.resolve_one("krea-style", 0.7, family = "flux.1")
    assert dl.resolve_one("krea-style", 0.7).path.endswith("krea-style.safetensors")


def test_resolve_specs_drops_zero_weight(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    out = dl.resolve_specs([("a", 0.0), ("a", 1.0)])
    assert len(out) == 1 and out[0].weight == 1.0


def test_resolve_specs_maps_unknown_id_to_valueerror(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    with pytest.raises(ValueError):
        dl.resolve_specs([("nope", 1.0)])


def test_resolve_specs_maps_hub_error_to_valueerror(tmp_path, monkeypatch):
    # The Hub message embeds the request URL, which must be scrubbed from the client-facing 400.
    from huggingface_hub.errors import RepositoryNotFoundError

    def _boom(spec_id, weight, **kw):
        # response is optional in huggingface_hub 0.x but required in 1.x; a stub works on either.
        raise RepositoryNotFoundError(
            "404 Client Error. Repository Not Found for url: "
            "https://huggingface.co/api/models/nope/nope (Request ID: abc)",
            response = types.SimpleNamespace(headers = {}, request = None),
        )

    monkeypatch.setattr(dl, "resolve_one", _boom)
    with pytest.raises(ValueError) as ei:
        dl.resolve_specs([("nope/nope", 1.0)])
    assert "http" not in str(ei.value)
    assert "Repository Not Found" in str(ei.value)


def test_scan_local_disambiguates_identical_stems(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "foo.safetensors").write_bytes(b"x")
    (d / "foo.gguf").write_bytes(b"y")
    (d / "solo.safetensors").write_bytes(b"z")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    by_id = {e.id: e for e in dl.list_loras()}
    assert "foo.safetensors" in by_id and "foo.gguf" in by_id
    assert by_id["foo.safetensors"].fmt == "safetensors"
    assert by_id["foo.gguf"].fmt == "gguf"
    assert "solo" in by_id


def test_resolve_one_rejects_traversal_weight_name(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "loras_dir", lambda: tmp_path)
    for bad in ("owner/name:../secret.safetensors", "owner/name:/etc/x.safetensors"):
        with pytest.raises(ValueError):
            dl.resolve_one(bad, 1.0)


def test_lora_spec_and_request_validation():
    from models.inference import DiffusionGenerateRequest, LoraSpec

    assert DiffusionGenerateRequest(prompt = "x").loras is None
    req = DiffusionGenerateRequest(
        prompt = "x", loras = [{"id": "a", "weight": 0.5}, {"id": "b", "weight": 1.0}]
    )
    assert [l.id for l in req.loras] == ["a", "b"]
    with pytest.raises(Exception):
        LoraSpec(id = "a", weight = 3.0)
    with pytest.raises(Exception):
        LoraSpec(id = "a", weight = -0.1)
    assert LoraSpec(id = "a").weight == 1.0
    # Duplicates would load as several suffixed adapters and stack past the weight bound.
    with pytest.raises(Exception):
        DiffusionGenerateRequest(
            prompt = "x", loras = [{"id": "a", "weight": 0.5}, {"id": "a", "weight": 1.0}]
        )


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
    assert getattr(pipe, "_unsloth_loras")


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
    assert pipe.loaded == first_loaded
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
    # int8/fp8 pipes bake adapters at load time; a bake-less quant pipe has frozen topology and must reload.
    import threading

    pipe = _FakePipe()
    with pytest.raises(ValueError, match = "Reload the model with the adapter selection"):
        _backend()._apply_loras(
            _fake_state(pipe, kind = "single_file", quant = "fp8"),
            [("styleA", 1.0)],
            threading.Event(),
        )
    assert pipe.loaded == []


def test_diffusers_apply_rejects_gguf_adapter(monkeypatch):
    import threading

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.gguf", "gguf", w) for i, w in specs
        ],
    )
    pipe = _FakePipe()
    with pytest.raises(ValueError, match = "GGUF LoRA"):
        _backend()._apply_loras(_fake_state(pipe), [("styleA", 1.0)], threading.Event())
    assert pipe.loaded == []


def test_scan_local_reads_family_sidecar(tmp_path, monkeypatch):
    import json

    d = tmp_path / "loras"
    d.mkdir()
    (d / "trained.safetensors").write_bytes(b"x")
    (d / "trained.json").write_text(
        json.dumps({"family": "sdxl", "base_model": "b", "weight_default": 0.8})
    )
    (d / "plain.safetensors").write_bytes(b"y")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)

    by_id = {e.id: e for e in dl.list_loras()}
    assert by_id["trained"].families == ("sdxl",)
    assert by_id["trained"].weight_default == 0.8
    assert by_id["plain"].families == ()
    assert by_id["plain"].weight_default == 1.0

    sdxl_ids = {e.id for e in dl.list_loras(family = "sdxl")}
    flux_ids = {e.id for e in dl.list_loras(family = "flux.1")}
    assert "trained" in sdxl_ids and "plain" in sdxl_ids
    assert "trained" not in flux_ids and "plain" in flux_ids


def test_scan_local_tolerates_bad_sidecar(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x")
    (d / "a.json").write_text("{ not valid json")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    entry = next(e for e in dl.list_loras() if e.id == "a")
    assert entry.families == () and entry.weight_default == 1.0
