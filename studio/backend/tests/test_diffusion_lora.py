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


def test_inject_prompt_tags_validated_weight_overrides_user_typed():
    r = dl.ResolvedLora("id", "style", "/p", "safetensors", 0.8)
    # A user-typed tag for a SELECTED adapter is replaced by the backend-validated weight
    # (so the recorded/validated 0-2 weight wins over whatever was typed), not duplicated.
    assert dl.inject_prompt_tags("a cat <lora:style:1>", [r]) == "a cat <lora:style:0.8>"


def test_inject_prompt_tags_strips_unselected_user_tags():
    r = dl.ResolvedLora("id", "style", "/p", "safetensors", 0.8)
    # A user tag for an alias that is NOT selected is stripped: only selected adapters are
    # materialized in the managed --lora-model-dir, so sd-cli would drop the dead tag anyway;
    # removing it keeps the prompt clean and unambiguous.
    out = dl.inject_prompt_tags("a cat <lora:other:0.5>", [r])
    assert "<lora:other:0.5>" not in out
    assert out == "a cat <lora:style:0.8>"


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
    # diffusers: bf16 yes, torchao int8/fp8 yes (load-time bake), nvfp4/mxfp8 no,
    # gguf-diffusers no
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
    # The quant fast path keeps the PICKER kind ("gguf") while the effective transformer is a
    # dense torchao build, so the quant check must decide BEFORE the gguf-kind check; and the
    # bake precedes compilation by construction, so compiled does not gate quant builds.
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
    # A torch.compile'd diffusers transformer (Speed=default/max) can't take a non-hotswap
    # adapter: diffusers needs the adapter loaded before compilation.
    assert not dl.supports_lora(
        engine = "diffusers",
        family = "flux.1",
        model_kind = "pipeline",
        transformer_quant = None,
        compiled = True,
    )
    # compiled is diffusers-only; the native path ignores it.
    assert dl.supports_lora(
        engine = "sd_cpp",
        family = "flux.1",
        model_kind = "gguf",
        transformer_quant = None,
        compiled = True,
    )


def test_resolve_specs_maps_cancelled_to_diffusion_sentinel(tmp_path, monkeypatch):
    # A Hub download cancelled mid-flight raises RuntimeError("Cancelled"); resolve_specs
    # must convert it to the diffusion cancellation sentinel so the route maps it to 409,
    # not a generic 500 server-error toast.
    def _boom(spec_id, weight, **kw):
        raise RuntimeError("Cancelled")

    monkeypatch.setattr(dl, "resolve_one", _boom)
    with pytest.raises(RuntimeError) as ei:
        dl.resolve_specs([("a", 1.0)])
    assert str(ei.value) == dl.DIFFUSION_CANCELLED_MSG

    # A non-cancellation RuntimeError is left untouched.
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
    # The merged catalog also carries the curated hub entries; the local scan is
    # exactly the weight files dropped in the directory.
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


def test_resolve_specs_drops_zero_weight(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    out = dl.resolve_specs([("a", 0.0), ("a", 1.0)])
    assert len(out) == 1 and out[0].weight == 1.0


def test_resolve_specs_maps_unknown_id_to_valueerror(tmp_path, monkeypatch):
    # An unknown / stale id raises FileNotFoundError in resolve_one; resolve_specs must
    # surface it as ValueError so the route returns 400, not a generic 500.
    d = tmp_path / "loras"
    d.mkdir()
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    with pytest.raises(ValueError):
        dl.resolve_specs([("nope", 1.0)])


def test_resolve_specs_maps_hub_error_to_valueerror(tmp_path, monkeypatch):
    # A mistyped Hub repo id makes the Hub resolution raise a huggingface_hub client error
    # (RepositoryNotFoundError, an HfHubHTTPError). resolve_specs must surface it as
    # ValueError so the route returns 400, not a generic 500. The Hub message embeds the
    # request URL, which must be scrubbed out of the client-facing 400.
    from huggingface_hub.errors import RepositoryNotFoundError

    def _boom(spec_id, weight, **kw):
        # response is optional in huggingface_hub 0.x but required in 1.x; both only
        # read .headers / .request, so a stub keeps the test working on either.
        raise RepositoryNotFoundError(
            "404 Client Error. Repository Not Found for url: "
            "https://huggingface.co/api/models/nope/nope (Request ID: abc)",
            response = types.SimpleNamespace(headers = {}, request = None),
        )

    monkeypatch.setattr(dl, "resolve_one", _boom)
    with pytest.raises(ValueError) as ei:
        dl.resolve_specs([("nope/nope", 1.0)])
    assert "http" not in str(ei.value)  # request URL scrubbed
    assert "Repository Not Found" in str(ei.value)


def test_scan_local_disambiguates_identical_stems(tmp_path, monkeypatch):
    # foo.safetensors and foo.gguf must get distinct ids so each is addressable; a
    # unique stem keeps its clean stem id.
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
    assert "solo" in by_id  # unique stem is untouched


def test_resolve_one_rejects_traversal_weight_name(tmp_path, monkeypatch):
    # A client-supplied weight file with traversal / absolute path is rejected before it
    # can reach the downloader (it must stay a plain filename inside the repo).
    monkeypatch.setattr(dl, "loras_dir", lambda: tmp_path)
    for bad in ("owner/name:../secret.safetensors", "owner/name:/etc/x.safetensors"):
        with pytest.raises(ValueError):
            dl.resolve_one(bad, 1.0)


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
    # duplicate ids are rejected: repeating an id would load the same adapter as several
    # distinct suffixed adapters and stack its effect past the per-adapter weight bound.
    with pytest.raises(Exception):
        DiffusionGenerateRequest(
            prompt = "x", loras = [{"id": "a", "weight": 0.5}, {"id": "a", "weight": 1.0}]
        )


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
    # int8/fp8 pipes bake adapters at load time; a bake-less quant pipe cannot take one at
    # generation time (frozen topology) and must direct the client to reload with the
    # selection. nvfp4/mxfp8 are never baked, so the same reload error is unreachable there
    # via the API (supports_lora blocks the load), but the backend path is shared.
    import threading
    pipe = _FakePipe()
    with pytest.raises(ValueError, match = "Reload the model with the adapter selection"):
        _backend()._apply_loras(
            _fake_state(pipe, kind = "single_file", quant = "fp8"),
            [("styleA", 1.0)],
            threading.Event(),
        )
    assert pipe.loaded == []  # rejected before touching the pipe


def test_diffusers_apply_rejects_gguf_adapter(monkeypatch):
    # A .gguf adapter (discoverable in the shared catalog) cannot load on the diffusers
    # engine; it must be rejected as a clean 400 before touching the pipe.
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
    assert pipe.loaded == []  # never touched the pipe


def test_scan_local_reads_family_sidecar(tmp_path, monkeypatch):
    import json

    d = tmp_path / "loras"
    d.mkdir()
    (d / "trained.safetensors").write_bytes(b"x")
    (d / "trained.json").write_text(
        json.dumps({"family": "sdxl", "base_model": "b", "weight_default": 0.8})
    )
    (d / "plain.safetensors").write_bytes(b"y")  # no sidecar -> unknown family
    monkeypatch.setattr(dl, "loras_dir", lambda: d)

    by_id = {e.id: e for e in dl.list_loras()}
    assert by_id["trained"].families == ("sdxl",)
    assert by_id["trained"].weight_default == 0.8
    assert by_id["plain"].families == ()
    assert by_id["plain"].weight_default == 1.0

    # Family filter: the sdxl-tagged adapter is kept for sdxl and hidden for flux.1;
    # the untagged one is always shown (unknown compatibility).
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
