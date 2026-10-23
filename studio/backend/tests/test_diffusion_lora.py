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


def test_materialize_native_dir_real_files_hardlinks_inside_dest(tmp_path):
    # sd-server rejects a staged LoRA whose canonical path leaves --lora-model-dir, so server staging needs a real file.
    src = tmp_path / "cache" / "a.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"x")
    dest = tmp_path / "managed"
    out = dl.materialize_native_dir(
        [dl.ResolvedLora("a", "a", str(src), "safetensors", 1.0)], dest, real_files = True
    )
    staged = Path(out[0].path)
    assert not staged.is_symlink() and staged.is_file()
    assert Path(os.path.realpath(staged)).parent == Path(os.path.realpath(dest))
    assert os.path.samefile(staged, src)  # a hard link, not a copy
    staged.unlink()
    assert src.read_bytes() == b"x"


def test_materialize_native_dir_real_files_copies_when_hardlink_fails(tmp_path, monkeypatch):
    src = tmp_path / "a.safetensors"
    src.write_bytes(b"x")

    def _no_link(*a, **k):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(dl.os, "link", _no_link)
    out = dl.materialize_native_dir(
        [dl.ResolvedLora("a", "a", str(src), "safetensors", 1.0)],
        tmp_path / "managed",
        real_files = True,
    )
    staged = Path(out[0].path)
    assert not staged.is_symlink() and staged.read_bytes() == b"x"
    assert not os.path.samefile(staged, src)


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


def test_baked_quant_pipe_disables_any_subset_of_its_adapters(monkeypatch):
    # Turning one of two baked adapters off arrives as the other alone: a weight change, not a new selection.
    # Every alias collides here, so a subset resolves to other names than the bake: adapters match by file.
    import threading

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, "style", f"/{i}.safetensors", "safetensors", w) for i, w in specs
        ],
    )
    b = _backend()
    pipe = _FakePipe()
    state = _fake_state(pipe, quant = "int8")
    pipe._unsloth_loras = b._resolve_lora_set(
        [("styleA", 1.0), ("styleB", 1.0)], family = "flux.1", hf_token = None
    )
    pipe._unsloth_loras_baked = True
    assert [n for (n, _p, _w) in pipe._unsloth_loras] == ["style", "style_2"]

    b._apply_loras(state, [("styleA", 1.0), ("styleB", 0.0)], threading.Event())
    assert pipe.active == (["style", "style_2"], [1.0, 0.0])
    b._apply_loras(state, [("styleB", 0.5)], threading.Event())
    assert pipe.active == (["style", "style_2"], [0.0, 0.5])
    b._apply_loras(state, [("styleB", 1.0), ("styleA", 0.7)], threading.Event())
    assert pipe.active == (["style", "style_2"], [0.7, 1.0])
    assert pipe.loaded == []  # never reloaded or re-attached
    # A file the bake does not hold still needs a reload with the new selection.
    with pytest.raises(ValueError, match = "Reload the model with the new adapter selection"):
        b._apply_loras(state, [("styleA", 1.0), ("styleC", 1.0)], threading.Event())
    assert pipe.active == (["style", "style_2"], [0.7, 1.0])


@pytest.mark.parametrize("mode, refused", [("group", True), ("resident", False), ("model", False)])
def test_baked_weights_stay_fixed_under_a_block_streamed_graph(monkeypatch, mode, refused):
    # Resetting a block-streamed graph for new LoRA scales left every later generate failing. Refuse the change.
    import threading

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w)
            for i, w in specs
        ],
    )
    b = _backend()
    pipe = _FakePipe()
    state = _fake_state(pipe, quant = "int8")
    state.cuda_graphs = (
        types.SimpleNamespace(placement = types.SimpleNamespace(mode = mode), reset = lambda: None),
    )
    pipe._unsloth_loras = b._resolve_lora_set(
        [("styleA", 1.0), ("styleB", 1.0)], family = "flux.1", hf_token = None
    )
    pipe._unsloth_loras_baked = True
    baked = pipe._unsloth_loras
    # The baked weights themselves never touch the graph.
    b._apply_loras(state, [("styleA", 1.0), ("styleB", 1.0)], threading.Event())
    for loras in ([("styleA", 1.0)], [("styleA", 0.5), ("styleB", 1.0)], []):
        if refused:
            with pytest.raises(ValueError, match = "Reload the model with the new adapter weights"):
                b._apply_loras(state, loras, threading.Event())
            assert pipe._unsloth_loras == baked and pipe.active is None
        else:
            b._apply_loras(state, loras, threading.Event())
    if not refused:
        assert pipe.active == (["styleA", "styleB"], [0.0, 0.0])


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


def test_moss_image_has_no_lora_loader_so_lora_is_gated_off(monkeypatch):
    import threading

    pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_moss_image import MOSSImagePipeline

    assert not hasattr(MOSSImagePipeline, "load_lora_weights")
    for quant in (None, "int8", "fp8"):
        assert not dl.supports_lora(
            engine = "diffusers", family = "moss-image", model_kind = "pipeline", transformer_quant = quant
        )
    assert dl.supports_lora(
        engine = "diffusers", family = "flux.1", model_kind = "pipeline", transformer_quant = None
    )
    fam = detect_family("OpenMOSS-Team/MOSS-Image-SFT")
    assert fam is not None and fam.name == "moss-image"

    backend = _backend()
    monkeypatch.setattr(backend, "validate_load_request", lambda *a, **k: fam)
    # Refused before anything loads; a weight-0 selection is not active and is not refused here.
    for load in (backend.begin_load, backend.load_pipeline):
        with pytest.raises(ValueError, match = "no LoRA loader"):
            load("OpenMOSS-Team/MOSS-Image-SFT", loras = [("styleA", 1.0)])
    # A generation-time selection on a loaded MOSS pipe is refused the same way, for every quant.
    for quant in (None, "int8"):
        state = types.SimpleNamespace(
            pipe = _FakePipe(), family = fam, kind = "pipeline", transformer_quant = quant, hf_token = None
        )
        with pytest.raises(ValueError, match = "no LoRA loader"):
            backend._apply_loras(state, [("styleA", 1.0)], threading.Event())
        backend._apply_loras(state, [("styleA", 0.0)], threading.Event())


def _peft_add_adapter(layer, name, config):
    # PEFT >= 0.19 takes ``config``; 0.18 takes every option positionally (``use_rslora`` required). Pass by keyword
    # what the installed signature names, read off ``config``, so the helper holds across PEFT 0.18.0 -> 0.20.0.
    import inspect

    known = {"r": config.r, "lora_alpha": config.lora_alpha, "config": config}
    kwargs = {}
    for param in list(inspect.signature(layer.update_layer).parameters.values())[1:]:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.name in known:
            kwargs[param.name] = known[param.name]
        elif hasattr(config, param.name):
            kwargs[param.name] = getattr(config, param.name)
    layer.update_layer(name, **kwargs)


def _peft_two_adapter_pipe():
    # A real PEFT LoRA Linear carrying two baked adapters, behind a pipe whose set_adapters does what diffusers does.
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear

    torch.manual_seed(0)
    config = LoraConfig(r = 2, lora_alpha = 4, lora_dropout = 0.0, init_lora_weights = False)
    layer = Linear(
        torch.nn.Linear(8, 8, bias = False),
        "a",
        config = config,
        r = 2,
        lora_alpha = 4,
        init_lora_weights = False,
    )
    _peft_add_adapter(layer, "b", config)
    layer.set_adapter(["a", "b"])
    layer.eval()

    class _Pipe:
        transformer = torch.nn.Sequential(layer)

        def set_adapters(
            self,
            names,
            adapter_weights = None,
        ):
            layer.set_adapter(list(names))
            for name, weight in zip(names, adapter_weights):
                layer.set_scale(name, weight)

    return _Pipe(), layer


def test_baked_lora_subset_and_weight_changes_keep_compiled_graphs(monkeypatch):
    # A baked int8/fp8 load keeps its adapters, so a subset / weight change only moves scales. PEFT's float scale is
    # a Dynamo guard (self.scaling['b'] == 1.0) and a CUDA-graph kernel argument: each change recompiled and dropped
    # every captured graph. Pinned as device tensors, the first prompt's graph serves every selection.
    import threading

    import torch
    from torch._dynamo.testing import CompileCounter

    from core.inference import diffusion as diff

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w)
            for i, w in specs
        ],
    )
    resets = []
    monkeypatch.setattr(diff.cuda_graph, "reset_all", lambda handles: resets.append(handles))
    b = _backend()
    pipe, layer = _peft_two_adapter_pipe()
    state = _fake_state(pipe, quant = "int8")
    state.cuda_graphs = ()
    pipe._unsloth_loras = b._resolve_lora_set(
        [("a", 1.0), ("b", 1.0)], family = "flux.1", hf_token = None
    )
    pipe._unsloth_loras_baked = True
    # What the load does once the denoiser is placed (absent before the fix: the floats stay).
    getattr(diff, "_pin_baked_lora_scales", lambda *_: 0)(pipe, "cpu")

    x = torch.randn(3, 8)

    def expected(wa, wb):
        with torch.no_grad():
            out = layer.base_layer(x)
            for name, w in (("a", wa), ("b", wb)):
                out = out + layer.lora_B[name](layer.lora_A[name](x)) * (w * 4 / 2)
            return out

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(pipe.transformer, backend = counter)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), expected(1.0, 1.0))
        compiled(x)
    assert counter.frame_count == 1

    for loras, wa, wb in (
        ([("a", 1.0)], 1.0, 0.0),  # subset: b omitted
        ([("a", 1.0), ("b", 1.0)], 1.0, 1.0),  # back
        ([("b", 0.5)], 0.0, 0.5),  # the other subset, new weight
        ([("a", 0.7), ("b", 1.0)], 0.7, 1.0),  # weight tweak
        ([], 0.0, 0.0),  # every adapter off
        ([("a", 1.0), ("b", 1.0)], 1.0, 1.0),
    ):
        b._apply_loras(state, loras, threading.Event())
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), expected(wa, wb))
        assert [w for (_n, _p, w) in pipe._unsloth_loras] == [wa, wb]
    assert counter.frame_count == 1  # no recompile
    assert resets == []  # no captured graph dropped
    torch._dynamo.reset()


def test_pinned_lora_scale_falls_back_when_a_scale_is_rewritten(monkeypatch):
    # Something rewriting a pinned scale as a float (a direct set_adapters) drops the pin: the next change takes the
    # set_adapters + graph reset path again instead of writing into a tensor the forward no longer reads.
    import threading

    from core.inference import diffusion as diff

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w)
            for i, w in specs
        ],
    )
    resets = []
    monkeypatch.setattr(diff.cuda_graph, "reset_all", lambda handles: resets.append(handles))
    b = _backend()
    pipe, layer = _peft_two_adapter_pipe()
    state = _fake_state(pipe, quant = "int8")
    state.cuda_graphs = ()
    pipe._unsloth_loras = b._resolve_lora_set(
        [("a", 1.0), ("b", 1.0)], family = "flux.1", hf_token = None
    )
    pipe._unsloth_loras_baked = True
    assert diff._pin_baked_lora_scales(pipe, "cpu") == 2
    # Idempotent: a second call pins nothing.
    assert diff._pin_baked_lora_scales(pipe, "cpu") == 0
    layer.set_scale("b", 1.0)
    b._apply_loras(state, [("a", 1.0)], threading.Event())
    assert layer.scaling["b"] == 0.0
    assert len(resets) == 1


def test_pin_declines_weights_off_the_target_device():
    from core.inference import diffusion as diff

    pipe, layer = _peft_two_adapter_pipe()
    pipe._unsloth_loras = (("a", "/a", 1.0), ("b", "/b", 1.0))
    pipe._unsloth_loras_baked = True
    assert diff._pin_baked_lora_scales(pipe, "cuda") == 0
    assert isinstance(layer.scaling["a"], float)


def test_replica_fingerprint_sees_an_in_place_scale_write():
    from core.inference import diffusion as diff
    from core.inference.diffusion_multigpu import weights_fingerprint

    pipe, _layer = _peft_two_adapter_pipe()
    pipe._unsloth_loras = (("a", "/a", 1.0), ("b", "/b", 1.0))
    pipe._unsloth_loras_baked = True
    assert diff._pin_baked_lora_scales(pipe, "cpu") == 2
    before = weights_fingerprint(pipe.transformer)
    assert weights_fingerprint(pipe.transformer) == before
    assert diff._set_pinned_lora_scales(pipe, {"a": 1.0})
    assert weights_fingerprint(pipe.transformer) != before


def test_pinned_lora_change_also_sets_the_text_encoder_scales(monkeypatch):
    # A LoRA file can carry text-encoder weights too. The pinned path skips set_adapters, so it has to set the text
    # encoder's scales itself, or prompts encode with the bake-time weight while the marker names the new one.
    import threading

    import torch
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear

    from core.inference import diffusion as diff

    monkeypatch.setattr(
        dl,
        "resolve_specs",
        lambda specs, **_: [
            dl.ResolvedLora(i, dl.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w)
            for i, w in specs
        ],
    )
    monkeypatch.setattr(diff.cuda_graph, "reset_all", lambda handles: None)
    pipe, _layer = _peft_two_adapter_pipe()
    config = LoraConfig(r = 2, lora_alpha = 4, lora_dropout = 0.0, init_lora_weights = False)
    te = Linear(
        torch.nn.Linear(8, 8, bias = False),
        "a",
        config = config,
        r = 2,
        lora_alpha = 4,
        init_lora_weights = False,
    )
    _peft_add_adapter(te, "b", config)
    pipe.components = {"transformer": pipe.transformer, "text_encoder": torch.nn.Sequential(te)}
    b = _backend()
    state = _fake_state(pipe, quant = "int8")
    state.cuda_graphs = ()
    pipe._unsloth_loras = b._resolve_lora_set(
        [("a", 1.0), ("b", 1.0)], family = "flux.1", hf_token = None
    )
    pipe._unsloth_loras_baked = True
    assert diff._pin_baked_lora_scales(pipe, "cpu") == 2
    b._apply_loras(state, [("a", 0.5)], threading.Event())
    assert te.scaling == {"a": 1.0, "b": 0.0}  # 0.5 * alpha / r, and the omitted adapter off


def _peft_embedding(name = "a"):
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.layer import Embedding

    config = LoraConfig(r = 2, lora_alpha = 4, lora_dropout = 0.0, init_lora_weights = False)
    return Embedding(
        torch.nn.Embedding(10, 8),
        name,
        config = config,
        r = 2,
        lora_alpha = 4,
        init_lora_weights = False,
    )


def test_pin_declines_a_denoiser_lora_it_cannot_pin(monkeypatch):
    # A PEFT Embedding keeps its factors in lora_embedding_A/B, not lora_B. Pinning only the Linear would make a
    # weight change skip set_adapters and leave the embedding at its bake-time scale: decline, keep floats.
    import torch

    from core.inference import diffusion as diff

    pipe, layer = _peft_two_adapter_pipe()
    emb = _peft_embedding()
    pipe.transformer = torch.nn.Sequential(layer, emb)
    pipe._unsloth_loras_baked = True
    assert diff._pin_baked_lora_scales(pipe, "cpu") == 0
    assert not getattr(pipe, diff._PINNED_LORA_SCALES_ATTR, None)
    assert all(not isinstance(v, torch.Tensor) for v in layer.scaling.values())
    assert diff._set_pinned_lora_scales(pipe, {"a": 0.5}) is False


def test_pinned_lora_change_also_sets_a_text_encoder_embedding_scale(monkeypatch):
    # The side layers include a text encoder's PEFT Embedding (no lora_B entry), so its scale follows the change.
    import torch

    from core.inference import diffusion as diff

    pipe, _layer = _peft_two_adapter_pipe()
    emb = _peft_embedding()
    pipe.components = {"transformer": pipe.transformer, "text_encoder": torch.nn.Sequential(emb)}
    pipe._unsloth_loras_baked = True
    assert diff._pin_baked_lora_scales(pipe, "cpu") == 2
    assert diff._set_pinned_lora_scales(pipe, {"a": 0.5}) is True
    assert emb.scaling == {"a": 1.0}  # 0.5 * alpha / r
