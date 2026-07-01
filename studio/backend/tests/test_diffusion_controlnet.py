"""Tests for diffusion ControlNet support: discovery/resolve/preprocess/gate helpers, the
request-model validation, the family wiring, and the diffusers ControlNet pipe manager."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference import diffusion_controlnet as dc


# ── Pure helpers ────────────────────────────────────────────────────────────


def test_sanitize_id():
    assert dc.sanitize_id("owner/My ControlNet") == "My_ControlNet"
    assert dc.sanitize_id("weird:<>name") == "weird_name"
    assert dc.sanitize_id("") == "controlnet"


def test_list_controlnets_family_filter():
    flux = {e.id for e in dc.list_controlnets(family = "flux.1")}
    qwen = {e.id for e in dc.list_controlnets(family = "qwen-image")}
    assert "flux-union-pro" in flux and "qwen-union" not in flux
    assert "qwen-union" in qwen and "flux-union-pro" not in qwen


def test_resolve_controlnet_catalog_bare_repo_and_unknown():
    r = dc.resolve_controlnet("flux-union-pro", family = "flux.1")
    assert r.path == "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro" and not r.is_local
    # A bare public repo id passes through.
    r2 = dc.resolve_controlnet("owner/some-controlnet")
    assert r2.path == "owner/some-controlnet" and not r2.is_local
    with pytest.raises(FileNotFoundError):
        dc.resolve_controlnet("not-a-known-id")


def test_resolve_controlnet_local(tmp_path, monkeypatch):
    d = tmp_path / "controlnets"
    d.mkdir()
    cn = d / "my-cn"
    cn.mkdir()
    (cn / "config.json").write_text("{}")
    monkeypatch.setattr(dc, "controlnets_dir", lambda: d)
    entries = {e.id for e in dc.list_controlnets()}
    assert "my-cn" in entries
    r = dc.resolve_controlnet("my-cn")
    assert r.is_local and r.path == str(cn)


def test_preprocess_control_passthrough_and_canny():
    from PIL import Image

    img = Image.new("RGB", (32, 24), (10, 20, 30))
    # passthrough returns the same object.
    assert dc.preprocess_control(img, "passthrough") is img
    # a flat image has no edges -> canny falls back to passthrough (no black map).
    assert dc.preprocess_control(img, "canny") is img
    # an image with structure yields an edge map: RGB, same size, some white pixels.
    import numpy as np

    arr = np.zeros((24, 32, 3), np.uint8)
    arr[:, 16:, :] = 255  # a hard vertical edge
    edged = dc.preprocess_control(Image.fromarray(arr), "canny")
    assert edged.mode == "RGB" and edged.size == (32, 24)
    assert np.asarray(edged).max() == 255  # traced the edge


def test_supports_controlnet_matrix():
    ok = dict(engine = "diffusers", family = "flux.1", has_controlnet_pipeline = True)
    assert dc.supports_controlnet(**ok, model_kind = "pipeline", transformer_quant = None)
    assert dc.supports_controlnet(**ok, model_kind = "single_file", transformer_quant = None)
    # GGUF-via-diffusers and fp8/int8 dense are gated off, like LoRA.
    assert not dc.supports_controlnet(**ok, model_kind = "gguf", transformer_quant = None)
    assert not dc.supports_controlnet(**ok, model_kind = "single_file", transformer_quant = "fp8")
    assert not dc.supports_controlnet(**ok, model_kind = "single_file", transformer_quant = "int8")
    # native engine + a family without a CN pipeline are off.
    assert not dc.supports_controlnet(
        engine = "sd_cpp", family = "flux.1", has_controlnet_pipeline = True,
        model_kind = "gguf", transformer_quant = None,
    )
    assert not dc.supports_controlnet(
        engine = "diffusers", family = "z-image", has_controlnet_pipeline = False,
        model_kind = "pipeline", transformer_quant = None,
    )


# ── Request-model validation ────────────────────────────────────────────────


def test_controlnet_spec_and_request_validation():
    from models.inference import ControlNetSpec, DiffusionGenerateRequest

    assert DiffusionGenerateRequest(prompt = "x").controlnet is None
    req = DiffusionGenerateRequest(
        prompt = "x",
        controlnet = {"id": "flux-union-pro", "image": "data", "control_type": "canny", "strength": 0.6},
    )
    assert req.controlnet.id == "flux-union-pro" and req.controlnet.strength == 0.6
    # defaults
    s = ControlNetSpec(id = "a", image = "b")
    assert s.control_type == "passthrough" and s.strength == 1.0
    assert s.guidance_start == 0.0 and s.guidance_end == 1.0
    # bounds
    with pytest.raises(Exception):
        ControlNetSpec(id = "a", image = "b", strength = 3.0)
    with pytest.raises(Exception):
        ControlNetSpec(id = "a", image = "b", guidance_end = 1.5)


# ── Family wiring ───────────────────────────────────────────────────────────


def test_families_declare_controlnet_classes():
    from core.inference.diffusion_families import _FAMILIES

    by_name = {f.name: f for f in _FAMILIES}
    assert by_name["flux.1"].controlnet_pipeline_class == "FluxControlNetPipeline"
    assert by_name["flux.1"].controlnet_model_class == "FluxControlNetModel"
    assert by_name["qwen-image"].controlnet_pipeline_class == "QwenImageControlNetPipeline"
    # z-image has no diffusers ControlNet pipeline -> gated off.
    assert by_name["z-image"].controlnet_pipeline_class is None


# ── Diffusers ControlNet pipe manager ───────────────────────────────────────


class _FakeCNModel:
    @classmethod
    def from_pretrained(cls, path, torch_dtype = None, token = None):
        m = cls()
        m.path = path
        return m

    def to(self, device):
        self.device = device
        return self


class _FakeCNPipe:
    @classmethod
    def from_pipe(cls, base, controlnet = None, torch_dtype = None):
        p = cls()
        p.base = base
        p.controlnet = controlnet
        return p


def _fake_diffusers():
    mod = types.ModuleType("diffusers")
    mod.FluxControlNetModel = _FakeCNModel
    mod.FluxControlNetPipeline = _FakeCNPipe
    return mod


def _state():
    fam = types.SimpleNamespace(
        name = "flux.1",
        controlnet_pipeline_class = "FluxControlNetPipeline",
        controlnet_model_class = "FluxControlNetModel",
    )
    return types.SimpleNamespace(
        family = fam, dtype = "bf16", device = "cpu", hf_token = None, pipe = object()
    )


def test_controlnet_pipe_loads_once_and_caches(monkeypatch):
    import threading

    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers())
    b = DiffusionBackend()
    st = _state()
    resolved = dc.ResolvedControlNet("flux-union-pro", "repo/id", is_local = False)
    p1 = b._controlnet_pipe(st, resolved, threading.Event())
    assert isinstance(p1, _FakeCNPipe) and isinstance(p1.controlnet, _FakeCNModel)
    assert p1.controlnet.path == "repo/id" and p1.controlnet.device == "cpu"
    # cached: same id -> same model + same pipe, no reload.
    p2 = b._controlnet_pipe(st, resolved, threading.Event())
    assert p2 is p1
    assert b._cn_models["flux-union-pro"] is p1.controlnet


def test_controlnet_pipe_rejects_family_without_classes():
    import threading

    from core.inference.diffusion import DiffusionBackend

    b = DiffusionBackend()
    st = _state()
    st.family.controlnet_pipeline_class = None
    with pytest.raises(ValueError, match = "not supported"):
        b._controlnet_pipe(st, dc.ResolvedControlNet("x", "y", False), threading.Event())
