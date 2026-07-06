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


def test_resolve_controlnet_rejects_filesystem_like_ids():
    # The bare-repo fallback must never accept a path-shaped id: from_pretrained
    # would treat it as a local directory, bypassing the controlnets_dir() contract.
    for bad in ("/tmp/model", "../some/model", "./x/y", "~/x/y", "a/b/c", "C:\\x/y", ".hidden/x"):
        with pytest.raises(FileNotFoundError):
            dc.resolve_controlnet(bad)


def test_resolve_controlnet_enforces_family_match():
    # A curated entry tagged for another family must be rejected before download so it
    # never reaches the wrong ControlNet pipeline class.
    with pytest.raises(ValueError, match = "not the"):
        dc.resolve_controlnet("qwen-union", family = "flux.1")
    # The matching family resolves fine, and no family (unfiltered) is permissive.
    assert dc.resolve_controlnet("qwen-union", family = "qwen-image").path
    assert dc.resolve_controlnet("qwen-union").path


def test_union_control_mode_maps_only_union_entries():
    # Union entries map a known control type to its integer mode; a union model always
    # needs a concrete mode, so an unmapped type (passthrough) defaults to 0. A non-union
    # id returns None so the caller omits control_mode.
    assert dc.union_control_mode("flux-union-pro", "canny") == 0
    assert dc.union_control_mode("flux-union-pro", "depth") == 2
    assert dc.union_control_mode("flux-union-pro", "pose") == 4
    assert dc.union_control_mode("flux-union-pro", "passthrough") == 0
    assert dc.union_control_mode("some/bare-repo", "canny") is None


def test_resolve_controlnet_local(tmp_path, monkeypatch):
    d = tmp_path / "controlnets"
    d.mkdir()
    cn = d / "my-cn"
    cn.mkdir()
    (cn / "config.json").write_text("{}")
    (cn / "diffusion_pytorch_model.safetensors").write_bytes(b"x")  # a loadable weight
    monkeypatch.setattr(dc, "controlnets_dir", lambda: d)
    entries = {e.id for e in dc.list_controlnets()}
    assert "my-cn" in entries
    r = dc.resolve_controlnet("my-cn")
    assert r.is_local and r.path == str(cn)


def test_scan_local_skips_config_only_folder(tmp_path, monkeypatch):
    # A folder with config.json but no weight/index (interrupted copy) must NOT be
    # advertised: it would otherwise fail deep in from_pretrained as a generic 500.
    d = tmp_path / "controlnets"
    d.mkdir()
    incomplete = d / "incomplete-cn"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")
    monkeypatch.setattr(dc, "controlnets_dir", lambda: d)
    assert "incomplete-cn" not in {e.id for e in dc.list_controlnets()}
    # A sharded weight index counts as a loadable weight.
    (incomplete / "diffusion_pytorch_model.safetensors.index.json").write_text("{}")
    assert "incomplete-cn" in {e.id for e in dc.list_controlnets()}


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
        engine = "sd_cpp",
        family = "flux.1",
        has_controlnet_pipeline = True,
        model_kind = "gguf",
        transformer_quant = None,
    )
    assert not dc.supports_controlnet(
        engine = "diffusers",
        family = "z-image",
        has_controlnet_pipeline = False,
        model_kind = "pipeline",
        transformer_quant = None,
    )


# ── Request-model validation ────────────────────────────────────────────────


def test_controlnet_spec_and_request_validation():
    from models.inference import ControlNetSpec, DiffusionGenerateRequest

    assert DiffusionGenerateRequest(prompt = "x").controlnet is None
    req = DiffusionGenerateRequest(
        prompt = "x",
        controlnet = {
            "id": "flux-union-pro",
            "image": "data",
            "control_type": "canny",
            "strength": 0.6,
        },
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
    def from_pretrained(
        cls,
        path,
        torch_dtype = None,
        token = None,
    ):
        m = cls()
        m.path = path
        return m

    def to(self, device):
        self.device = device
        return self


class _FakeCNPipe:
    @classmethod
    def from_pipe(
        cls,
        base,
        controlnet = None,
        torch_dtype = None,
    ):
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


def _allow_cn_security(monkeypatch):
    """Stub the Hub malware preflight to allow the load (hermetic, no network)."""
    import utils.security
    monkeypatch.setattr(
        utils.security,
        "evaluate_file_security",
        lambda name, hf_token = None, **kw: types.SimpleNamespace(blocked = False, reason = ""),
    )


def test_controlnet_pipe_loads_once_and_caches(monkeypatch):
    import threading

    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers())
    _allow_cn_security(monkeypatch)
    b = DiffusionBackend()
    st = _state()
    # The pipe cache only commits while ``st`` is the CURRENT load (an unload racing
    # from_pipe must not repopulate the cache), so mirror the loaded invariant.
    b._state = st
    resolved = dc.ResolvedControlNet("flux-union-pro", "repo/id", is_local = False)
    p1 = b._controlnet_pipe(st, resolved, threading.Event())
    assert isinstance(p1, _FakeCNPipe) and isinstance(p1.controlnet, _FakeCNModel)
    assert p1.controlnet.path == "repo/id" and p1.controlnet.device == "cpu"
    # cached: same id -> same model + same pipe, no reload.
    p2 = b._controlnet_pipe(st, resolved, threading.Event())
    assert p2 is p1
    assert b._cn_models["flux-union-pro"] is p1.controlnet


def test_controlnet_pipe_blocks_flagged_remote_repo(monkeypatch):
    # A bare owner/name ControlNet is accepted by resolve_controlnet without the base
    # trust gate, so the load path must run the Hub malware preflight: a flagged remote
    # repo must raise BEFORE from_pretrained downloads/deserializes it.
    import threading

    import utils.security
    from core.inference.diffusion import DiffusionBackend

    loaded = {"called": False}

    class _TrapModel(_FakeCNModel):
        @classmethod
        def from_pretrained(
            cls,
            path,
            torch_dtype = None,
            token = None,
        ):
            loaded["called"] = True
            return super().from_pretrained(path, torch_dtype = torch_dtype, token = token)

    mod = _fake_diffusers()
    mod.FluxControlNetModel = _TrapModel
    monkeypatch.setitem(sys.modules, "diffusers", mod)
    monkeypatch.setattr(
        utils.security,
        "evaluate_file_security",
        lambda name, hf_token = None, **kw: types.SimpleNamespace(
            blocked = True, reason = "Hugging Face security scan flagged unsafe files: evil.bin"
        ),
    )
    b = DiffusionBackend()
    st = _state()
    b._state = st
    resolved = dc.ResolvedControlNet("evil/cn", "evil/cn", is_local = False)
    with pytest.raises(ValueError, match = "security scan flagged"):
        b._controlnet_pipe(st, resolved, threading.Event())
    assert loaded["called"] is False


def test_controlnet_pipe_skips_scan_for_local_dir(monkeypatch, tmp_path):
    # A local dir the user picked has no Hub scan; the preflight must not block it even
    # if the (unused) scan stub would say blocked.
    import threading

    import utils.security
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers())
    monkeypatch.setattr(
        utils.security,
        "evaluate_file_security",
        lambda name, hf_token = None, **kw: types.SimpleNamespace(blocked = True, reason = "x"),
    )
    b = DiffusionBackend()
    st = _state()
    b._state = st
    resolved = dc.ResolvedControlNet("my-cn", str(tmp_path), is_local = True)
    p = b._controlnet_pipe(st, resolved, threading.Event())
    assert isinstance(p, _FakeCNPipe)


def test_controlnet_pipe_rejects_family_without_classes():
    import threading

    from core.inference.diffusion import DiffusionBackend

    b = DiffusionBackend()
    st = _state()
    st.family.controlnet_pipeline_class = None
    with pytest.raises(ValueError, match = "not supported"):
        b._controlnet_pipe(st, dc.ResolvedControlNet("x", "y", False), threading.Event())


def test_controlnet_pipe_not_cached_after_unload_race(monkeypatch):
    # An unload that lands while from_pipe is assembling must not let the wrapper
    # repopulate the cache around the torn-down base pipe.
    import threading

    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers())
    b = DiffusionBackend()
    st = _state()  # never committed to b._state: the load is already gone
    resolved = dc.ResolvedControlNet("flux-union-pro", "repo/id", is_local = False)
    with pytest.raises(RuntimeError, match = "cancelled"):
        b._controlnet_pipe(st, resolved, threading.Event())
    assert b._cn_pipes == {}
