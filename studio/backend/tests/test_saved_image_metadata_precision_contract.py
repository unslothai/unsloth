# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract: a saved image's recipe records the build that ENGAGED, not the one requested.

``transformer_quant`` is a *request*: the loader may decline it (no dense source, not
enough VRAM, torchao missing) and run the GGUF as-is, or resolve "auto" to a concrete
scheme. Only ``_LoadState`` knows what actually ran, so the whole chain reads from it:

    load_pipeline(transformer_quant=...)  ->  _LoadState.transformer_quant  (ENGAGED)
                                                  |
                       diffusion.generate() returns state.kind / .gguf_filename / .transformer_quant
                                                  |
                        routes/inference.py persists result[...] into the PNG recipe
                                                  |
                                     images-page.tsx RecipePopover "Quant" row

If any hop starts echoing the request instead, a Recipe popover claims an image was made
with a quant that never loaded. These tests pin the divergent case at both ends.

They also pin ``image_gallery._REQUIRED_META``: the build keys are additive, so a PNG
written before they existed must still list rather than be dropped as foreign.

Hermetic: torch / diffusers are stubbed via ``sys.modules`` (same approach as
``test_diffusion_backend.py``, stubbed as packages so the loader's submodule imports resolve
without a real install), and the route half runs against a fake backend.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import io
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.diffusion as diffusion_module
import core.inference.gpu_arbiter as gpu_arbiter
import core.inference.image_gallery as gallery
from auth.authentication import get_current_subject
from core.inference.diffusion import DiffusionBackend
from routes.inference import router as openai_router, studio_router

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"

# The build fields the recipe carries beyond the plain generation settings. All sourced from
# the committed load state, all optional on an older PNG.
# Load-time build identity the route persists and GalleryImage defaults for older records.
# baked_loras belongs here for the same reason the other three do: promoting any of them into
# image_gallery._REQUIRED_META would stop every PNG written before it existed from listing.
_BUILD_KEYS = ("model_kind", "gguf_filename", "transformer_quant", "baked_loras")


# ── stub runtime (pared-down twin of test_diffusion_backend's) ────────────────


class _FakeDtype:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"torch.{self._name}"

    __str__ = __repr__


class _FakeGenerator:
    def __init__(self, device = None) -> None:
        self.device = device

    def seed(self) -> int:
        return 4242

    def manual_seed(self, value: int):
        return self


class _FakeImage:
    """Stand-in for a generated PIL image."""


class _FakePipe:
    def __init__(self) -> None:
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self, device = None) -> None:
        pass

    def enable_sequential_cpu_offload(self, device = None) -> None:
        pass

    def enable_vae_tiling(self) -> None:
        pass

    def enable_vae_slicing(self) -> None:
        pass

    # Explicit signature (not just **kwargs) so generate()'s signature-gated guards fire.
    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        **kwargs,
    ):
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakePipeline:
    @classmethod
    def from_pretrained(cls, base, **kwargs):
        return _FakePipe()


class _FakeTransformer:
    last: dict = {}

    @classmethod
    def from_single_file(cls, path, **kwargs):
        _FakeTransformer.last = {"path": path, **kwargs}
        return object()


def _stub_package(name: str) -> types.ModuleType:
    """A stub module the import machinery will treat as a PACKAGE.

    ``types.ModuleType`` alone has no ``__path__``, so ``import torch.nn.functional`` cannot
    resolve a submodule through it -- see ``stub_runtime`` for why that matters.
    """
    module = types.ModuleType(name)
    module.__path__ = []  # empty: submodules are registered by hand, never found on disk
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader = None, is_package = True)
    return module


@pytest.fixture
def stub_runtime(monkeypatch):
    """Enough torch / diffusers for a z-image GGUF load + one txt2img generate.

    ``load_pipeline`` lazily imports ``diffusion_eager_patches``, whose module body runs
    ``import torch.nn.functional as F``, and the GGUF prefix-strip shim imports
    ``diffusers.loaders.single_file_model``. Neither resolves through a bare ``ModuleType``.
    A dev box hides that -- something (``tests/conftest.py`` -> ``unsloth_zoo`` -> ``import
    torch``) has usually already seeded ``sys.modules["torch.nn.functional"]``, so the import
    short-circuits on the cached entry and never looks at the stub's missing ``__path__``. On a
    clean CPU-only CI interpreter nothing seeds it and the load dies with "'torch' is not a
    package". So register the submodules explicitly and make the stubs real packages: same
    hermetic runtime in both environments, whatever ran before.
    """
    torch = _stub_package("torch")
    torch.bfloat16 = _FakeDtype("bfloat16")
    torch.float16 = _FakeDtype("float16")
    torch.float32 = _FakeDtype("float32")
    torch.Generator = _FakeGenerator
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    torch.backends = types.SimpleNamespace(mps = None)
    torch.inference_mode = lambda: contextlib.nullcontext()
    # torch.nn.functional: imported by diffusion_eager_patches. Empty -- the patch installers
    # only probe it (hasattr F, "rms_norm") and no patched forward runs under the fake pipe.
    torch_nn = _stub_package("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn.functional = torch_nn_functional
    torch.nn = torch_nn

    diffusers = _stub_package("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)
    diffusers.ZImagePipeline = _FakePipeline
    diffusers.ZImageTransformer2DModel = _FakeTransformer
    # diffusers.loaders.single_file_model: the GGUF prefix-strip shim looks the transformer class
    # up in this registry. Empty -> the shim finds no entry and returns, the same no-op it
    # performs against a real diffusers that has no converter for the class.
    diffusers_loaders = _stub_package("diffusers.loaders")
    single_file_model = types.ModuleType("diffusers.loaders.single_file_model")
    single_file_model.SINGLE_FILE_LOADABLE_CLASSES = {}
    diffusers_loaders.single_file_model = single_file_model
    diffusers.loaders = diffusers_loaders

    for name, module in (
        ("torch", torch),
        ("torch.nn", torch_nn),
        ("torch.nn.functional", torch_nn_functional),
        ("diffusers", diffusers),
        ("diffusers.loaders", diffusers_loaders),
        ("diffusers.loaders.single_file_model", single_file_model),
    ):
        # setitem restores the previous entry (or deletes it, if there was none) on teardown,
        # so a real torch/diffusers imported by another test is left untouched.
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", lambda: None)
    _FakeTransformer.last = {}
    yield torch
    # A load that COMMITS deliberately keeps its process-wide eager/arch patches installed --
    # only unload() reverts them (diffusion.py's finally covers the pre-commit failure path
    # alone), and the GGUF default speed profile is "default", not "off", so the install runs.
    # Today nothing survives here because diffusers 0.39's bodies do not match the drift guards,
    # but that is a version accident, so revert unconditionally rather than rely on it. Both
    # calls are idempotent, and this runs before monkeypatch restores the stub modules.
    try:
        from core.inference.diffusion_arch_patches import uninstall_arch_patches
        from core.inference.diffusion_eager_patches import uninstall_patches

        uninstall_patches()
        uninstall_arch_patches()
    except Exception:  # noqa: BLE001 - teardown must not mask the test's own failure
        pass
    # ...and evict the patch modules themselves. They were imported (lazily, by load_pipeline)
    # WHILE the fakes were installed, so their module-level `torch`, `F` and diffusers class
    # globals are bound to the stubs. monkeypatch puts sys.modules["torch"] back but not these,
    # so every later test in the process -- and any real load -- would go on running against
    # module bodies that closed over the fakes. Dropping the cache entries makes the next import
    # rebind them under whatever runtime is installed then.
    for cached in (
        "core.inference.diffusion_eager_patches",
        "core.inference.diffusion_arch_patches",
    ):
        sys.modules.pop(cached, None)


@pytest.fixture
def backend(stub_runtime):
    """A backend that is unloaded afterwards, so a committed load's process-wide state does not
    outlive the test that created it."""
    instance = DiffusionBackend()
    try:
        yield instance
    finally:
        try:
            instance.unload()
        except Exception:  # noqa: BLE001 - a stub teardown failure is not this test's verdict
            pass


def _load(backend, tmp_path, monkeypatch, torch, **kwargs):
    (tmp_path / "m.gguf").write_bytes(b"x")
    # Drive the loader down the CUDA (dense-quant capable) path under the stub.
    monkeypatch.setattr(backend, "_pick_device_and_dtype", lambda: ("cuda", torch.bfloat16))
    return backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", **kwargs
    )


def _generate(backend):
    return backend.generate(prompt = "a sloth", width = 512, height = 512, steps = 2, guidance = 1.0, seed = 7)


# ── backend: generate() reports the committed load state ──────────────────────


def test_a_declined_quant_request_is_not_reported_as_engaged(
    backend, stub_runtime, tmp_path, monkeypatch
):
    """The user asked for fp8; the host has no dense source, so the GGUF loaded as-is.
    The recipe must say "GGUF, no quant", not "fp8"."""
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: False)
    status = _load(backend, tmp_path, monkeypatch, stub_runtime, transformer_quant = "fp8")

    assert status["transformer_quant"] is None
    assert backend._state.transformer_quant is None
    # The provenance record keeps BOTH sides, so the UI can explain the difference: the
    # request was explicit, the engaged value is "off".
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["value"] == "off" and resolved["source"] == "explicit"
    assert "GGUF transformer loaded" in resolved["reason"]

    result = _generate(backend)
    assert result["transformer_quant"] is None, (
        "generate() echoed the declined request; the saved recipe would claim a quant "
        "that never loaded"
    )
    assert result["model_kind"] == "gguf"
    assert result["gguf_filename"] == "m.gguf"


def test_generate_reports_the_engaged_scheme_when_it_differs_from_the_request(
    backend, stub_runtime, tmp_path, monkeypatch
):
    """The request said fp8; the resolver picked int8 for this GPU. int8 is what ran, so
    int8 is what the recipe has to record."""
    requested, engaged = "fp8", "int8"

    @classmethod
    def _from_pretrained(cls, base, **kwargs):
        return object()

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _from_pretrained, raising = False)
    monkeypatch.setattr(diffusion_module, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        diffusion_module,
        "select_transformer_quant_scheme",
        lambda target, mode, family = None: engaged,
    )
    monkeypatch.setattr(diffusion_module, "resolve_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(
        diffusion_module, "quantize_transformer", lambda pipe, target, *, mode, **kw: engaged
    )

    status = _load(backend, tmp_path, monkeypatch, stub_runtime, transformer_quant = requested)

    assert status["transformer_quant"] == engaged
    assert backend._state.transformer_quant == engaged

    result = _generate(backend)
    assert result["transformer_quant"] == engaged, (
        f"generate() reported {result['transformer_quant']!r}; the ENGAGED scheme was "
        f"{engaged!r} and the request was {requested!r}"
    )
    assert result["transformer_quant"] != requested
    # The dense build is no longer a GGUF transformer, and that is part of the build identity too.
    assert result["model_kind"] == "gguf" and result["gguf_filename"] == "m.gguf"


# ── route: the persisted recipe carries what generate() reported ──────────────


class _EngagedBackend:
    """A backend that accepts one precision and engages another, so the route cannot
    satisfy the assertions by reading the load request."""

    requested = "fp8"
    engaged = "int8"

    def __init__(self) -> None:
        self.loaded = False
        self.loading: tuple = ()
        self.last_load_kwargs: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def loading_repo_ids(self) -> tuple:
        return tuple(self.loading)

    def validate_load_request(self, model_path, **kwargs):
        from core.inference.diffusion_families import detect_family
        return detect_family(model_path, kwargs.get("family_override"))

    def preflight_base_access(self, model_path, fam, **kwargs):
        return None

    def begin_load(self, model_path, **kwargs):
        self.loaded = True
        self.last_load_kwargs = dict(kwargs)
        return {
            "loaded": True,
            "repo_id": model_path,
            "family": "z-image",
            "base_repo": "base/repo",
            "device": "cuda",
            "dtype": "bfloat16",
            "cpu_offload": False,
            "offload_policy": "none",
            "vae_tiling": False,
            "memory_mode": "auto",
            # The loader declined fp8 and engaged int8 instead.
            "transformer_quant": self.engaged,
        }

    def load_progress(self):
        return {
            "phase": "ready" if self.loaded else None,
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "fraction": 1.0,
            "error": None,
        }

    def generate(
        self,
        *,
        seed = None,
        batch_size = 1,
        prompts = None,
        seeds = None,
        **kwargs,
    ):
        if not self.loaded:
            raise RuntimeError("No diffusion model is loaded.")
        return {
            "images": [object() for _ in range(batch_size)],
            "seed": seed if seed is not None else 4242,
            "repo_id": "x/z-image",
            # Straight off the committed load state, as the real backend does.
            "model_kind": "gguf",
            "gguf_filename": "z-image-Q4_K_M.gguf",
            "transformer_quant": self.engaged,
            "workflow": "txt2img",
        }

    def generate_progress(self):
        return {"active": False, "step": 0, "total_steps": 0, "fraction": 0.0, "eta_seconds": None}

    def unload(self):
        self.loaded = False
        return {"loaded": False}

    def status(self):
        return {"loaded": self.loaded, "repo_id": None, "family": None, "cpu_offload": False}


@pytest.fixture
def engaged_client(monkeypatch, tmp_path):
    backend = _EngagedBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    import core.inference.diffusion_engine_router as engine_router

    monkeypatch.setattr(engine_router, "select_and_activate_engine", lambda fam, **kw: backend)
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(engine_router, "predict_engine", lambda fam, **kw: "diffusers")
    monkeypatch.setattr(engine_router, "_active_engine_name", "diffusers")
    monkeypatch.setattr(engine_router, "_fallback_reason", None)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: None)

    # Record exactly the metadata dict the route hands the gallery.
    saved: list[dict] = []

    def _save(image, meta):
        saved.append(meta)
        image_id = f"img{len(saved)}"
        (tmp_path / f"{image_id}.png").write_bytes(b"PNG")
        return {**meta, "id": image_id, "url": f"/api/inference/images/gallery/{image_id}/file"}

    monkeypatch.setattr(gallery, "save", _save)

    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    # The OpenAI-compatible images route lives on the other router, mounted at /v1 in
    # production. Both persistence paths reach the same gallery, so both are exercised here.
    app.include_router(openai_router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), backend, saved


def test_the_persisted_recipe_records_the_engaged_build_not_the_load_request(engaged_client):
    client, backend, saved = engaged_client
    load = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-Q4_K_M.gguf",
            "transformer_quant": _EngagedBackend.requested,
        },
    )
    assert load.status_code == 200, load.text
    # The request really did ask for the other scheme.
    assert backend.last_load_kwargs["transformer_quant"] == _EngagedBackend.requested

    gen = client.post("/api/inference/images/generate", json = {"prompt": "a sloth", "seed": 7})
    assert gen.status_code == 200, gen.text

    assert len(saved) == 1
    meta = saved[0]
    assert meta["transformer_quant"] == _EngagedBackend.engaged, (
        f"the recipe recorded {meta['transformer_quant']!r}; the load REQUESTED "
        f"{_EngagedBackend.requested!r} and the backend ENGAGED {_EngagedBackend.engaged!r}"
    )
    assert meta["transformer_quant"] != _EngagedBackend.requested
    assert meta["model_kind"] == "gguf"
    assert meta["gguf_filename"] == "z-image-Q4_K_M.gguf"

    # ...and again through the response model, because that is what the Recipe popover reads.
    # The dict above is pre-serialization: a GalleryImage that stops declaring these fields has
    # FastAPI silently strip them from the wire while every assertion above still passes.
    body = gen.json()["images"][0]
    for field, expected in (
        ("transformer_quant", _EngagedBackend.engaged),
        ("model_kind", "gguf"),
        ("gguf_filename", "z-image-Q4_K_M.gguf"),
    ):
        assert body.get(field) == expected, (
            f"the generate response dropped {field!r}: the route recorded {meta[field]!r} and "
            f"the serialized body says {body.get(field)!r}"
        )


def test_the_openai_route_persists_the_same_build(engaged_client):
    """The other supported way to make an image.

    /v1/images/generations goes through the same backend and the same gallery, and its recipe
    was missing every build key -- so an image made through an OpenAI client listed with no
    quant, no kind and no filename while the contract above stayed green.
    """
    client, backend, saved = engaged_client
    load = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Z-Image-Turbo-GGUF",
            "gguf_filename": "z-image-Q4_K_M.gguf",
            "transformer_quant": _EngagedBackend.requested,
        },
    )
    assert load.status_code == 200, load.text

    generated = client.post(
        "/v1/images/generations",
        json = {"prompt": "a sloth", "n": 1, "response_format": "url"},
    )
    assert generated.status_code == 200, generated.text

    assert len(saved) == 1
    meta = saved[0]
    assert meta["transformer_quant"] == _EngagedBackend.engaged
    assert meta["model_kind"] == "gguf"
    assert meta["gguf_filename"] == "z-image-Q4_K_M.gguf"
    assert meta["baked_loras"] == []


def test_the_stub_runtime_does_not_outlive_its_own_test(stub_runtime):
    """The patch modules are imported lazily by load_pipeline, i.e. WHILE the fakes are
    installed, so their module globals close over them. monkeypatch restores sys.modules["torch"]
    but not those, and every later test in the process would then run against module bodies
    bound to a fake torch. The fixture has to evict them."""
    import core.inference.diffusion_eager_patches  # noqa: F401 — imported under the stubs
    assert sys.modules["torch"] is stub_runtime
    # The eviction itself is asserted by the sibling test below, which runs after teardown.


def test_the_patch_modules_are_not_left_cached_against_the_fakes():
    """Runs outside the stub fixture: whatever the test above imported must be gone."""
    for cached in (
        "core.inference.diffusion_eager_patches",
        "core.inference.diffusion_arch_patches",
    ):
        module = sys.modules.get(cached)
        if module is None:
            continue
        # Present only because something imported it under the REAL runtime.
        torch_global = getattr(module, "torch", None)
        assert torch_global is None or torch_global is sys.modules.get(
            "torch"
        ), f"{cached} is cached with a torch that is not the live one"


# ── gallery: the build keys stay additive ─────────────────────────────────────


@pytest.fixture
def tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)
    return tmp_path


def _old_schema_meta() -> dict:
    """A recipe as written before the build fields existed."""
    return {
        "prompt": "a sloth",
        "negative_prompt": None,
        "width": 1024,
        "height": 1024,
        "steps": 9,
        "guidance": 0.0,
        "seed": 7,
        "model": "unsloth/Z-Image-Turbo-GGUF",
        "created_at": 100.0,
    }


def test_the_build_keys_are_never_required_to_list_a_png(tmp_gallery):
    """``_REQUIRED_META`` is the "is this PNG ours" gate. Promoting a build key into it
    would silently hide every image generated before that key existed."""
    for key in _BUILD_KEYS:
        assert key not in gallery._REQUIRED_META, (
            f"{key!r} became a required recipe key; every PNG written before it existed would "
            "stop listing"
        )


def test_a_png_without_the_build_keys_still_lists(tmp_gallery):
    pytest.importorskip("PIL")
    from PIL import Image

    meta = _old_schema_meta()
    for key in _BUILD_KEYS:
        assert key not in meta
    record = gallery.save(Image.new("RGB", (16, 16), (10, 20, 30)), meta)

    listed = gallery.list_images()
    assert [r["id"] for r in listed] == [record["id"]]
    assert listed[0]["prompt"] == "a sloth"
    # Absent, not null-filled: the popover keys off truthiness and hides the rows.
    for key in _BUILD_KEYS:
        assert key not in listed[0]
    assert gallery.owned_image_path(record["id"]) is not None


def test_a_png_with_the_build_keys_round_trips_them(tmp_gallery):
    pytest.importorskip("PIL")
    from PIL import Image

    meta = {
        **_old_schema_meta(),
        "model_kind": "gguf",
        "gguf_filename": "z-image-Q4_K_M.gguf",
        "transformer_quant": "int8",
    }
    record = gallery.save(Image.new("RGB", (16, 16), (10, 20, 30)), meta)
    listed = gallery.list_images()
    assert listed[0]["transformer_quant"] == "int8"

    # Through the listing's response model too. Pydantic drops anything the model does not
    # declare, so a GalleryImage that stops carrying a build key leaves the raw assertion above
    # green and the wire silently short -- which is the popover going blank.
    from models.inference import GalleryListResponse

    wire = GalleryListResponse(images = listed).model_dump()["images"][0]
    for key in ("model_kind", "gguf_filename", "transformer_quant"):
        assert wire.get(key) == meta[key], (
            f"GalleryImage no longer serializes {key!r}: the record has {meta[key]!r}, the wire "
            f"has {wire.get(key)!r}"
        )

    # The PNG itself carries the recipe, so a downloaded file keeps the build identity.
    raw = (gallery.gallery_dir() / f"{record['id']}.png").read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        embedded = json.loads(im.text["unsloth"])
    assert embedded["transformer_quant"] == "int8"
    assert embedded["gguf_filename"] == "z-image-Q4_K_M.gguf"


# ── frontend: the recipe popover still shows the engaged build ────────────────


def test_the_recipe_popover_renders_the_build_fields():
    src = (_FRONTEND / "features" / "images" / "images-page.tsx").read_text(encoding = "utf-8")
    popover = src[src.index("function RecipePopover(") :]
    popover = popover[: popover.index("\ntype Busy")]
    assert '<RecipeRow label="Quant" value={image.transformer_quant} />' in popover
    assert '<RecipeRow label="File" value={image.gguf_filename} mono />' in popover
    # Rendered conditionally, so an older PNG without them shows the rest of the recipe.
    for key in ("transformer_quant", "gguf_filename"):
        assert f"image.{key} ?" in popover
