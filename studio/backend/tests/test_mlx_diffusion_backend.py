# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Engine-surface tests for the MLX image engine with mlx / mflux stubbed out."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from core.inference import mlx_diffusion_backend as mb
from core.inference.diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    _FAMILIES,
    DiffusionModelReplacedError,
    load_identity,
)

ZIMAGE = next(f for f in _FAMILIES if f.name == "z-image")

REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF = "z-image-turbo-Q4_0.gguf"
_REAL_RESOLVE_VARIANT_BASE = getattr(mb, "resolve_variant_base", None)
BASE = "Tongyi-MAI/Z-Image-Turbo"


class FakeRuntime:
    def __init__(self):
        self.builds = []
        self.generates = []
        self.freed = 0
        self.step_gate: threading.Event | None = None
        self.in_step = threading.Event()

    def build(self, model_path, *, model_config, precision, te_bits):
        self.builds.append(
            dict(
                model_path = model_path,
                model_config = model_config,
                precision = precision,
                te_bits = te_bits,
            )
        )
        return {"model": model_config}

    def cache_text_encodings(self, model):
        model["cached"] = True

    def supports_guidance(self, model):
        return model["model"] == "z-image"

    def generate(
        self, model, *, prompt, negative_prompt, seed, steps, width, height, guidance, on_step
    ):
        from PIL import Image

        self.generates.append(
            dict(
                prompt = prompt, seed = seed, steps = steps, width = width, height = height, guidance = guidance
            )
        )
        for t in range(steps):
            if self.step_gate is not None:
                self.in_step.set()
                self.step_gate.wait(5)
            on_step(t + 1)
        return Image.new("RGB", (width, height), (seed % 255, 0, 0))

    def free(self):
        self.freed += 1


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A backend whose fetch / convert / runtime are fakes; returns (backend, runtime, calls)."""
    gguf_path = tmp_path / "hub" / "snapshots" / "abc123" / GGUF
    gguf_path.parent.mkdir(parents = True)
    gguf_path.write_bytes(b"GGUF" + b"\0" * 64)
    base_dir = tmp_path / "base"
    for sub in ("text_encoder", "tokenizer", "vae"):
        (base_dir / sub).mkdir(parents = True)
    calls = {"convert": 0, "fetch_gguf": [], "fetch_base": []}

    def _fetch_gguf(repo_id, gguf_filename, hf_token, *, cancel_event, local_files_only):
        calls["fetch_gguf"].append((repo_id, gguf_filename, local_files_only))
        return str(gguf_path)

    def _fetch_base(
        base_repo,
        hf_token,
        *,
        local_files_only,
        cancel_event = None,
    ):
        calls["fetch_base"].append((base_repo, local_files_only))
        return str(base_dir)

    def _convert(
        gguf,
        out_dir,
        *,
        dense_bits,
        cancel_event = None,
        progress = None,
        **_,
    ):
        calls["convert"] += 1
        calls["dense_bits"] = dense_bits
        out = Path(out_dir) / "transformer"
        out.mkdir(parents = True, exist_ok = True)
        (out / "0.safetensors").write_bytes(b"x" * 8)
        (out / "unsloth_mlx_convert.json").write_text(
            json.dumps({"source": {"size": Path(gguf).stat().st_size}, "bytes": 8})
        )
        if progress:
            progress(1, 1)
        return {"n_out_tensors": 1}

    monkeypatch.setattr(mb, "fetch_gguf", _fetch_gguf)
    monkeypatch.setattr(mb, "fetch_base_snapshot", _fetch_base)
    # No card lookup in unit tests: the family default stands unless a test says otherwise.
    monkeypatch.setattr(
        mb, "resolve_variant_base", lambda *a, fallback, **k: fallback, raising = False
    )
    monkeypatch.setattr(mb.convert, "convert_gguf_to_mflux", _convert)
    monkeypatch.setattr(
        mb.convert,
        "conversion_complete",
        lambda d, **_k: (Path(d) / "transformer" / "0.safetensors").exists(),
    )
    monkeypatch.setattr(
        mb.convert,
        "cache_dir_for",
        lambda repo, rev, fn, *, dense_bits, **_: tmp_path / "cache" / f"{rev}-{fn}-d{dense_bits}",
    )
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX_PRECISION", "fp16")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX_TE_BITS", "4")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX_DENSE_BITS", "4")
    runtime = FakeRuntime()
    backend = mb.MlxDiffusionBackend(runtime_factory = lambda: runtime)
    return backend, runtime, calls


def _wait_loaded(backend, timeout = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        progress = backend.load_progress()
        if progress["phase"] in ("ready", "error"):
            return progress
        time.sleep(0.01)
    raise AssertionError(f"load did not finish: {backend.load_progress()}")


def _load(backend, **kw):
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf", **kw)
    return _wait_loaded(backend)


def test_surface_matches_the_other_engines():
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    shared = {
        "begin_load",
        "generate",
        "cancel_generate",
        "unload",
        "status",
        "load_progress",
        "generate_progress",
        "loading_repo_ids",
        "loaded_repo_ids",
        "download_plan",
        "preflight_base_access",
        "is_loaded",
    }
    for name in shared:
        assert hasattr(SdCppDiffusionBackend, name)
        assert hasattr(mb.MlxDiffusionBackend, name), name
    import inspect

    sd_params = set(inspect.signature(SdCppDiffusionBackend.begin_load).parameters)
    mlx_params = inspect.signature(mb.MlxDiffusionBackend.begin_load).parameters
    # Every keyword the route passes to sd.cpp is accepted, and the route's extra component kwargs land in **_ignored.
    assert sd_params <= set(mlx_params)
    assert any(p.kind is p.VAR_KEYWORD for p in mlx_params.values())
    assert set(inspect.signature(SdCppDiffusionBackend.generate).parameters) <= set(
        inspect.signature(mb.MlxDiffusionBackend.generate).parameters
    )


def test_load_then_status_and_guards(env):
    backend, runtime, calls = env
    assert backend.status()["loaded"] is False
    assert backend.status()["engine"] == "mlx"
    assert _load(backend)["phase"] == "ready"
    status = backend.status()
    assert status["loaded"] and status["engine"] == "mlx"
    assert (
        status["repo_id"] == REPO and status["base_repo"] == BASE and status["family"] == "z-image"
    )
    assert status["gguf_filename"] == GGUF and status["model_kind"] == "gguf"
    assert status["workflows"] == ["txt2img"]
    assert status["mlx"]["precision"] == "float16" and status["mlx"]["text_encoder_bits"] == 4
    assert runtime.builds[0]["model_config"] == "z-image-turbo"
    model_dir = Path(runtime.builds[0]["model_path"])
    for sub in ("text_encoder", "tokenizer", "vae"):
        assert (model_dir / sub).is_symlink()
    assert backend.loaded_repo_ids() == (REPO, BASE)
    assert backend.loading_repo_ids() == ()
    assert calls["dense_bits"] == 4


def test_conversion_is_cached_across_loads(env):
    backend, runtime, calls = env
    _load(backend)
    backend.unload()
    _load(backend)
    assert calls["convert"] == 1
    assert backend.status()["mlx"]["timings"]["convert_cached"] is True


def test_local_files_only_reaches_every_fetch(env):
    backend, _rt, calls = env
    _load(backend, local_files_only = True)
    assert calls["fetch_gguf"][-1][2] is True
    assert calls["fetch_base"][-1] == (BASE, True)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"gguf_filename": None}, "gguf_filename is required"),
        ({"gguf_filename": "z-image-turbo-Q4_K_M.gguf"}, "Q4_K_M"),
        ({"gguf_filename": GGUF, "model_kind": "pipeline"}, "GGUF"),
        ({"gguf_filename": "sdxl-Q4_0.gguf", "family_override": "sdxl"}, "no MLX implementation"),
        ({"gguf_filename": GGUF, "loras": [("a/b", 1.0)]}, "LoRA"),
    ],
)
def test_begin_load_refuses_what_it_cannot_run(env, kwargs, needle):
    backend, _rt, _calls = env
    repo = "unsloth/SDXL-GGUF" if "sdxl" in str(kwargs.get("gguf_filename")) else REPO
    with pytest.raises(ValueError, match = needle):
        backend.begin_load(repo, **kwargs)
    assert backend.loading_repo_ids() == ()


def test_load_failure_surfaces_in_load_progress(env, monkeypatch):
    backend, _rt, _calls = env

    def _boom(*_a, **_k):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(mb.convert, "convert_gguf_to_mflux", _boom)
    progress = _load(backend)
    assert progress["phase"] == "error" and "converter exploded" in progress["error"]
    assert backend.status()["loaded"] is False


def test_generate_batch_seeds_and_progress(env):
    backend, runtime, _calls = env
    _load(backend)
    seen = []
    runtime_generate = runtime.generate

    def _spy(*a, **k):
        inner = k["on_step"]

        def _on_step(n):
            inner(n)
            seen.append(backend.generate_progress())

        k["on_step"] = _on_step
        return runtime_generate(*a, **k)

    runtime.generate = _spy
    result = backend.generate(prompt = "a fox", width = 280, height = 270, steps = 3, seed = 7, batch_size = 2)
    assert result["seeds"] == [7, 8] and result["seed"] == 7
    assert len(result["images"]) == 2
    assert result["images"][0].size == (272, 256)  # rounded down to multiples of 16
    assert (
        result["model_kind"] == "gguf"
        and result["gguf_filename"] == GGUF
        and result["workflow"] == "txt2img"
    )
    assert [p["step"] for p in seen] == [1, 2, 3, 4, 5, 6]
    assert all(p["active"] and p["total_steps"] == 6 for p in seen)
    assert backend.generate_progress()["active"] is False
    # Turbo has no CFG, so guidance is not forwarded and the negative prompt is not recorded.
    assert runtime.generates[0]["guidance"] is None
    assert backend.last_timings["images"][0]["s_per_step"] >= 0
    assert mb.generation_in_flight() is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"init_image": "data:"},
        {"mask_image": "data:"},
        {"reference_images": ["data:"]},
        {"workflow": "edit"},
        {"upscale": 2.0},
        {"prompts": ["a", "b"]},
        {"seeds": [1, 2]},
        {"loras": [("x/y", 1.0)]},
        {"controlnet": ("c", "img", "canny", 1.0, 0.0, 1.0)},
    ],
)
def test_generate_refuses_unsupported_workflows(env, kwargs):
    backend, _rt, _calls = env
    _load(backend)
    with pytest.raises(ValueError):
        backend.generate(prompt = "x", steps = 1, **kwargs)


def test_generate_without_a_model_and_with_a_replaced_model(env):
    backend, _rt, _calls = env
    with pytest.raises(RuntimeError, match = DIFFUSION_NOT_LOADED_MSG):
        backend.generate(prompt = "x", steps = 1)
    _load(backend)
    stale = load_identity("someone/else", BASE, "z-image")
    with pytest.raises(DiffusionModelReplacedError):
        backend.generate(prompt = "x", steps = 1, expected_load = stale)
    ok = load_identity(REPO, BASE, "z-image")
    assert backend.generate(prompt = "x", steps = 1, seed = 1, width = 256, height = 256, expected_load = ok)[
        "images"
    ]


def test_cancel_stops_at_the_next_step(env):
    backend, runtime, _calls = env
    _load(backend)
    assert backend.cancel_generate() is False
    runtime.step_gate = threading.Event()
    errors = []

    def _run():
        try:
            backend.generate(prompt = "x", steps = 4, seed = 1, width = 256, height = 256)
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target = _run)
    worker.start()
    assert runtime.in_step.wait(5)
    assert mb.generation_in_flight() is False  # the module singleton is not this instance
    assert backend.generate_progress()["active"] is True
    assert backend.cancel_generate() is True
    runtime.step_gate.set()
    worker.join(5)
    assert errors == [DIFFUSION_CANCELLED_MSG]
    assert backend.generate_progress()["active"] is False
    assert backend.cancel_generate() is False


def test_unload_frees_and_cancels(env):
    backend, runtime, _calls = env
    _load(backend)
    status = backend.unload()
    assert status["loaded"] is False and status["engine"] == "mlx"
    assert runtime.freed == 1
    assert backend.loaded_repo_ids() == ()
    with pytest.raises(RuntimeError, match = DIFFUSION_NOT_LOADED_MSG):
        backend.generate(prompt = "x", steps = 1)


def test_unload_during_load_waits_for_the_build_and_discards_it(env, monkeypatch):
    # A build cannot be interrupted, and the router treats unload()'s return as "memory is free"
    # before it activates another engine, so unload must wait for the superseded worker.
    backend, runtime, _calls = env
    gate = threading.Event()
    started = threading.Event()
    real_build = runtime.build

    def _slow_build(*a, **k):
        started.set()
        gate.wait(10)
        return real_build(*a, **k)

    runtime.build = _slow_build
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    assert started.wait(5)
    assert backend.loading_repo_ids() == (REPO, BASE, "unsloth/Z-Image-Turbo")
    with pytest.raises(RuntimeError, match = "already in progress"):
        backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    done = threading.Event()
    unloader = threading.Thread(target = lambda: (backend.unload(), done.set()))
    unloader.start()
    assert not done.wait(0.5), "unload() returned while the superseded build was still running"
    freed_before = runtime.freed
    gate.set()
    assert done.wait(5)
    unloader.join(5)
    assert runtime.freed == freed_before + 1  # the superseded worker freed what it built
    assert backend.status()["loaded"] is False
    assert backend.load_progress()["phase"] is None


def test_a_load_racing_unload_waits_until_the_superseded_build_is_gone(env):
    backend, runtime, _calls = env
    gate = threading.Event()
    started = threading.Event()
    real_build = runtime.build

    def _slow_build(*a, **k):
        started.set()
        gate.wait(10)
        return real_build(*a, **k)

    runtime.build = _slow_build
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    assert started.wait(5)
    unloader = threading.Thread(target = backend.unload)
    unloader.start()
    deadline = time.time() + 5
    while backend.loading_repo_ids() and time.time() < deadline:
        time.sleep(0.01)
    try:
        with pytest.raises(RuntimeError, match = "unload is in progress"):
            backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
        assert len(runtime.builds) == 0
    finally:
        gate.set()
        unloader.join(5)
    assert len(runtime.builds) == 1  # the superseded build, discarded
    runtime.build = real_build
    assert _load(backend)["phase"] == "ready"
    assert len(runtime.builds) == 2


def test_a_cache_only_load_resolves_the_variant_without_the_hub(env, monkeypatch):
    import core.inference.diffusion as diffusion
    import hub.utils.companion_assets as companion

    backend, runtime, calls = env
    monkeypatch.setattr(mb, "resolve_variant_base", _REAL_RESOLVE_VARIANT_BASE, raising = False)

    def _no_hub(*_a, **_k):
        raise AssertionError("a cache-only load read the model card")

    monkeypatch.setattr(diffusion, "_hf_base_model", _no_hub)
    links = {REPO.lower(): ["Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image", "Qwen/Qwen3-4B"]}
    monkeypatch.setattr(companion, "read_companion_links", lambda: links)
    _load(backend, local_files_only = True)
    assert calls["fetch_base"][-1] == ("Tongyi-MAI/Z-Image", True)
    assert runtime.builds[-1]["model_config"] == "z-image"
    backend.unload()
    links.clear()
    _load(backend, local_files_only = True)
    assert calls["fetch_base"][-1] == (BASE, True)


def test_a_cache_only_undistilled_pick_with_no_record_builds_the_base_config(env, monkeypatch):
    import hub.utils.companion_assets as companion

    backend, runtime, calls = env
    monkeypatch.setattr(mb, "resolve_variant_base", _REAL_RESOLVE_VARIANT_BASE, raising = False)
    monkeypatch.setattr(companion, "read_companion_links", lambda: {})
    backend.begin_load(
        "unsloth/Z-Image-GGUF",
        gguf_filename = "z-image-Q4_0.gguf",
        model_kind = "gguf",
        local_files_only = True,
    )
    assert _wait_loaded(backend)["phase"] == "ready"
    assert calls["fetch_base"][-1] == ("Tongyi-MAI/Z-Image", True)
    assert runtime.builds[-1]["model_config"] == "z-image"


@pytest.mark.parametrize(
    "repo_id, gguf_filename, expected",
    [
        ("unsloth/Z-Image-GGUF", "z-image-Q4_K_M.gguf", "Tongyi-MAI/Z-Image"),
        ("unsloth/Z-Image-GGUF", "z-image-BF16.gguf", "Tongyi-MAI/Z-Image"),
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", BASE),
        ("unsloth/Z-Image-GGUF", "z-image-turbo-Q4_0.gguf", BASE),
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-Q4_0.gguf", BASE),
        ("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K.gguf", BASE),
        ("someone/my-gguf-models", "model-Q4_K_M.gguf", BASE),
        ("someone/Z-Image-GGUF", "model-Q4_K_M.gguf", "Tongyi-MAI/Z-Image"),
        ("someone/z-image-turbo", "model-Q4_K_M.gguf", BASE),
    ],
)
def test_the_name_resolves_the_z_image_variant_offline(
    monkeypatch, repo_id, gguf_filename, expected
):
    import hub.utils.companion_assets as companion

    fam = ZIMAGE
    monkeypatch.setattr(companion, "read_companion_links", lambda: {})
    assert mb._cached_variant_base(repo_id, fam, gguf_filename, BASE) == expected
    assert mb.mlx_model_config_for(fam, expected) == (
        "z-image" if expected != BASE else "z-image-turbo"
    )


def test_a_recorded_default_does_not_overrule_an_undistilled_name(monkeypatch):
    import hub.utils.companion_assets as companion

    fam = ZIMAGE
    links = {"unsloth/z-image-gguf": [BASE]}
    monkeypatch.setattr(companion, "read_companion_links", lambda: links)
    assert mb._cached_variant_base("unsloth/Z-Image-GGUF", fam, "z-image-Q4_K_M.gguf", BASE) == (
        "Tongyi-MAI/Z-Image"
    )
    links["unsloth/z-image-turbo-gguf"] = ["Tongyi-MAI/Z-Image"]
    assert (
        mb._cached_variant_base("unsloth/Z-Image-Turbo-GGUF", fam, "z-image-turbo-Q4_0.gguf", BASE)
        == "Tongyi-MAI/Z-Image"
    )


def test_a_local_undistilled_gguf_resolves_without_the_hub(tmp_path, monkeypatch):
    import core.inference.diffusion as diffusion

    fam = ZIMAGE
    monkeypatch.setattr(diffusion, "_remember_companion_base", lambda *a: None)
    (tmp_path / "z-image-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"x")
    resolve = _REAL_RESOLVE_VARIANT_BASE
    assert resolve(str(tmp_path), fam, None, "z-image-Q4_K_M.gguf", fallback = BASE) == (
        "Tongyi-MAI/Z-Image"
    )
    assert resolve(str(tmp_path), fam, None, "z-image-turbo-Q4_K_M.gguf", fallback = BASE) == BASE


def test_policy_helpers(monkeypatch):
    for name in (
        "UNSLOTH_DIFFUSION_MLX_TE_BITS",
        "UNSLOTH_DIFFUSION_MLX_DENSE_BITS",
        "UNSLOTH_DIFFUSION_MLX_PRECISION",
    ):
        monkeypatch.delenv(name, raising = False)
    gib = 1024**3
    assert mb.text_encoder_bits(8 * gib) == 4 and mb.text_encoder_bits(16 * gib) == 4
    assert mb.text_encoder_bits(32 * gib) == 8
    assert mb.dense_bits(7 * gib) == 4 and mb.dense_bits(64 * gib) == 0
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX_TE_BITS", "16")
    assert mb.text_encoder_bits(8 * gib) is None
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX_DENSE_BITS", "3")  # not allowed: ignored
    assert mb.dense_bits(64 * gib) == 0
    assert mb.compute_precision_name("Apple M1 (Virtual)") == "float16"
    assert mb.compute_precision_name("Apple M2 Max") == "float16"
    assert mb.compute_precision_name("Apple M3 Pro") == "bfloat16"
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX_PRECISION", "bf16")
    assert mb.compute_precision_name("Apple M1") == "bfloat16"


def test_importing_the_engine_does_not_import_mlx():
    import subprocess
    import sys

    code = (
        "import sys; import core.inference.mlx_diffusion_backend as m; "
        "print(any(k == 'mlx' or k.startswith(('mlx.', 'mflux')) for k in sys.modules))"
    )
    backend_root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", code], cwd = backend_root, capture_output = True, text = True, timeout = 300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_a_replacing_load_frees_the_old_model_first(env):
    backend, runtime, _calls = env
    _load(backend)
    freed_before = runtime.freed
    seen = []
    real_build = runtime.build

    def _build(*a, **k):
        seen.append(backend.status()["loaded"])
        return real_build(*a, **k)

    runtime.build = _build
    _load(backend)
    assert seen == [False]
    assert runtime.freed == freed_before + 1
    assert backend.status()["loaded"] is True


def test_a_cancelled_run_releases_its_frames():
    # A cancel unwinds through the model's own frames; clearing them is what lets unload() free the model.
    import gc
    import weakref

    class Model:
        pass

    def _inner(model):
        raise mb.MlxCancelled("stop")

    model = Model()
    ref = weakref.ref(model)
    try:
        _inner(model)
    except mb.MlxCancelled as exc:
        kept = exc
        mb.drop_exception_frames(exc)
    del model
    gc.collect()
    assert ref() is None
    assert kept.__traceback__ is None


def test_a_cached_exception_below_the_load_does_not_pin_the_model(env, monkeypatch):
    # The Hub helpers memoize import failures, and a memoized traceback keeps every frame up the
    # stack, the load worker's included. The worker must not leave the model in its locals.
    import gc
    import weakref

    backend, runtime, _calls = env
    cache = {}
    real_fetch = mb.fetch_gguf

    def _fetch(*a, **k):
        try:
            raise ModuleNotFoundError("No module named 'unsloth_zoo'")
        except ModuleNotFoundError as exc:
            cache["error"] = exc  # what hf_xet_fallback._load_shared does
        return real_fetch(*a, **k)

    class Model(dict):
        pass

    monkeypatch.setattr(mb, "fetch_gguf", _fetch)
    real_build = runtime.build
    runtime.build = lambda *a, **k: Model(real_build(*a, **k))
    _load(backend)
    ref = weakref.ref(backend._state.model)
    backend.unload()
    gc.collect()
    assert "error" in cache
    assert ref() is None


def test_generate_rejects_sizes_beyond_the_family_limits(env):
    backend, runtime, _calls = env
    _load(backend)
    with pytest.raises(ValueError):
        backend.generate(prompt = "x", steps = 1, width = 4096, height = 4096)
    assert runtime.generates == []


def test_a_failed_generation_does_not_pin_the_model(env):
    # Any error raised inside the render holds the model in its frames; a caller or logger that
    # keeps the exception must not keep the model alive past unload().
    import gc
    import weakref

    backend, runtime, _calls = env

    class Model(dict):
        pass

    real_build = runtime.build
    runtime.build = lambda *a, **k: Model(real_build(*a, **k))

    def _boom(model, **_k):
        raise RuntimeError("metal failure")

    runtime.generate = _boom
    _load(backend)
    ref = weakref.ref(backend._state.model)
    with pytest.raises(RuntimeError, match = "metal failure") as info:
        backend.generate(prompt = "x", steps = 1, width = 256, height = 256)
    kept = info.value  # what an error handler or log record would hold
    backend.unload()
    gc.collect()
    assert ref() is None
    assert kept.__traceback__ is not None  # the traceback text survives for the log


def test_generated_images_record_the_mlx_build(env):
    backend, _rt, _calls = env
    _load(backend)
    result = backend.generate(prompt = "x", steps = 1, width = 256, height = 256, seed = 1)
    assert result["text_encoder_quant"] == "mlx_int4"
    assert result["transformer_quant"] == "mlx_int4_dense"


def test_an_undistilled_base_tag_selects_the_base_config_and_companions(env, monkeypatch):
    # No explicit base_repo, as for a normal Hub pick: the repo's base_model card tag decides.
    import core.inference.diffusion as diffusion

    backend, runtime, calls = env
    monkeypatch.setattr(mb, "resolve_variant_base", _REAL_RESOLVE_VARIANT_BASE, raising = False)
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo_id, token: "Tongyi-MAI/Z-Image")
    _load(backend)
    assert calls["fetch_base"][-1][0] == "Tongyi-MAI/Z-Image"
    assert runtime.builds[-1]["model_config"] == "z-image"
    assert backend.status()["base_repo"] == "Tongyi-MAI/Z-Image"


@pytest.mark.parametrize(
    "card_tag, expected_base, expected_config",
    [
        (None, "Tongyi-MAI/Z-Image", "z-image"),
        ("Tongyi-MAI/Z-Image-Turbo", BASE, "z-image-turbo"),
    ],
)
def test_an_online_undistilled_pick_without_a_card_tag_keeps_the_named_variant(
    env, monkeypatch, card_tag, expected_base, expected_config
):
    # A failed card read returns no tag: the name still decides, a real tag still wins.
    import core.inference.diffusion as diffusion

    backend, runtime, calls = env
    monkeypatch.setattr(mb, "resolve_variant_base", _REAL_RESOLVE_VARIANT_BASE, raising = False)
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo_id, token: card_tag)
    monkeypatch.setattr(diffusion, "_remember_companion_base", lambda *a: None)
    backend.begin_load("unsloth/Z-Image-GGUF", gguf_filename = "z-image-Q4_0.gguf", model_kind = "gguf")
    assert _wait_loaded(backend)["phase"] == "ready"
    assert calls["fetch_base"][-1] == (expected_base, False)
    assert runtime.builds[-1]["model_config"] == expected_config


def test_a_resolver_error_keeps_the_named_variant(monkeypatch):
    import core.inference.diffusion as diffusion

    def _boom(*_a, **_k):
        raise RuntimeError("hub down")

    monkeypatch.setattr(diffusion, "_hf_base_model", _boom)
    out = mb.resolve_variant_base(
        "unsloth/Z-Image-GGUF", ZIMAGE, None, "z-image-Q4_0.gguf", fallback = BASE
    )
    assert out == "Tongyi-MAI/Z-Image"


@pytest.mark.parametrize(
    "folder, expected", [("Z-Image", "z-image"), ("Z-Image-Turbo", "z-image-turbo")]
)
def test_an_explicit_local_base_selects_the_config_its_name_spells(env, tmp_path, folder, expected):
    backend, runtime, calls = env
    local_base = tmp_path / "models" / folder
    local_base.mkdir(parents = True)
    _load(backend, base_repo = str(local_base))
    assert calls["fetch_base"][-1][0] == str(local_base)
    assert runtime.builds[-1]["model_config"] == expected
    assert backend.status()["base_repo"] == str(local_base)


def test_base_fetch_is_cancellable_between_files(monkeypatch):
    import huggingface_hub
    import utils.hf_xet_fallback as xet

    files = [
        "text_encoder/a.safetensors",
        "text_encoder/b.safetensors",
        "vae/c.safetensors",
        "README.md",
    ]
    info = type(
        "Info", (), {"sha": "abc", "siblings": [type("S", (), {"rfilename": f}) for f in files]}
    )
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, *a, **k: info)
    cancel = threading.Event()
    fetched = []

    def _download(
        repo,
        filename,
        token,
        *,
        cancel_event = None,
        **_k,
    ):
        assert cancel_event is cancel
        fetched.append(filename)
        cancel.set()  # an unload lands during the first file
        return f"/hub/models--x/snapshots/abc/{filename}"

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)
    with pytest.raises(mb.MlxCancelled):
        mb.fetch_base_snapshot(
            "Tongyi-MAI/Z-Image-Turbo", None, local_files_only = False, cancel_event = cancel
        )
    assert fetched == ["text_encoder/a.safetensors"]
    cancel.clear()
    fetched.clear()
    monkeypatch.setattr(
        xet,
        "hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: fetched.append(fn) or f"/hub/snapshots/abc/{fn}",
    )
    out = mb.fetch_base_snapshot(
        "Tongyi-MAI/Z-Image-Turbo", None, local_files_only = False, cancel_event = cancel
    )
    assert out == "/hub/snapshots/abc"
    assert fetched == files[:3]  # README is outside the allow patterns


@pytest.mark.parametrize("predicted", ["mlx", None])
def test_the_load_route_takes_the_gpu_arbiter_for_mlx_on_a_cpu_torch_host(monkeypatch, predicted):
    # torch can resolve "cpu" on an Apple-Silicon Mac, but MLX still drives the Apple GPU: the route
    # must evict / own the GPU through the arbiter instead of releasing it. Also when the prediction
    # failed and only the activation says MLX.
    import asyncio
    import types
    from unittest.mock import create_autospec

    import core.inference.diffusion_device as device_module
    import core.inference.diffusion_engine_router as router_module
    import core.inference.gpu_arbiter as arbiter
    from models.inference import DiffusionLoadRequest
    from routes.inference import load_diffusion_model_gated

    engine = create_autospec(mb.MlxDiffusionBackend, instance = True)
    engine.begin_load.return_value = {"loaded": False, "repo_id": None}
    engine.preflight_base_access.return_value = None

    def _predict(*a, **k):
        if predicted is None:
            raise RuntimeError("probe failed")
        return predicted

    monkeypatch.setattr(router_module, "predict_engine", _predict)
    monkeypatch.setattr(router_module, "active_engine_name", lambda: "mlx")
    monkeypatch.setattr(router_module, "engine_for", lambda name: engine)
    monkeypatch.setattr(router_module, "select_and_activate_engine", lambda *a, **k: engine)
    monkeypatch.setattr(router_module, "begin_load_on", lambda _engine, start: start())
    monkeypatch.setattr(router_module, "annotate_status", lambda status: status)
    monkeypatch.setattr(
        device_module,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(device = "cpu"),
    )
    monkeypatch.setattr("routes.inference._guard_diffusion_load_against_training", lambda: None)

    async def _no_ordinal(_gpu_ids):
        return None

    monkeypatch.setattr("routes.inference._selected_gpu_ordinal", _no_ordinal)
    calls = []
    monkeypatch.setattr(
        arbiter, "acquire_for_request", lambda owner, fn: calls.append("acquire") or fn()
    )
    monkeypatch.setattr(arbiter, "release", lambda owner: calls.append("release"))
    asyncio.run(
        load_diffusion_model_gated(
            DiffusionLoadRequest(model_path = REPO, gguf_filename = GGUF),
            "test-user",
            user_initiated = False,
        )
    )
    engine.begin_load.assert_called_once()
    assert calls == ["acquire"]


def test_the_anonymous_token_sentinel_reaches_every_fetch(env, monkeypatch):
    backend, _rt, _calls = env
    seen = []
    real_fetch_gguf, real_fetch_base = mb.fetch_gguf, mb.fetch_base_snapshot
    monkeypatch.setattr(
        mb,
        "fetch_gguf",
        lambda r, f, tok, **k: (seen.append(tok), real_fetch_gguf(r, f, tok, **k))[1],
    )
    monkeypatch.setattr(
        mb,
        "fetch_base_snapshot",
        lambda b, tok, **k: (seen.append(tok), real_fetch_base(b, tok, **k))[1],
    )
    monkeypatch.setattr(
        mb,
        "resolve_variant_base",
        lambda r, f, tok, *a, fallback, **k: (seen.append(tok), fallback)[1],
    )
    assert _load(backend, hf_token = False)["phase"] == "ready"
    assert seen == [False, False, False]
    seen.clear()
    backend.unload()
    assert _load(backend, hf_token = "  hf_x  ")["phase"] == "ready"
    assert seen == ["hf_x", "hf_x", "hf_x"]
    seen.clear()
    backend.unload()
    assert _load(backend, hf_token = "  ")["phase"] == "ready"
    assert seen == [None, None, None]


def test_unload_racing_begin_load_never_joins_an_unstarted_worker(env, monkeypatch):
    backend, _rt, _calls = env
    started = []
    real_thread = mb.account_thread

    def _thread(**kw):
        worker = real_thread(**kw)
        real_start = worker.start

        def _start():
            # Publication and start share one critical section, so unload() never sees an unstarted worker.
            held = not backend._lock.acquire(blocking = False)
            if not held:
                backend._lock.release()
            assert held
            started.append(worker)
            real_start()

        worker.start = _start
        return worker

    monkeypatch.setattr(mb, "account_thread", _thread)
    _load(backend)
    assert len(started) == 1


def test_a_load_cancelled_while_waiting_on_generation_skips_the_build(env):
    backend, runtime, _calls = env
    _load(backend)
    runtime.step_gate = threading.Event()
    errors = []
    gen = threading.Thread(
        target = lambda: _swallow(errors, lambda: backend.generate(prompt = "x", steps = 1))
    )
    gen.start()
    assert runtime.in_step.wait(5)
    builds_before = len(runtime.builds)
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    deadline = time.time() + 5
    while backend._state is not None and time.time() < deadline:
        time.sleep(0.01)
    backend._cancel_event.set()
    runtime.step_gate.set()
    gen.join(5)
    backend._load_thread.join(5)
    assert len(runtime.builds) == builds_before
    assert backend.status()["loaded"] is False


def _swallow(errors, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)


@pytest.mark.parametrize("stage", ["build", "cache_text_encodings"])
def test_a_failed_build_frees_the_runtime(env, stage):
    backend, runtime, _calls = env

    def _boom(*_a, **_k):
        raise MemoryError("[metal::malloc] out of memory")

    setattr(runtime, stage, _boom)
    progress = _load(backend)
    assert progress["phase"] == "error" and "out of memory" in progress["error"]
    assert runtime.freed == 1
    assert backend.status()["loaded"] is False


def test_status_reports_the_engaged_mlx_quantization(env):
    backend, _rt, _calls = env
    _load(backend)
    status = backend.status()
    assert status["text_encoder_quant"] == "mlx_int4"
    assert status["transformer_quant"] == "mlx_int4_dense"


def test_derived_batch_seeds_stay_json_safe(env):
    backend, runtime, _calls = env
    _load(backend)
    result = backend.generate(
        prompt = "x", steps = 1, width = 256, height = 256, seed = 2**53 - 1, batch_size = 2
    )
    assert result["seeds"] == [2**53 - 1, 0]
    assert [g["seed"] for g in runtime.generates] == [2**53 - 1, 0]


def test_a_failed_replacement_hands_the_resident_model_back_to_its_owner(env, monkeypatch):
    import core.inference.gpu_arbiter as arbiter
    import hub.services.models.account_access as account_access

    backend, _rt, _calls = env
    restored = []
    monkeypatch.setattr(
        arbiter, "restore_owner_account", lambda owner, *a, **k: restored.append(owner)
    )
    monkeypatch.setattr(
        account_access, "restore_resident_metadata", lambda kind: restored.append(kind)
    )
    real_fetch_gguf = mb.fetch_gguf
    failing = [True]

    def _fetch(*a, **k):
        if failing[0]:
            raise OSError("fetch failed")
        return real_fetch_gguf(*a, **k)

    monkeypatch.setattr(mb, "fetch_gguf", _fetch)
    assert _load(backend)["phase"] == "error"
    assert restored == []  # nothing resident to hand back
    failing[0] = False
    assert _load(backend)["phase"] == "ready"
    failing[0] = True
    assert _load(backend)["phase"] == "error"
    assert backend.is_loaded
    assert restored == [arbiter.DIFFUSION, "diffusion"]


def _hub_listing(monkeypatch, files):
    import huggingface_hub

    listed = []
    info = type(
        "Info", (), {"sha": "abc", "siblings": [type("S", (), {"rfilename": f}) for f in files]}
    )
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda self, repo, *a, **k: listed.append(repo) or info,
    )
    return listed


def test_base_companions_come_from_the_repo_the_plan_staged(monkeypatch):
    import utils.hf_xet_fallback as xet

    files = ["text_encoder/a.safetensors", "vae/c.safetensors"]
    listed = _hub_listing(monkeypatch, files)
    judged = []
    monkeypatch.setattr(
        mb,
        "prefer_ungated_mirror",
        lambda base, tok = None, *, files = None: judged.append(files) or "unsloth/Z-Image-Turbo",
        raising = False,
    )
    fetched = []
    monkeypatch.setattr(
        xet,
        "hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: fetched.append((repo, fn, k.get("revision")))
        or f"/hub/snapshots/abc/{fn}",
    )
    mb.fetch_base_snapshot(BASE, None, local_files_only = False)
    assert judged == [files]
    assert listed == [BASE, "unsloth/Z-Image-Turbo"]
    assert fetched == [("unsloth/Z-Image-Turbo", f, "abc") for f in files]


def _companion_snapshot(root, complete = True, tokenizer_config = True):
    root.mkdir(parents = True, exist_ok = True)
    for sub in ("text_encoder", "tokenizer", "vae"):
        (root / sub).mkdir(exist_ok = True)
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    if tokenizer_config:
        (root / "tokenizer" / "tokenizer_config.json").write_text("{}")
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (root / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}})
    )
    for shard in shards if complete else shards[:1]:
        (root / "text_encoder" / shard).write_bytes(b"x")
    return str(root)


def test_cache_only_companions_read_the_mirror_and_the_live_cache_root(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    import utils.hf_cache_settings as cache_settings

    live = str(tmp_path / "live")
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: live)
    monkeypatch.setattr(
        mb,
        "prefer_ungated_mirror",
        lambda base, tok = None, **k: "unsloth/Z-Image-Turbo",
        raising = False,
    )
    mirror_snapshot = _companion_snapshot(tmp_path / "live" / "mirror-snap")
    asked = []

    def _snapshot(
        repo,
        *,
        cache_dir = None,
        local_files_only = False,
        **_k,
    ):
        assert local_files_only
        asked.append((repo, cache_dir))
        if (repo, cache_dir) == ("unsloth/Z-Image-Turbo", live):
            return mirror_snapshot
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot)
    out = mb.fetch_base_snapshot(BASE, None, local_files_only = True)
    assert out == mirror_snapshot
    assert asked == [("unsloth/Z-Image-Turbo", live)]
    asked.clear()
    monkeypatch.setattr(mb, "prefer_ungated_mirror", lambda base, tok = None, **k: base)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    with pytest.raises(LocalEntryNotFoundError):
        mb.fetch_base_snapshot(BASE, None, local_files_only = True)
    assert asked and all(repo == BASE for repo, _ in asked)
    assert asked[0] == (BASE, live)


def test_cache_only_companions_skip_an_incomplete_upstream_for_the_staged_mirror(
    monkeypatch, tmp_path
):
    # The upstream holds some weights, so the fetch-repo choice keeps it, but the plan staged the mirror.
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    import utils.hf_cache_settings as cache_settings

    live = str(tmp_path / "live")
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: live)
    monkeypatch.setattr(mb, "prefer_ungated_mirror", lambda base, tok = None, **k: base)
    snapshots = {
        BASE: _companion_snapshot(tmp_path / "upstream", complete = False),
        "unsloth/Z-Image-Turbo": _companion_snapshot(tmp_path / "mirror"),
    }

    def _snapshot(repo, *, cache_dir = None, local_files_only = False, **_k):
        assert local_files_only
        if cache_dir != live or repo not in snapshots:
            raise LocalEntryNotFoundError("not cached")
        return snapshots[repo]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot)
    assert mb.fetch_base_snapshot(BASE, None, local_files_only = True) == snapshots["unsloth/Z-Image-Turbo"]
    del snapshots["unsloth/Z-Image-Turbo"]
    with pytest.raises(LocalEntryNotFoundError, match = "complete"):
        mb.fetch_base_snapshot(BASE, None, local_files_only = True)


def _mlx_route_stubs(monkeypatch, *, predicted = "mlx"):
    import core.inference.diffusion_engine_router as router_module
    import core.inference.gpu_arbiter as arbiter

    touched = []
    monkeypatch.setattr(router_module, "predict_engine", lambda *a, **k: predicted)
    monkeypatch.setattr(router_module, "active_engine_name", lambda: "diffusers")
    monkeypatch.setattr(
        router_module, "engine_for", lambda name: touched.append(("engine_for", name))
    )
    monkeypatch.setattr(
        router_module,
        "select_and_activate_engine",
        lambda *a, **k: touched.append("select"),
    )
    monkeypatch.setattr(arbiter, "acquire_for_request", lambda *a, **k: touched.append("acquire"))
    monkeypatch.setattr("routes.inference._guard_diffusion_load_against_training", lambda: None)
    monkeypatch.setattr("routes.inference._training_is_active", lambda: False)

    async def _no_ordinal(_gpu_ids, **_k):
        return None

    monkeypatch.setattr("routes.inference._selected_gpu_ordinal", _no_ordinal)
    return touched


def test_the_load_route_refuses_mlx_loras_before_activation_or_eviction(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from models.inference import DiffusionLoadRequest
    from routes.inference import load_diffusion_model_gated

    touched = _mlx_route_stubs(monkeypatch)
    request = DiffusionLoadRequest(
        model_path = REPO, gguf_filename = GGUF, loras = [{"id": "a/b", "weight": 1.0}]
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(load_diffusion_model_gated(request, "test-user", user_initiated = False))
    assert err.value.status_code == 400
    assert "LoRA" in err.value.detail
    assert touched == []


def test_the_download_plan_refuses_mlx_loras_before_planning(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from models.inference import DiffusionLoadRequest
    from routes import inference as routes

    touched = _mlx_route_stubs(monkeypatch)
    monkeypatch.setattr(
        mb.MlxDiffusionBackend,
        "download_plan",
        lambda self, *a, **k: touched.append("plan") or {},
    )
    request = DiffusionLoadRequest(
        model_path = REPO, gguf_filename = GGUF, loras = [{"id": "a/b", "weight": 1.0}]
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(routes.diffusion_download_plan(request, current_subject = "u"))
    assert err.value.status_code == 400
    assert "LoRA" in err.value.detail
    assert "plan" not in touched


def test_a_concurrent_unload_waits_for_the_active_teardown(env):
    backend, runtime, _calls = env
    gate = threading.Event()
    started = threading.Event()
    real_build = runtime.build

    def _slow_build(*a, **k):
        started.set()
        gate.wait(10)
        return real_build(*a, **k)

    runtime.build = _slow_build
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    assert started.wait(5)
    first = threading.Thread(target = backend.unload)
    first.start()
    deadline = time.time() + 5
    while not backend._unloading and time.time() < deadline:
        time.sleep(0.01)
    done = threading.Event()
    second = threading.Thread(target = lambda: (backend.unload(), done.set()))
    second.start()
    try:
        assert not done.wait(
            0.5
        ), "a second unload returned while the first was still joining the build"
    finally:
        gate.set()
        first.join(5)
        second.join(5)
    assert done.is_set()
    assert runtime.freed == 1


def test_a_cancelled_load_keeps_its_repos_draining_until_the_worker_is_joined(env, monkeypatch):
    import core.inference.diffusion_engine_router as router
    from hub.services.models import deletion

    backend, runtime, _calls = env
    monkeypatch.setattr(router, "get_active_diffusion_engine", lambda: backend)
    gate = threading.Event()
    started = threading.Event()
    real_build = runtime.build

    def _slow_build(*a, **k):
        started.set()
        gate.wait(10)
        return real_build(*a, **k)

    runtime.build = _slow_build
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    assert started.wait(5)
    unloader = threading.Thread(target = backend.unload)
    unloader.start()
    deadline = time.time() + 5
    while not backend._unloading and time.time() < deadline:
        time.sleep(0.01)
    try:
        assert backend.loading_repo_ids() == ()
        assert REPO in backend.draining_repo_ids()
        assert "still releasing" in (deletion._diffusion_blocks_delete(REPO) or "")
    finally:
        gate.set()
        unloader.join(5)
    assert backend.draining_repo_ids() == ()


def test_online_companions_land_under_one_cache_root(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub import constants

    import utils.hf_cache_settings as cache_settings
    import utils.hf_xet_fallback as xet

    files = ["text_encoder/a.safetensors", "tokenizer/b.json", "vae/c.safetensors"]
    _hub_listing(monkeypatch, files)
    monkeypatch.setattr(mb, "prefer_ungated_mirror", lambda base, tok = None, **k: base)
    live, old = str(tmp_path / "live"), str(tmp_path / "old")
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: live)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", old)
    # Only the first file survives under the import-time root; the rest are cached nowhere.
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, fn, *, cache_dir = None, **k: (
            f"{old}/snapshots/abc/{fn}" if cache_dir in (None, old) and fn == files[0] else None
        ),
    )
    roots = []

    def _download(
        repo,
        fn,
        tok,
        *,
        cache_dir = None,
        reuse_other_cache_root = False,
        **k,
    ):
        root = old if reuse_other_cache_root and fn == files[0] else (cache_dir or live)
        roots.append(root)
        return f"{root}/snapshots/abc/{fn}"

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)
    out = mb.fetch_base_snapshot(BASE, None, local_files_only = False)
    assert roots == [live] * len(files)
    assert out == f"{live}/snapshots/abc"


def test_cache_only_companions_skip_a_snapshot_without_tokenizer_config(monkeypatch, tmp_path):
    # AutoTokenizer cannot resolve the tokenizer class without tokenizer_config.json.
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    import utils.hf_cache_settings as cache_settings

    live = str(tmp_path / "live")
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: live)
    monkeypatch.setattr(mb, "prefer_ungated_mirror", lambda base, tok = None, **k: base)
    snapshots = {
        BASE: _companion_snapshot(tmp_path / "upstream", tokenizer_config = False),
        "unsloth/Z-Image-Turbo": _companion_snapshot(tmp_path / "mirror"),
    }

    def _snapshot(repo, *, cache_dir = None, local_files_only = False, **_k):
        if cache_dir != live or repo not in snapshots:
            raise LocalEntryNotFoundError("not cached")
        return snapshots[repo]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot)
    assert mb.fetch_base_snapshot(BASE, None, local_files_only = True) == snapshots["unsloth/Z-Image-Turbo"]


def test_a_cancelled_load_keeps_the_companion_mirror_guarded(env, monkeypatch):
    import core.inference.diffusion_engine_router as router
    from hub.services.models import deletion

    backend, runtime, _calls = env
    monkeypatch.setattr(router, "get_active_diffusion_engine", lambda: backend)
    gate = threading.Event()
    started = threading.Event()
    real_build = runtime.build

    def _slow_build(*a, **k):
        started.set()
        gate.wait(10)
        return real_build(*a, **k)

    runtime.build = _slow_build
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    assert started.wait(5)
    assert "unsloth/Z-Image-Turbo" in backend.loading_repo_ids()
    assert "using this repo" in (deletion._diffusion_blocks_delete("unsloth/Z-Image-Turbo") or "")
    unloader = threading.Thread(target = backend.unload)
    unloader.start()
    deadline = time.time() + 5
    while not backend._unloading and time.time() < deadline:
        time.sleep(0.01)
    try:
        assert "unsloth/Z-Image-Turbo" in backend.draining_repo_ids()
        assert "still releasing" in (deletion._diffusion_blocks_delete("unsloth/Z-Image-Turbo") or "")
    finally:
        gate.set()
        unloader.join(5)
    assert backend.draining_repo_ids() == ()


class _PlanRecorder:
    def __init__(self):
        self.calls = []

    def download_plan(self, repo_id, **kwargs):
        self.calls.append(kwargs)
        return {"entries": [], "total_bytes": 0}


def test_the_mlx_plan_forces_off_the_memory_verdict(monkeypatch):
    import core.inference.diffusion as diffusion

    recorder = _PlanRecorder()
    monkeypatch.setattr(diffusion, "get_diffusion_backend", lambda: recorder)
    mb.MlxDiffusionBackend().download_plan(
        REPO, gguf_filename = GGUF, base_repo = BASE, memory_verdict = True
    )
    assert recorder.calls[-1]["memory_verdict"] is False


def test_the_mlx_plan_names_the_base_the_load_will_read(monkeypatch, tmp_path):
    import core.inference.diffusion as diffusion

    recorder = _PlanRecorder()
    monkeypatch.setattr(diffusion, "get_diffusion_backend", lambda: recorder)
    local = tmp_path / "picks"
    local.mkdir()
    gguf = "z-image-Q4_0.gguf"
    (local / gguf).write_bytes(b"GGUF")
    mb.MlxDiffusionBackend().download_plan(str(local), gguf_filename = gguf)
    assert recorder.calls[-1]["base_repo"] == "Tongyi-MAI/Z-Image"
    mb.MlxDiffusionBackend().download_plan(str(local), gguf_filename = "z-image-turbo-Q4_0.gguf")
    assert recorder.calls[-1]["base_repo"] == BASE
    mb.MlxDiffusionBackend().download_plan(str(local), gguf_filename = gguf, base_repo = BASE)
    assert recorder.calls[-1]["base_repo"] == BASE
