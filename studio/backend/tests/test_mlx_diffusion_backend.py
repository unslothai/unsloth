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
    DiffusionModelReplacedError,
    load_identity,
)

REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF = "z-image-turbo-Q4_0.gguf"
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

    def _fetch_base(base_repo, hf_token, *, local_files_only):
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
    result = backend.generate(prompt = "a fox", width = 260, height = 250, steps = 3, seed = 7, batch_size = 2)
    assert result["seeds"] == [7, 8] and result["seed"] == 7
    assert len(result["images"]) == 2
    assert result["images"][0].size == (256, 240)  # rounded down to multiples of 16
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
    assert backend.generate(prompt = "x", steps = 1, seed = 1, width = 64, height = 64, expected_load = ok)[
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
            backend.generate(prompt = "x", steps = 4, seed = 1, width = 64, height = 64)
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


def test_unload_during_load_discards_the_result(env, monkeypatch):
    backend, runtime, _calls = env
    gate = threading.Event()
    started = threading.Event()
    real_build = runtime.build

    def _slow_build(*a, **k):
        started.set()
        gate.wait(5)
        return real_build(*a, **k)

    runtime.build = _slow_build
    backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    assert started.wait(5)
    assert backend.loading_repo_ids() == (REPO, BASE)
    with pytest.raises(RuntimeError, match = "already in progress"):
        backend.begin_load(REPO, gguf_filename = GGUF, model_kind = "gguf")
    backend.unload()
    gate.set()
    time.sleep(0.2)
    assert backend.status()["loaded"] is False
    assert backend.load_progress()["phase"] is None


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
