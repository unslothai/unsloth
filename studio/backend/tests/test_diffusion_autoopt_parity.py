# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The auto-optimizer resolves EXACTLY the optimization set every registered family got before it existed.

``fixtures/diffusion_autoopt_parity.json`` was captured from origin/main 5f38afbc2d (before diffusion_autoopt) by
running :func:`snapshot` with :func:`legacy_entry`, which replays the old load-path sequence (patch installs, then
``apply_speed_optims`` per view with the video family branches). This test runs the same cases through the
optimizer (``install_patches`` + ``run_speed_layer``) and asserts equality, for every family in the image and video
registries x speed tier x dtype x offload x GGUF. Stub pipes carry a denoiser of the family's real class name,
module and ``_repeated_blocks``, so every class / module gated install sees what a real load shows it.

Regenerate (only when a registry change is intended): run ``snapshot(legacy_entry)`` against the base tree.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

FIXTURE = Path(__file__).with_name("fixtures") / "diffusion_autoopt_parity.json"

SPEEDS = ("off", "eager", "default", "max")
DTYPES = ("bfloat16", "float16")
_ENV_CLEAR = (
    "UNSLOTH_DIFFUSION_CUDA_GRAPH",
    "UNSLOTH_DIFFUSION_OFFLOAD_CUDA_GRAPH",
    "TORCHDYNAMO_DISABLE",
    "UNSLOTH_DIFFUSION_AUTOOPT_DISCOVER",
)


class _Patch:
    """Minimal attribute patcher (the capture runs outside pytest)."""

    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, obj: Any, name: str, value: Any) -> None:
        missing = object()
        old = getattr(obj, name, missing)
        self._undo.append((obj, name, old, missing))
        setattr(obj, name, value)

    def undo(self) -> None:
        for obj, name, old, missing in reversed(self._undo):
            if old is missing:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, old)
        self._undo.clear()


def _real_class(name: str) -> Any:
    try:
        import diffusers
        return getattr(diffusers, name, None)
    except Exception:  # noqa: BLE001
        return None


_STUB_CLASSES: dict[str, type] = {}


def _stub_class(name: str) -> type:
    import torch

    cls = _STUB_CLASSES.get(name)
    if cls is not None:
        return cls
    real = _real_class(name)

    def compile_repeated_blocks(self, **kwargs):
        if not getattr(self, "_repeated_blocks", None):
            raise ValueError("_repeated_blocks is empty")
        self.compile_calls.append(dict(kwargs))

    def compile(self, **kwargs):  # the U-Net whole-module compile
        self.compile_calls.append({"whole": True, **kwargs})

    def forward(self, *args, **kwargs):
        return None

    ns = {
        "_repeated_blocks": list(getattr(real, "_repeated_blocks", None) or []),
        "_no_split_modules": list(getattr(real, "_no_split_modules", None) or []),
        "compile_repeated_blocks": compile_repeated_blocks,
        "forward": forward,
        "__module__": getattr(real, "__module__", __name__),
    }
    if name == "UNet2DConditionModel":
        ns["compile"] = compile
        del ns["compile_repeated_blocks"]
    cls = type(name, (torch.nn.Module,), ns)
    _STUB_CLASSES[name] = cls
    return cls


def _stub_denoiser(name: str) -> Any:
    module = _stub_class(name)()
    module.compile_calls = []
    return module


def _image_pipe(fam: Any) -> Any:
    pipe = types.SimpleNamespace()
    setattr(
        pipe,
        getattr(fam, "denoiser_attr", "transformer") or "transformer",
        _stub_denoiser(fam.transformer_class),
    )
    return pipe


def _video_pipe(fam: Any) -> Any:
    pipe = types.SimpleNamespace()
    setattr(
        pipe,
        getattr(fam, "denoiser_attr", "transformer") or "transformer",
        _stub_denoiser(fam.transformer_class),
    )
    if getattr(fam, "is_moe", False):
        pipe.transformer_2 = _stub_denoiser(fam.transformer_class)
    return pipe


def _target(dtype_name: str) -> Any:
    import torch
    return types.SimpleNamespace(
        device = "cuda",
        backend = "cuda",
        dtype = getattr(torch, dtype_name),
        ordinal = None,
        torch_device = "cuda:0",
        supports_default_torch_compile = True,
    )


def _engaged(result: Any) -> bool:
    if isinstance(result, dict):
        return any(bool(v) for v in result.values())
    return bool(result)


def _teardown() -> None:
    """Restore every process-global patch a case may have installed (old and new teardown entry points)."""
    from core.inference import diffusion_arch_patches, diffusion_eager_patches

    diffusion_eager_patches.uninstall_patches()
    diffusion_arch_patches.uninstall_arch_patches()
    for mod_name in (
        "diffusion_flux2_rope",
        "diffusion_qwenimage_rope",
        "diffusion_zimage_fused",
        "diffusion_qwenimage21_fused",
        "diffusion_rocm_fused",
        "diffusion_qwenimage21",
        "diffusion_qwenimage21_rope",
        "video_wan_fused",
    ):
        try:
            mod = __import__(f"core.inference.{mod_name}", fromlist = ["uninstall"])
            mod.uninstall()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------- entry points
def legacy_entry(
    backend: str, pipe: Any, target: Any, fam: Any, speed: str, offload: bool, gguf: bool
) -> dict:
    """The pre-optimizer load-path sequence (diffusion.py / video.py at 5f38afbc2d), patch installs + speed layer."""
    from core.inference import diffusion_speed as ds

    patches: dict = {}
    per_view: list = []
    if backend == "image":
        from core.inference.diffusion_arch_patches import (
            install_arch_patches,
            uninstall_arch_patches,
        )
        from core.inference.diffusion_eager_patches import (
            install_compile_safe_patches,
            uninstall_patches,
        )

        if speed != "off":
            patches["eager_fusions"] = _engaged(install_compile_safe_patches())
            patches["arch_fusions"] = _engaged(install_arch_patches())
        else:
            uninstall_patches()
            uninstall_arch_patches()
        from core.inference.diffusion_flux2_rope import install_for_pipe as flux2
        from core.inference.diffusion_qwenimage21 import install_for_pipe as q21
        from core.inference.diffusion_rocm_fused import install_for_pipe as rocm

        patches["qwenimage21_fast_step"] = _engaged(q21(pipe, None))
        patches["flux2_rope"] = _engaged(flux2(pipe, target.dtype, target.device, None))
        patches["rocm_fused"] = _engaged(rocm(pipe, target.dtype, target.device, None))
        applied = ds.apply_speed_optims(
            pipe,
            target,
            is_gguf = gguf,
            family = fam,
            speed_mode = speed,
            cache_active = False,
            cache_engaged = False,
            offload_active = offload,
            denoiser_offloaded = offload,
            stream_int8_gemm = True,
        )
        per_view.append(applied)
        return {"patches": patches, "views": per_view}
    from core.inference import video_wan_fused
    from core.inference.video import _views_for
    from core.inference.video_families import LTX_FAMILIES

    if speed != "off":
        patches["wan_fused"] = _engaged(
            video_wan_fused.install_for_pipe(pipe, target.dtype, target.device)
        )
    else:
        video_wan_fused.uninstall()
    for view in _views_for(pipe, fam):
        if fam.name in LTX_FAMILIES:
            dit = getattr(view, fam.denoiser_attr, None)
            if dit is not None:
                dit._unsloth_compile_static = True
        applied = ds.apply_speed_optims(
            view,
            target,
            is_gguf = False,
            family = fam,
            speed_mode = speed,
            cache_active = False,
            offload_active = offload,
            cuda_graph_default = False,
        )
        if fam.name in LTX_FAMILIES and applied.get("compiled"):
            from core.inference.video_ltx2 import install_stg_compile_adapter
            install_stg_compile_adapter(getattr(view, fam.denoiser_attr, None))
        if not getattr(fam, "cudnn_benchmark", True) and applied.get("cudnn_benchmark"):
            from core.inference.video_ltx2 import disable_cudnn_benchmark
            if disable_cudnn_benchmark():
                applied = {**applied, "cudnn_benchmark": False}
        per_view.append(applied)
    return {"patches": patches, "views": per_view}


def autoopt_entry(
    backend: str, pipe: Any, target: Any, fam: Any, speed: str, offload: bool, gguf: bool
) -> dict:
    """The same load through the optimizer."""
    from core.inference import diffusion_speed as ds
    from core.inference.diffusion_autoopt import apply as ao
    from core.inference.diffusion_autoopt import patches as ap

    audit = ao.new_audit(backend = backend, family = fam)
    results: dict = {}
    if backend == "image":
        for stage in (ap.STAGE_LOAD_EAGER, ap.STAGE_LOAD):
            results.update(
                ao.install_patches(
                    stage,
                    pipe = pipe,
                    target = target,
                    family = fam,
                    dtype = target.dtype,
                    device = target.device,
                    speed_mode = speed,
                    backend = "image",
                    audit = audit,
                )
            )
        applied = ao.run_speed_layer(
            pipe,
            target,
            family = fam,
            backend = "image",
            speed_mode = speed,
            is_gguf = gguf,
            offload_active = offload,
            audit = audit,
            apply = lambda family, log: ds.apply_speed_optims(
                pipe,
                target,
                is_gguf = gguf,
                family = family,
                speed_mode = speed,
                cache_active = False,
                cache_engaged = False,
                offload_active = offload,
                denoiser_offloaded = offload,
                stream_int8_gemm = True,
                logger = log,
            ),
        )
        views = [applied]
    else:
        from core.inference.video import _views_for
        results.update(
            ao.install_patches(
                ap.STAGE_LOAD,
                pipe = pipe,
                target = target,
                family = fam,
                dtype = target.dtype,
                device = target.device,
                speed_mode = speed,
                backend = "video",
                audit = audit,
            )
        )
        views = [
            ao.run_speed_layer(
                view,
                target,
                family = fam,
                backend = "video",
                speed_mode = speed,
                offload_active = offload,
                audit = audit,
                apply = lambda family, log, view = view: ds.apply_speed_optims(
                    view,
                    target,
                    is_gguf = False,
                    family = family,
                    speed_mode = speed,
                    cache_active = False,
                    offload_active = offload,
                    cuda_graph_default = False,
                    logger = log,
                ),
            )
            for view in _views_for(pipe, fam)
        ]
    patches = {k: _engaged(v) for k, v in results.items()}
    _check_audit(audit, views[0], speed)
    return {"patches": patches, "views": views}


def _check_audit(audit: Any, applied: dict, speed: str) -> None:
    """Every speed key lands in the audit with a status that agrees with what engaged, and a reason when it did not."""
    for key, on in applied.items():
        entry = audit.entries.get(key)
        assert entry is not None, key
        if on:
            assert entry["status"] == "engaged", (key, entry)
        else:
            assert entry["status"] in ("declined", "failed", "engaged"), (key, entry)
            if entry["status"] == "declined":
                assert entry["reason"], (key, entry)


# ------------------------------------------------------------------------------------------------------- harness
def _families() -> list[tuple[str, Any]]:
    from core.inference.diffusion_families import _FAMILIES as image
    from core.inference.video_families import _FAMILIES as video
    return [("image", f) for f in image] + [("video", f) for f in video]


def _observe(pipe: Any, out: dict) -> dict:
    mods = {}
    for attr in ("transformer", "transformer_2", "unconditional_transformer", "unet"):
        m = getattr(pipe, attr, None)
        if m is None:
            continue
        mods[attr] = {
            "compile_calls": [
                {
                    k: (v if isinstance(v, (bool, int, float, str, type(None))) else repr(v))
                    for k, v in c.items()
                }
                for c in getattr(m, "compile_calls", [])
            ],
            "static": bool(getattr(m, "_unsloth_compile_static", False)),
            "auto_dynamic": getattr(m, "_unsloth_auto_dynamic", None),
            "int8_gemm": bool(getattr(m, "_unsloth_int8_gemm", 0)),
        }
    return {
        "patches": dict(sorted(out["patches"].items())),
        "applied": [sorted(k for k, v in view.items() if v) for view in out["views"]],
        "keys": [sorted(view) for view in out["views"]],
        "graph_reason": _stable_reason(getattr(pipe, "_unsloth_cuda_graph_reason", None)),
        "graph_after_placement": bool(getattr(pipe, "_unsloth_cuda_graph_after_placement", False)),
        "modules": mods,
    }


def _stable_reason(reason: Any) -> Any:
    """The capture-safe decline names WHY the rewrite did not apply, which depends on what this process imported
    first (``its defining module is not imported`` vs a source mismatch on the stub class); keep the verdict."""
    if isinstance(reason, str) and "was not rewritten" in reason:
        return reason.split("was not rewritten", 1)[0] + "was not rewritten"
    return reason


def snapshot(entry: Callable) -> dict:
    """Run every (family, speed, dtype, offload, gguf) case through ``entry``; returns ``{case_id: observation}``."""
    import torch

    from core.inference import diffusion_cuda_graph, diffusion_speed as ds

    # Real classes (and their lazy diffusers modules) resolve BEFORE torch.cuda is faked below.
    for _backend, fam in _families():
        _stub_class(fam.transformer_class)
    patch = _Patch()
    # A deterministic stand-in for unsloth_zoo's patch_function / restore_original: the real one imports unsloth
    # (GPU detection, transformers patches) the moment torch.cuda reports a device, which a test must never do.
    from core.inference import diffusion_patch_backend

    stash: dict = {}

    def _patch_function(
        target,
        attr,
        new_fn,
        match_level = "relaxed",
        force = False,
    ):
        stash.setdefault((target, attr), getattr(target, attr))
        setattr(target, attr, new_fn)
        return True

    def _restore_original(target, attr):
        if (target, attr) not in stash:
            return False
        setattr(target, attr, stash.pop((target, attr)))
        return True

    patch.setattr(
        diffusion_patch_backend,
        "_HELPERS",
        {"patch": _patch_function, "restore": _restore_original},
    )
    saved_env = {k: os.environ.pop(k, None) for k in _ENV_CLEAR}
    patch.setattr(torch.cuda, "is_available", lambda: True)
    patch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    patch.setattr(diffusion_cuda_graph, "install_cuda_graphs", lambda pipe, logger = None: True)
    # The fp16-accumulation gate asks the real device (datacenter vs consumer); pin it so the snapshot is host-free.
    from core.inference import diffusion_transformer_quant

    patch.setattr(diffusion_transformer_quant, "_is_consumer_gpu", lambda *a, **k: True)
    out: dict = {}
    try:
        for backend, fam in _families():
            for speed in SPEEDS:
                for dtype_name in DTYPES:
                    for offload in (False, True):
                        for gguf in (False, True) if backend == "image" else (False,):
                            case = f"{backend}|{fam.name}|{speed}|{dtype_name}|offload={int(offload)}|gguf={int(gguf)}"
                            pipe = (_image_pipe if backend == "image" else _video_pipe)(fam)
                            flags = ds.snapshot_backend_flags()
                            try:
                                result = entry(
                                    backend, pipe, _target(dtype_name), fam, speed, offload, gguf
                                )
                                out[case] = _observe(pipe, result)
                            except Exception as exc:  # noqa: BLE001 - recorded, compared like any outcome
                                out[case] = {"error": f"{type(exc).__name__}: {exc}"}
                            finally:
                                ds.restore_backend_flags(flags)
                                _teardown()
    finally:
        patch.undo()
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v
    return out


# ---------------------------------------------------------------------------------------------------------- tests
@pytest.fixture(scope = "module")
def captured() -> dict:
    if not FIXTURE.exists():
        pytest.fail(f"missing parity fixture {FIXTURE}")
    return json.loads(FIXTURE.read_text())


def test_fixture_covers_every_registered_family(captured):
    meta = captured["meta"]
    fams = {f"{b}|{f.name}" for b, f in _families()}
    assert fams == set(
        meta["families"]
    ), "a registry family was added / removed: regenerate the fixture"


def test_autoopt_resolves_the_same_optimizations_as_before(captured):
    got = snapshot(autoopt_entry)
    want = captured["cases"]
    assert set(got) == set(want)
    diffs = [case for case in want if got[case] != want[case]]
    assert not diffs, {case: {"before": want[case], "after": got[case]} for case in diffs[:5]}


def test_legacy_sequence_still_matches_fixture(captured):
    """The fixture is not stale: the old sequence over today's primitives still produces it."""
    got = snapshot(legacy_entry)
    want = captured["cases"]
    diffs = [case for case in want if got.get(case) != want[case]]
    assert not diffs, {case: {"before": want[case], "now": got.get(case)} for case in diffs[:5]}


def capture() -> dict:
    """Fixture payload (run against the base tree)."""
    return {
        "meta": {
            "families": sorted(f"{b}|{f.name}" for b, f in _families()),
            "speeds": list(SPEEDS),
            "dtypes": list(DTYPES),
        },
        "cases": snapshot(legacy_entry),
    }
