# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native stable-diffusion.cpp diffusion backend (the no-GPU tier).

``SdCppDiffusionBackend`` presents the SAME public surface the image routes use on
the diffusers ``DiffusionBackend`` (``begin_load`` / ``load_progress`` / ``generate``
/ ``generate_progress`` / ``unload`` / ``status``), but is backed by the ``sd-cli``
subprocess (``SdCppEngine``) instead of an in-process diffusers pipeline. The engine
router (``diffusion_engine_router.py``) selects this backend only when no usable
CUDA/ROCm/XPU GPU is present, where it is measurably faster and far lighter on RAM
than diffusers (see outputs/sdcpp_cpu).

It reuses the transformer GGUF the diffusers path already downloads and additionally
fetches the per-family single-file VAE + text encoders declared in
``diffusion_families`` (sd-cli cannot read the sharded diffusers components). The
binary is installed lazily on first use; if it is unavailable or the family has no
native mapping, the router falls back to diffusers, so this backend is only ever
asked to run requests it can serve.

Import-light on purpose: no torch / diffusers here, so selecting it on a CPU box
does not drag the heavy GPU stack into the process.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.inference.diffusion_device import resolve_diffusion_device_target
from core.inference.diffusion_families import (
    DiffusionFamily,
    detect_family,
    family_sd_cpp_supported,
    resolve_base_repo,
    resolve_local_gguf_child,
)
from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles
from core.inference.sd_cpp_engine import (
    SdCppCancelled,
    SdCppEngine,
    find_sd_cpp_binary,
)
from loggers import get_logger

logger = get_logger(__name__)

# A sampling-progress line like "  4/4" / "[ 12/ 28]" / "sampling: 50%|...| 14/28".
# We only trust a match whose denominator equals the requested step count, so an
# unrelated "1/100" elsewhere in the log can't move the bar.
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Serialises the one-time binary install so concurrent first-loads don't race on the
# download / extract / chmod.
_install_lock = threading.Lock()


def ensure_sd_cpp_binary(*, allow_install: bool = True, accelerator: str = "cpu") -> Optional[str]:
    """Path to a usable ``sd-cli`` binary, installing the prebuilt once if needed.

    Returns the binary path, or None when it is absent and cannot be installed
    (install disabled, no network, unsupported platform). Never raises -- a None
    return is the router's signal to fall back to diffusers.
    """
    found = find_sd_cpp_binary()
    if found:
        return found
    if not allow_install:
        return None
    with _install_lock:
        # Re-check inside the lock: a concurrent first-load may have installed it.
        found = find_sd_cpp_binary()
        if found:
            return found
        try:
            import sys

            studio_dir = Path(__file__).resolve().parents[3]  # .../studio
            if str(studio_dir) not in sys.path:
                sys.path.insert(0, str(studio_dir))
            from install_sd_cpp_prebuilt import install as _install
        except Exception as exc:  # noqa: BLE001 -- import path / module issues are non-fatal
            logger.warning("sd-cli installer import failed: %s", exc)
            return None
        try:
            path = _install(accelerator = accelerator)
            logger.info("sd-cli installed at %s", path)
            return str(path)
        except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
            logger.warning("sd-cli auto-install failed: %s", exc)
            return None


@dataclass(frozen = True)
class _SdState:
    """The loaded native checkpoint: resolved asset paths + run settings."""

    repo_id: str
    base_repo: str
    family: DiffusionFamily
    device: str
    files: SdCppModelFiles
    vae_format: Optional[str] = None
    native_speed: str = "off"
    offload_flags: tuple[str, ...] = ()
    threads: Optional[int] = None


@dataclass
class _SdLoading:
    """An in-flight asset download, polled for progress."""

    repo_id: str
    base_repo: str
    expected_bytes: int = 0
    downloaded_bytes: int = 0
    error: Optional[str] = None


@dataclass
class _SdGen:
    """An in-flight generation, updated from parsed sd-cli progress lines."""

    total_steps: int
    step: int = 0
    first_step_at: float = 0.0
    eta_seconds: Optional[float] = None


def _estimate_eta(total_steps: int, step: int, first_step_at: float, now: float) -> Optional[float]:
    steps_since_first = step - 1
    if not first_step_at or steps_since_first <= 0:
        return None
    per_step = (now - first_step_at) / steps_since_first
    return max(0.0, (total_steps - step) * per_step)


def _map_guidance(
    fam: DiffusionFamily, guidance: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """(cfg_scale, guidance) for sd-cli from the single diffusers ``guidance`` value.

    FLUX families take a distilled embedded ``--guidance``; everyone else uses real
    classifier-free ``--cfg-scale``. A distilled 0/1 means CFG off (sd-cli's 1.0); a
    value > 1 is real CFG. Mirrors the engine mapping validated in the CPU benchmark.
    """
    if fam.name in ("flux.1", "flux.2-klein"):
        return None, (float(guidance) if guidance is not None else None)
    cfg = float(guidance) if (guidance is not None and guidance > 1.0) else 1.0
    return cfg, None


class SdCppDiffusionBackend:
    """Native sd.cpp backend with the diffusers ``DiffusionBackend`` method surface."""

    def __init__(self, engine: Optional[SdCppEngine] = None) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._engine = engine  # resolved lazily on first load so import stays cheap
        self._state: Optional[_SdState] = None
        self._loading: Optional[_SdLoading] = None
        self._load_token = 0
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        self._gen: Optional[_SdGen] = None

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    def _resolve_engine(self) -> SdCppEngine:
        """The SdCppEngine, installing the binary on first use. Raises if unusable."""
        if self._engine is not None and self._engine.is_available():
            return self._engine
        binary = ensure_sd_cpp_binary(allow_install = _install_allowed())
        if not binary:
            raise RuntimeError("sd-cli (stable-diffusion.cpp) binary is unavailable.")
        self._engine = SdCppEngine(binary = binary)
        return self._engine

    # ── Background load + progress ─────────────────────────────────────────

    def begin_load(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        # diffusers-only knobs accepted (so the route calls both engines uniformly)
        # and ignored -- sd.cpp has no torchao quant / SDPA dispatcher / fbcache.
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Validate, then fetch assets on a daemon thread. Returns at once."""
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the native engine loads single-file GGUF checkpoints only."
            )
        fam = detect_family(repo_id, family_override)
        if fam is None:
            raise ValueError(f"Could not infer a diffusion family for '{repo_id}'.")
        if not family_sd_cpp_supported(fam):
            raise ValueError(f"Family '{fam.name}' has no native sd.cpp asset mapping.")

        base = resolve_base_repo(fam, base_repo)
        with self._lock:
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            self._load_token += 1
            token = self._load_token
            self._cancel_event.clear()
            self._loading = _SdLoading(repo_id = repo_id, base_repo = base)

        threading.Thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                base = base,
                fam = fam,
                hf_token = hf_token,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(
        self,
        *,
        repo_id: str,
        gguf_filename: str,
        base: str,
        fam: DiffusionFamily,
        hf_token: Optional[str],
        _load_token: int,
    ) -> None:
        try:
            # Ensure the binary up front so an install failure surfaces before the
            # multi-GB asset pull (the router also pre-checks, but a forced reload here
            # must not silently download then fail at generate).
            engine = self._resolve_engine()

            assets = self._asset_specs(repo_id, gguf_filename, fam)
            self._set_expected_bytes(assets, hf_token)
            paths = self._fetch_assets(assets, hf_token)

            files = SdCppModelFiles(
                diffusion_model = paths["diffusion_model"],
                vae = paths.get("vae"),
                clip_l = paths.get("clip_l"),
                clip_g = paths.get("clip_g"),
                t5xxl = paths.get("t5xxl"),
                llm = paths.get("llm"),
                qwen2vl = paths.get("qwen2vl"),
            )
            device = resolve_diffusion_device_target().device
            state = _SdState(
                repo_id = repo_id,
                base_repo = base,
                family = fam,
                device = device,
                files = files,
                vae_format = fam.sd_cpp_vae_format,
                native_speed = "off",
                offload_flags = (),
                threads = None,
            )
            # Probe the binary version (cheap) so a broken install fails the load.
            engine.version()
            with self._lock:
                if self._load_token != _load_token:
                    return  # superseded / cancelled
                self._state = state
                self._loading = None
        except SdCppCancelled:
            return
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            if self._load_token != _load_token:
                return
            logger.error("sd_cpp.load_failed: %s", exc)
            with self._lock:
                if self._load_token == _load_token and self._loading is not None:
                    self._loading.error = str(exc)

    def _asset_specs(
        self, repo_id: str, gguf_filename: str, fam: DiffusionFamily
    ) -> list[tuple[str, str, str]]:
        """(repo, filename, kind) for every file sd-cli needs. ``kind`` is the
        SdCppModelFiles field; the transformer reuses the diffusers GGUF."""
        specs: list[tuple[str, str, str]] = [(repo_id, gguf_filename, "diffusion_model")]
        if fam.sd_cpp_vae:
            specs.append((fam.sd_cpp_vae[0], fam.sd_cpp_vae[1], "vae"))
        for terepo, tefile, kind in fam.sd_cpp_text_encoders:
            specs.append((terepo, tefile, kind))
        return specs

    def _set_expected_bytes(
        self, assets: list[tuple[str, str, str]], hf_token: Optional[str]
    ) -> None:
        """Best-effort total download size for the progress bar (0 if unknown)."""
        total = 0
        try:
            from huggingface_hub import HfApi
            api = HfApi(token = hf_token)
            for repo, fn, _ in assets:
                if Path(repo).expanduser().exists():
                    continue
                try:
                    info = api.get_paths_info(repo, paths = [fn], expand = False)
                    for it in info:
                        total += int(getattr(it, "size", 0) or 0)
                except Exception:  # noqa: BLE001 -- one missing size is non-fatal
                    continue
        except Exception:  # noqa: BLE001 -- estimate is best-effort
            total = 0
        loading = self._loading
        if loading is not None:
            loading.expected_bytes = total

    def _fetch_assets(
        self, assets: list[tuple[str, str, str]], hf_token: Optional[str]
    ) -> dict[str, str]:
        """Download every asset (cancellable), returning kind -> local path."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        paths: dict[str, str] = {}
        for repo, fn, kind in assets:
            if self._cancel_event.is_set():
                raise SdCppCancelled("load cancelled")
            local_root = Path(repo).expanduser()
            if kind == "diffusion_model" and local_root.exists():
                path = str(resolve_local_gguf_child(local_root, fn))
            else:
                path = hf_hub_download_with_xet_fallback(
                    repo, fn, hf_token, cancel_event = self._cancel_event
                )
            paths[kind] = path
            with self._lock:
                if self._loading is not None:
                    try:
                        self._loading.downloaded_bytes += os.path.getsize(path)
                    except OSError:
                        pass
        return paths

    def load_progress(self) -> dict[str, Any]:
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)
        downloaded = loading.downloaded_bytes
        expected = loading.expected_bytes
        if expected > 0 and downloaded >= expected * 0.999:
            return _progress("finalizing", min(downloaded, expected), expected, 1.0)
        fraction = min(downloaded / expected, 1.0) if expected > 0 else 0.0
        return _progress("downloading", downloaded, expected, fraction)

    # ── Generate ───────────────────────────────────────────────────────────

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        import tempfile

        from PIL import Image

        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError("No diffusion model is loaded.")
                self._active_generate_cancel = cancel
            engine = self._resolve_engine()
            try:
                if seed is None:
                    seed = int.from_bytes(os.urandom(6), "big") & ((1 << 53) - 1)
                else:
                    seed = int(seed)
                cfg_scale, flux_guidance = _map_guidance(state.family, guidance)
                vae_extra = ["--vae-format", state.vae_format] if state.vae_format else None

                self._gen = _SdGen(total_steps = int(steps))
                images = []
                with tempfile.TemporaryDirectory(prefix = "sdcpp_gen_") as tmpdir:
                    for index in range(max(1, int(batch_size))):
                        if cancel.is_set():
                            raise RuntimeError("Diffusion generation was cancelled.")
                        # Distinct seed per batch image (sd-cli is one image/run here),
                        # so a batch is reproducible image-by-image from the base seed.
                        seed_i = (seed + index) & ((1 << 53) - 1)
                        out_path = str(Path(tmpdir) / f"img_{index}.png")
                        params = SdCppGenParams(
                            prompt = prompt,
                            negative_prompt = negative_prompt or None,
                            width = int(width),
                            height = int(height),
                            steps = int(steps),
                            cfg_scale = cfg_scale,
                            guidance = flux_guidance,
                            seed = seed_i,
                            batch_count = 1,
                        )
                        engine.generate(
                            state.files,
                            params,
                            output_path = out_path,
                            offload = list(state.offload_flags) or None,
                            native_speed = state.native_speed,
                            threads = state.threads,
                            extra_args = vae_extra,
                            on_log = self._on_log,
                            cancel_event = cancel,
                        )
                        with Image.open(out_path) as im:
                            images.append(im.copy())
                if cancel.is_set():
                    raise RuntimeError("Diffusion generation was cancelled.")
                return {"images": images, "seed": int(seed), "repo_id": state.repo_id}
            except SdCppCancelled as exc:
                raise RuntimeError("Diffusion generation was cancelled.") from exc
            finally:
                self._gen = None
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None

    def _on_log(self, line: str) -> None:
        gen = self._gen
        if gen is None or gen.total_steps <= 0:
            return
        for a, b in _STEP_RE.findall(line):
            if int(b) == gen.total_steps:
                now = time.time()
                gen.step = min(int(a), gen.total_steps)
                if gen.first_step_at == 0.0:
                    gen.first_step_at = now
                gen.eta_seconds = _estimate_eta(gen.total_steps, gen.step, gen.first_step_at, now)

    def generate_progress(self) -> dict[str, Any]:
        gen = self._gen
        if gen is None or gen.total_steps <= 0:
            return {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
            }
        return {
            "active": True,
            "step": gen.step,
            "total_steps": gen.total_steps,
            "fraction": min(gen.step / gen.total_steps, 1.0),
            "eta_seconds": gen.eta_seconds,
        }

    # ── Unload / status ──────────────────────────────────────────────────────

    def unload(self) -> dict[str, Any]:
        self._cancel_event.set()
        with self._lock:
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._state = None
            self._load_token += 1
            self._loading = None
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "loaded": False,
                "repo_id": None,
                "family": None,
                "base_repo": None,
                "device": None,
                "dtype": None,
                "cpu_offload": False,
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "text_encoder_quant": None,
                "transformer_quant": None,
                "attention_backend": None,
                "transformer_cache": None,
                "engine": "sd_cpp",
            }
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": "gguf",
            "cpu_offload": False,
            "offload_policy": "none",
            "vae_tiling": False,
            "memory_mode": None,
            "speed_mode": state.native_speed,
            "speed_optims": [],
            "text_encoder_quant": None,
            "transformer_quant": None,
            "attention_backend": None,
            "transformer_cache": None,
            "engine": "sd_cpp",
        }


def _install_allowed() -> bool:
    """Whether lazy binary install is permitted (UNSLOTH_DIFFUSION_SD_CPP_INSTALL)."""
    val = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "auto").strip().lower()
    return val not in ("0", "off", "false", "no")


def _progress(
    phase: Optional[str],
    bytes_downloaded: int = 0,
    bytes_total: int = 0,
    fraction: float = 0.0,
    *,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "bytes_downloaded": bytes_downloaded,
        "bytes_total": bytes_total,
        "fraction": fraction,
        "error": error,
    }


_sd_cpp_backend: Optional[SdCppDiffusionBackend] = None


def get_sd_cpp_backend() -> SdCppDiffusionBackend:
    global _sd_cpp_backend
    if _sd_cpp_backend is None:
        _sd_cpp_backend = SdCppDiffusionBackend()
    return _sd_cpp_backend
