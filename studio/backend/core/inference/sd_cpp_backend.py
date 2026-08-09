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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from core.inference.diffusion_device import resolve_diffusion_device_target
from core.inference.diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    DiffusionFamily,
    detect_family_for_pick,
    family_sd_cpp_supported,
    mirror_repo,
    legacy_source_repo,
    prefer_cached_legacy_source,
    prefer_ungated_mirror,
    resolve_base_repo,
    resolve_local_gguf_child,
    sd_cpp_text_encoders_for,
    supported_family_names,
)
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
)
from core.inference.sd_cpp_args import (
    CPU_BACKEND_FLAGS,
    SdCppGenParams,
    SdCppModelFiles,
    build_img_gen_request,
    is_ggml_unsupported_op_abort,
    offload_flags,
)
from core.inference.sd_cpp_engine import (
    NATIVE_GENERATION_TIMEOUT_S,
    SdCppCancelled,
    SdCppEngine,
    find_sd_cpp_binary,
    find_sd_server_binary,
    is_managed_binary,
    runtime_env,
)
from core.inference.sd_cpp_server import SdCppServer
from loggers import get_logger
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = get_logger(__name__)

# A sampling-progress line ("4/4", "[ 12/ 28]", "sampling: 50%|...| 14/28"). Only a denominator matching the requested step count is trusted, so a stray "1/100" cannot move the bar.
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Serialises the one-time binary install so concurrent first-loads don't race.
_install_lock = threading.Lock()

# Max images per img_gen job; larger Studio batches (up to 32) are split into these chunks.
_MAX_SERVER_BATCH = 8


def _default_threads() -> int:
    """Physical-core thread count for the sd.cpp CPU backend.

    ``threads = None`` lets sd.cpp pick its own default, which is the logical-core
    count (all hyperthreads). For the compute-bound GGML matmuls the diffusion CPU
    path runs, oversubscribing the hyperthreads adds scheduling contention without
    extra throughput, so pin to physical cores (``cpu_count // 2``) instead. Falls
    back to 8 when the count is unknown, and clamps to at least 1."""
    cpu = os.cpu_count()
    return max(1, cpu // 2 if cpu else 8)


def _server_binary_runnable(binary: str) -> bool:
    """Best-effort probe that ``binary`` can actually execute (not just exist).

    Runs ``<binary> --help`` with the same runtime env the server will use, so a present
    but unrunnable build (wrong arch, missing shared libs, no execute bit) is caught before
    a multi-GB asset download. Conservative: only a clear "cannot launch" signal (OSError,
    or the dynamic-loader exit codes 126/127) returns False; anything else is treated as
    runnable so a quirky ``--help`` exit code never blocks a working binary."""
    import subprocess

    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output = True,
            timeout = 20,
            env = runtime_env(binary),
            **windows_hidden_subprocess_kwargs(),
        )
    except OSError:
        return False  # cannot exec at all (wrong arch / no execute bit / missing loader)
    except Exception:  # noqa: BLE001 -- don't block on a flaky probe (timeout etc.)
        return True
    # Negative return code = signal death (e.g. -4 SIGILL from an incompatible prebuilt): launches then crashes, so treat as unavailable.
    return proc.returncode >= 0 and proc.returncode not in (126, 127)


def _usable_or_discard_managed(binary: str) -> bool:
    """True if ``binary`` can be kept; False if it is an unusable copy WE own (now removed).

    ``find_sd_*_binary`` only checks that the path is a file, so an interrupted extraction (or a
    prebuilt for the wrong CPU) left a present-but-unrunnable binary that the installer then never
    retried: every load probed it, rejected it, and fell back to diffusers, so native inference
    stayed off until the user deleted the directory by hand. Probing here closes that loop.

    Only a copy the installer may replace is removed, i.e. one under the installer-owned root that
    carries its ownership marker (``is_managed_binary``). SD_CLI_PATH, UNSLOTH_SD_CPP_PATH, an
    in-tree build, anything on PATH and an unmarked directory at the default path (a user's own
    stable-diffusion.cpp checkout looks exactly like that) are the user's, so an unrunnable one of
    those is reported as-is (the router still rejects it) rather than deleted or reinstalled over.
    Deleting one would also be unrepairable: install() refuses an unmarked non-empty target, so the
    binary would be gone AND the reinstall refused."""
    if _server_binary_runnable(binary):
        return True
    if not is_managed_binary(binary):
        logger.warning(
            "sd.cpp binary %s is not runnable; leaving it alone (not a Studio-owned install we may "
            "replace). Delete its directory to have Studio reinstall the prebuilt.",
            binary,
        )
        return True  # not ours to replace; the router's own probe still refuses it
    logger.warning("managed sd.cpp binary %s is not runnable; removing it so it reinstalls", binary)
    try:
        Path(binary).unlink()
    except OSError as exc:
        logger.warning("could not remove the unusable managed binary %s: %s", binary, exc)
        return True  # cannot repair it; don't spin on a reinstall that will find it again
    return False


def ensure_sd_cpp_binary(*, allow_install: bool = True, accelerator: str = "cpu") -> Optional[str]:
    """Path to a usable ``sd-cli`` binary, installing the prebuilt once if needed.

    Returns the binary path, or None when it is absent and cannot be installed
    (install disabled, no network, unsupported platform). Never raises -- a None
    return is the router's signal to fall back to diffusers.
    """
    found = find_sd_cpp_binary()
    if found and _usable_or_discard_managed(found):
        return found
    if not allow_install:
        return found
    with _install_lock:
        # Re-check inside the lock: a concurrent first-load may have installed it.
        found = find_sd_cpp_binary()
        if found and _usable_or_discard_managed(found):
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


def ensure_sd_server_binary(
    *, allow_install: bool = True, accelerator: str = "cpu"
) -> Optional[str]:
    """Path to a usable ``sd-server`` binary, installing the prebuilt once if needed.

    Unlike ``ensure_sd_cpp_binary``, this installs when *sd-server specifically* is
    missing -- even if an ``sd-cli`` from an older install is already present -- so an
    existing one-shot install is upgraded to the persistent server (the prebuilt archive
    ships both). Returns None when it is absent and cannot be installed; the backend then
    uses the one-shot fallback. Never raises.
    """
    found = find_sd_server_binary()
    if found and _usable_or_discard_managed(found):
        return found
    if not allow_install:
        return found
    with _install_lock:
        found = find_sd_server_binary()
        if found and _usable_or_discard_managed(found):
            return found
        try:
            import sys

            studio_dir = Path(__file__).resolve().parents[3]  # .../studio
            if str(studio_dir) not in sys.path:
                sys.path.insert(0, str(studio_dir))
            from install_sd_cpp_prebuilt import install as _install
        except Exception as exc:  # noqa: BLE001 -- import path / module issues are non-fatal
            logger.warning("sd-server installer import failed: %s", exc)
            return None
        try:
            _install(accelerator = accelerator)  # extracts sd-cli AND sd-server
        except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
            logger.warning("sd-server auto-install failed: %s", exc)
            return None
        return find_sd_server_binary()


@dataclass(frozen = True)
class _SdState:
    """The loaded native checkpoint: resolved asset paths + run settings.

    ``server`` is the resident ``sd-server`` process (the model is loaded once, inside
    it) when ``mode == "server"``; in the ``"oneshot"`` fallback it is ``None`` and each
    generation re-runs ``sd-cli``."""

    repo_id: str
    base_repo: str
    family: DiffusionFamily
    device: str
    files: SdCppModelFiles
    vae_format: Optional[str] = None
    native_speed: str = "off"
    offload_flags: tuple[str, ...] = ()
    threads: Optional[int] = None
    sampling_method: Optional[str] = None
    flow_shift: Optional[float] = None
    server: Optional[SdCppServer] = None
    mode: str = "server"
    # Token kept so LoRA adapters selected at generate time can be fetched from the Hub.
    hf_token: Optional[str] = None
    # The GGUF basename this load committed: some variants pick their encoder by filename, and a local *klein-9B*.gguf carries that keyword only in the basename.
    gguf_filename: Optional[str] = None


def _memory_policy(memory_mode: Optional[str], cpu_offload: bool) -> str:
    """Map the diffusers memory knobs onto an sd-cli offload policy. Only meaningful
    off-CPU (forced sd_cpp / MPS); on CPU everything is resident in RAM anyway."""
    mode = (memory_mode or "").strip().lower()
    if mode == "low_vram":
        return OFFLOAD_SEQUENTIAL
    if mode == "balanced":
        return OFFLOAD_GROUP
    if cpu_offload and mode in ("", "auto"):
        return OFFLOAD_MODEL
    return OFFLOAD_NONE


def _native_speed_for(speed_mode: Optional[str]) -> str:
    mode = (speed_mode or "off").strip().lower()
    return mode if mode in ("default", "max") else "off"


@dataclass
class _SdLoading:
    """An in-flight asset download, polled for progress."""

    repo_id: str
    base_repo: str
    # Companion asset repos (VAE / text encoders) so the delete-cached guard protects them.
    asset_repos: tuple[str, ...] = ()
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
    if fam.name in ("flux.1", "flux.2-klein", "flux.2-dev"):
        return None, (float(guidance) if guidance is not None else None)
    cfg = float(guidance) if (guidance is not None and guidance > 1.0) else 1.0
    return cfg, None


def _fetch_repo_map(assets: list[tuple[str, str, str]], hf_token: Optional[str]) -> dict[str, str]:
    """upstream asset repo -> the repo to actually fetch from (its ungated mirror, or itself).

    Decided per REPO over its whole file list, the same input ``download_plan`` uses, so staging
    and the load agree even when a repo carries several assets.

    Two swaps, in order: a GATED vendor base goes to its ungated mirror, then a mirror whose
    community repack is already cached goes back to the repack. The second only ever spares an
    existing install a re-download of bytes it already holds under the old repo key; a fresh one
    still pulls the mirror."""
    by_repo: dict[str, list[str]] = {}
    for repo, filename, _kind in assets:
        by_repo.setdefault(repo, []).append(filename)
    return {
        repo: prefer_cached_legacy_source(prefer_ungated_mirror(repo, hf_token, files = names), names)
        for repo, names in by_repo.items()
    }


def _with_mirrors(repo_ids) -> tuple[str, ...]:
    """``repo_ids`` plus the ungated mirror and the community repack of each, de-duplicated, order
    preserved.

    The delete-cached guard must protect whichever of the set the bytes landed in, and that
    decision is re-taken per load; naming all of them is cheap and cannot under-protect."""
    out: list[str] = []
    for rid in repo_ids:
        if not rid:
            continue
        out.append(rid)
        mirror = mirror_repo(rid)
        if mirror:
            out.append(mirror)
        legacy = legacy_source_repo(rid)
        if legacy:
            out.append(legacy)
    return tuple(dict.fromkeys(out))


class SdCppDiffusionBackend:
    """Native sd.cpp backend with the diffusers ``DiffusionBackend`` method surface."""

    def __init__(self, engine: Optional[SdCppEngine] = None) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._engine = engine  # resolved lazily on first load so import stays cheap
        # An injected engine (test seam) pins one-shot mode; a fallback-cached engine must NOT, so a now-available server can still be used next load.
        self._engine_injected = engine is not None
        self._state: Optional[_SdState] = None
        self._loading: Optional[_SdLoading] = None
        self._load_token = 0
        # Replaced (never cleared) per load, so a cancelled asset pull stays cancelled.
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        # sd-server started for an in-flight load, before it commits to _state; tracked so an unload can stop it mid-startup.
        self._pending_server: Optional[SdCppServer] = None
        self._gen: Optional[_SdGen] = None
        # Set once this load's graph proved unrunnable on the GPU backend (a ggml unsupported-op abort), so the CPU restart happens once per load. Cleared by each load.
        self._cpu_backend_forced = False

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

    def _resolve_backend(self) -> tuple[str, Optional[str], Optional[SdCppEngine]]:
        """Pick the native execution mode: ("server", binary, None) or ("oneshot", None, engine).

        The persistent ``sd-server`` is preferred (load once, serve many). The one-shot
        ``sd-cli`` is the fallback for older / custom builds that lack the server target.
        An explicitly injected engine forces one-shot (the unit-test seam and an escape
        hatch), so a test never spawns a real server or triggers an install. A lazily
        cached fallback engine does NOT force one-shot: once a resident server becomes
        available (installed, or a per-model start that previously failed now works), the
        next load can use it, instead of being pinned to one-shot for the whole session.
        """
        if self._engine_injected and self._engine is not None:
            return "oneshot", None, self._resolve_engine()
        # Install the server build matching the resolved backend (ROCm/Vulkan/CUDA), not the default CPU build. Lazy import avoids an import cycle.
        from core.inference.diffusion_engine_router import _install_accelerator_for

        accelerator = _install_accelerator_for(
            getattr(resolve_diffusion_device_target(), "backend", "cpu")
        )
        server_binary = ensure_sd_server_binary(
            allow_install = _install_allowed(), accelerator = accelerator
        )
        if server_binary is not None:
            return "server", server_binary, None
        logger.warning(
            "sd-server not found; falling back to one-shot sd-cli (reloads the model per image)."
        )
        return "oneshot", None, self._resolve_engine()

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
        # diffusers-only knobs accepted for a uniform call and ignored (sd.cpp has no torchao quant / SDPA dispatcher / fbcache).
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        # Accepted for interface parity; native is GGUF-only (router forces diffusers otherwise).
        model_kind: Optional[str] = None,
        # Parity with the diffusers load-time LoRA bake; native applies LoRA per generation, so a load-time selection is ignored.
        loras: Optional[list[tuple[str, float]]] = None,
    ) -> dict[str, Any]:
        """Validate, then fetch assets on a daemon thread. Returns at once."""
        # Empty/whitespace token = "no token"; "" verbatim breaks the anonymous fallback.
        hf_token = hf_token.strip() if hf_token and hf_token.strip() else None
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the native engine loads single-file GGUF checkpoints only."
            )
        # Filename-fallback detector (as the route validated) so a local .gguf whose family keyword lives only in the basename still loads.
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name."
            )
        if not family_sd_cpp_supported(fam):
            raise ValueError(f"Family '{fam.name}' has no native sd.cpp asset mapping.")

        base = resolve_base_repo(fam, base_repo)
        with self._lock:
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            # A superseding load must stop any in-flight generation, else the old run can still persist an image after the new load starts.
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._load_token += 1
            token = self._load_token
            # A NEW event per load, never a clear() of the shared one: unload() sets the event the running worker holds but also
            # drops _loading, so a clear() here would un-cancel its still-running multi-gigabyte pull.
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._loading = _SdLoading(
                repo_id = repo_id,
                base_repo = base,
                asset_repos = tuple(
                    dict.fromkeys(
                        r
                        for r, _f, kind in self._asset_specs(repo_id, gguf_filename, fam)
                        if kind != "diffusion_model"
                    )
                ),
            )

        threading.Thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                base = base,
                fam = fam,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                _load_token = token,
                _cancel_event = cancel_event,
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
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        _load_token: int,
        _cancel_event: Optional[threading.Event] = None,
    ) -> None:
        # This load's own event: a later load replaces self._cancel_event rather than clearing it.
        cancel_event = _cancel_event if _cancel_event is not None else self._cancel_event
        try:
            # Resolve mode (server preferred, one-shot fallback) + binary up front so an install failure surfaces before the multi-GB pull.
            mode, server_binary, engine = self._resolve_backend()
            if mode == "server":
                # Probe the server binary before the pull: a present-but-unrunnable build would download everything then fail.
                assert server_binary is not None
                if not _server_binary_runnable(server_binary):
                    logger.warning(
                        "sd-server at %s is present but not runnable; trying one-shot sd-cli.",
                        server_binary,
                    )
                    try:
                        usable = self._resolve_engine().version() is not None
                    except Exception:  # noqa: BLE001
                        usable = False
                    if not usable:
                        raise RuntimeError("sd-server binary is present but not runnable.")
                    mode, server_binary, engine = "oneshot", None, self._resolve_engine()
            if mode == "oneshot":
                # version() is None when a present binary can't run; fail now, not on the first generation.
                assert engine is not None
                if engine.version() is None:
                    raise RuntimeError("sd-cli binary is present but not runnable.")

            # Swap ONCE so the size probe and the download agree: sizes come from paths-info, which
            # -- unlike model_info -- 401s anonymously on a gated repo, so probing the upstream
            # drops the VAE from the progress total the mirror then pulls.
            specs = self._asset_specs(repo_id, gguf_filename, fam)
            fetch_repo = _fetch_repo_map(specs, hf_token)
            assets = [(fetch_repo[repo], fn, kind) for repo, fn, kind in specs]
            # Same preflight the plan runs, on POST-swap repos: catch a gated companion here, not
            # 15 GiB into the prefetch, without refusing one an ungated mirror stands in for. The
            # plan alone is not enough: the images page falls back to this load when it fails.
            self._preflight_companion_repos(
                self._assets_by_repo(assets), fetch_repo.get(repo_id, repo_id), hf_token
            )
            self._set_expected_bytes(assets, hf_token)
            paths = self._fetch_assets(assets, hf_token, cancel_event = cancel_event)

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
            # Honor speed everywhere; offload only off-CPU (on CPU weights are resident, so the flags are no-ops).
            offload: tuple[str, ...] = ()
            if device != "cpu":
                offload = tuple(offload_flags(_memory_policy(memory_mode, cpu_offload)))
            native_speed = _native_speed_for(speed_mode)

            # Tear down the old model then commit the new one under _generate_lock: abort and WAIT for a generation started during
            # the download, so no stale run persists an image. Taken only now, so the download never serialises generation.
            with self._lock:
                if self._load_token != _load_token:
                    return  # superseded / cancelled
                if self._active_generate_cancel is not None:
                    self._active_generate_cancel.set()
            with self._generate_lock:
                with self._lock:
                    if self._load_token != _load_token:
                        return  # superseded / cancelled while waiting
                    old_state = self._state
                    self._state = None  # the old model is being torn down
                if old_state is not None and old_state.server is not None:
                    old_state.server.stop()
                # A new checkpoint earns a fresh attempt on the GPU backend: the previous abort says nothing about this graph.
                self._cpu_backend_forced = False
                server: Optional[SdCppServer] = None
                if mode == "server":
                    assert server_binary is not None
                    server = SdCppServer(server_binary)
                    # Publish the uncommitted server so unload() / a superseding load can stop it mid-startup instead of waiting out the timeout.
                    with self._lock:
                        self._pending_server = server
                    try:
                        # Blocks until the model is loaded and answering; raises with the log tail on failure.
                        server.start(
                            files,
                            vae_format = fam.sd_cpp_vae_format,
                            offload = list(offload),
                            native_speed = native_speed,
                            # Pin to physical cores (sd.cpp's default oversubscribes; see _default_threads).
                            threads = _default_threads(),
                        )
                    except SdCppCancelled:
                        # Aborted by unload / superseding load: stop the half-started server and bail.
                        server.stop()
                        raise
                    except Exception as start_exc:  # noqa: BLE001
                        # Fall back to one-shot sd-cli if usable, else surface the server error.
                        logger.warning(
                            "sd-server failed to start (%s); falling back to one-shot sd-cli.",
                            start_exc,
                        )
                        server.stop()
                        server = None
                        try:
                            usable = self._resolve_engine().version() is not None
                        except Exception:  # noqa: BLE001
                            usable = False
                        if not usable:
                            raise start_exc
                        mode = "oneshot"
                    finally:
                        with self._lock:
                            if self._pending_server is server:
                                self._pending_server = None
                state = _SdState(
                    repo_id = repo_id,
                    base_repo = base,
                    family = fam,
                    device = device,
                    files = files,
                    vae_format = fam.sd_cpp_vae_format,
                    native_speed = native_speed,
                    offload_flags = offload,
                    # One-shot sd-cli reads this per generation; pin to physical cores.
                    threads = _default_threads(),
                    sampling_method = fam.sd_cpp_sampling_method,
                    flow_shift = fam.sd_cpp_flow_shift,
                    server = server,
                    mode = mode,
                    hf_token = hf_token,
                    gguf_filename = gguf_filename,
                )
                with self._lock:
                    if self._load_token != _load_token:
                        # Superseded / unloaded while loading: discard the started server so it doesn't leak.
                        if server is not None:
                            server.stop()
                        return
                    self._state = state
                    self._loading = None
        except SdCppCancelled:
            return
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            if self._load_token != _load_token:
                return
            logger.error("sd_cpp.load_failed: %s", exc)
            # Redact filesystem paths before this reaches /images/load-progress (as diffusers does).
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == _load_token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    def download_plan(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        hf_token: Optional[str] = None,
        **_load_kwargs: Any,
    ) -> dict[str, Any]:
        """The repos + exact files a NATIVE load of this pick needs, in the same envelope the
        diffusers backend returns, so the Hub download manager stages what sd-cli will actually
        open.

        The two engines want different files: diffusers builds a pipeline around the base repo's
        sharded components, while sd-cli reads the single-file VAE + text encoders declared in
        ``diffusion_families``. Planning with the wrong engine stages tens of GB the load never
        opens and then pulls the native assets inline, outside the manager's progress and disk
        preflight -- so the route asks whichever engine it predicts the load will select.

        The diffusers-only kwargs (quant / memory / LoRA) are accepted and ignored, exactly as
        ``begin_load`` accepts them: nothing sd-cli fetches depends on them."""
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the native engine loads single-file GGUF checkpoints only."
            )
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None or not family_sd_cpp_supported(fam):
            # Unreachable through the route, but a direct caller gets the same message begin_load would raise.
            raise ValueError(f"'{repo_id}' has no native sd.cpp asset mapping.")

        specs = self._asset_specs(repo_id, gguf_filename, fam)
        by_repo = self._assets_by_repo(specs)

        # STAGED before the load runs, so each entry must name the repo _fetch_assets will pull
        # from: some asset repos are gated (the FLUX.1 VAE lives in black-forest-labs/FLUX.1-schnell)
        # and an anonymous user would 401 at staging, never reaching the swap. Same per-repo file
        # list on both sides, so both take the same decision.
        fetch_repo = _fetch_repo_map(specs, hf_token)
        by_repo = {fetch_repo[repo]: names for repo, names in by_repo.items()}
        fetch_repo_id = fetch_repo.get(repo_id, repo_id)
        # AFTER the swap: preflighting the upstream id would refuse the very picks the ungated
        # mirror exists to rescue.
        self._preflight_companion_repos(by_repo, fetch_repo_id, hf_token)
        sizes = self._plan_file_sizes(by_repo, hf_token)
        entries: list[dict[str, Any]] = []
        total = 0
        for repo, names in by_repo.items():
            repo_bytes = int(sum(sizes.get((repo, n), 0) for n in names))
            total += repo_bytes
            entries.append(
                {
                    "repo_id": repo,
                    "files": names,
                    "bytes": repo_bytes,
                    # Only the transformer entry carries the GGUF filename; the VAE / encoder entries are plain single files.
                    "gguf_filename": gguf_filename if repo == fetch_repo_id else None,
                }
            )
        return {"entries": entries, "total_bytes": total}

    @staticmethod
    def _assets_by_repo(specs: list[tuple[str, str, str]]) -> dict[str, list[str]]:
        """repo -> the files this pick needs from it, first-seen order (transformer first), so a
        family whose VAE and text encoder share a repo yields one entry."""
        by_repo: dict[str, list[str]] = {}
        for repo, filename, kind in specs:
            # A local GGUF directory is already on disk; nothing to stage or preflight for it.
            if kind == "diffusion_model":
                try:
                    if Path(repo).expanduser().exists():
                        continue
                except (OSError, RuntimeError, ValueError):
                    pass  # unresolvable home / invalid path characters -> a remote id
            names = by_repo.setdefault(repo, [])
            if filename not in names:
                names.append(filename)
        return by_repo

    @staticmethod
    def _preflight_companion_repos(
        by_repo: dict[str, list[str]], repo_id: str, hf_token: Optional[str]
    ) -> None:
        """Refuse a companion repo this pick cannot read, before any byte is fetched.

        The native asset list carries its own companion repos (flux.1's VAE is the gated
        black-forest-labs/FLUX.1-schnell), and neither ``_plan_file_sizes`` nor the size probe
        surfaces the 401: the entry is planned at 0 bytes and the fetch dies on the bare Hub token
        error this replaces. Run from BOTH the plan and ``_run_load``, as the diffusers backend
        does, because the UI falls back to /images/load when the plan call fails."""
        from core.inference.diffusion import _assert_base_repo_accessible
        for repo, names in by_repo.items():
            # Companions only: the picker only lists repos it could already read.
            if repo != repo_id and names:
                # Probe an asset THIS pick stages: a VAE-only repo has no pipeline manifest, so the
                # default name would neither verify access nor see the cache.
                _assert_base_repo_accessible(repo, hf_token, names[0])

    def preflight_base_access(
        self,
        repo_id: str,
        fam: Optional[DiffusionFamily],
        *,
        gguf_filename: Optional[str] = None,
        model_kind: Optional[str] = None,
        base_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        """The companion refusal ``_run_load`` makes, run by the route BEFORE it takes the GPU.

        Same signature and reason as the diffusers backend's: ``_run_load`` runs on the load thread,
        after a forced-native load on a GPU host already evicted chat, so a pick refused only there
        unloads the resident model first. Nothing to check without a family or checkpoint name."""
        if fam is None or not gguf_filename:
            return
        # Post-swap, as the plan and the load are: the swap is pure, so all three decide alike.
        specs = self._asset_specs(repo_id, gguf_filename, fam)
        fetch_repo = _fetch_repo_map(specs, hf_token)
        self._preflight_companion_repos(
            self._assets_by_repo([(fetch_repo[r], fn, kind) for r, fn, kind in specs]),
            fetch_repo.get(repo_id, repo_id),
            hf_token,
        )

    @staticmethod
    def _plan_file_sizes(
        by_repo: dict[str, list[str]], hf_token: Optional[str]
    ) -> dict[tuple[str, str], int]:
        """(repo, filename) -> size in bytes, best-effort (0 for anything the Hub won't answer).

        A missing size only understates the manager's progress total; it must not fail the plan,
        which is the cheap pre-flight for a load that would otherwise download inline."""
        out: dict[tuple[str, str], int] = {}
        try:
            from huggingface_hub import HfApi
            api = HfApi(token = hf_token)
        except Exception:  # noqa: BLE001 -- sizes are best-effort
            return out
        for repo, names in by_repo.items():
            try:
                for info in api.get_paths_info(repo, paths = names, expand = False):
                    out[(repo, getattr(info, "path", ""))] = int(getattr(info, "size", 0) or 0)
            except Exception:  # noqa: BLE001 -- one unreadable repo is non-fatal
                continue
        return out

    def _asset_specs(
        self, repo_id: str, gguf_filename: str, fam: DiffusionFamily
    ) -> list[tuple[str, str, str]]:
        """(repo, filename, kind) for every file sd-cli needs. ``kind`` is the
        SdCppModelFiles field; the transformer reuses the diffusers GGUF."""
        specs: list[tuple[str, str, str]] = [(repo_id, gguf_filename, "diffusion_model")]
        if fam.sd_cpp_vae:
            specs.append((fam.sd_cpp_vae[0], fam.sd_cpp_vae[1], "vae"))
        # Pick the encoder per variant from the load identity so a 9B GGUF fetches the right one.
        for terepo, tefile, kind in sd_cpp_text_encoders_for(fam, repo_id, gguf_filename):
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
            for repo, fn, kind in assets:
                # Only the transformer can be a local path; others are always HF ids.
                if kind == "diffusion_model" and Path(repo).expanduser().exists():
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
        self,
        assets: list[tuple[str, str, str]],
        hf_token: Optional[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, str]:
        """Download every asset (cancellable via this load's own ``cancel_event``, so
        a replacement load cannot un-cancel this pull), returning kind -> local path."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # Callers without a per-load event (tests, direct use) fall back to the current one.
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        paths: dict[str, str] = {}
        # This backend fetches the ASSET repos, never the base, and some are gated, so the swap goes
        # here. Decided per REPO over its whole file list, exactly as download_plan does, so the
        # load pulls from the repo the manager already staged.
        fetch_repo = _fetch_repo_map(assets, hf_token)
        assets = [(fetch_repo[repo], fn, kind) for repo, fn, kind in assets]
        for repo, fn, kind in assets:
            if cancel.is_set():
                raise SdCppCancelled("load cancelled")
            local_root = Path(repo).expanduser()
            if kind == "diffusion_model" and local_root.exists():
                path = str(resolve_local_gguf_child(local_root, fn))
            else:
                # Resolve an asset cached only under huggingface_hub's import-time root through
                # that root, as the preflight does. Pinned to the live root, a cache-folder change
                # re-downloads every moved asset and 401s on an already-downloaded gated base.
                path = hf_hub_download_with_xet_fallback(
                    repo, fn, hf_token, cancel_event = cancel, reuse_other_cache_root = True
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

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repo ids an in-flight background load is downloading (empty when idle).
        Mirrors the diffusers backend so the delete-cached guard can query whichever
        engine is active without caring which one it got. Includes the companion
        VAE / text-encoder repos: deleting one of those mid-load would remove files
        the committed SdCppModelFiles paths need, and the mirror of each, where those bytes
        land once a gated asset repo is swapped out."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            ids = (loading.repo_id, loading.base_repo, *loading.asset_repos)
            return _with_mirrors(ids)

    def loaded_repo_ids(self) -> tuple[str, ...]:
        """Repo ids the COMMITTED native model reads from disk (empty when unloaded).

        The one-shot sd-cli re-reads the companion VAE / text-encoder files from the HF
        cache on every generation (server mode keeps them in the resident process, but the
        extra ids are harmless there), so the delete-cached guard must refuse those
        companion repos while the model is loaded -- status().repo_id covers only the main
        GGUF. Reconstructed from the committed family, mirroring loading_repo_ids(), and
        carrying the mirrors too: one-shot sd-cli re-reads whichever of the pair was fetched."""
        with self._lock:
            state = self._state
            if state is None:
                return ()
            fam = state.family
            repos = [state.repo_id, state.base_repo]
            if fam.sd_cpp_vae:
                repos.append(fam.sd_cpp_vae[0])
            # Same per-variant selection as _asset_specs (keyed on repo id AND GGUF filename) so the delete guard protects the encoder repo this load downloaded.
            repos.extend(
                terepo
                for terepo, _f, _k in sd_cpp_text_encoders_for(
                    fam, state.repo_id, state.gguf_filename
                )
            )
            return _with_mirrors(repos)

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
        # Batched prompt/seed lists are diffusers-engine features; accepted for parity and rejected below, since sd-cli would render them serially.
        prompts: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        # Accepted for interface parity; native is text-to-image only, so image-conditioned requests are rejected below.
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        upscale: Optional[float] = None,  # needs an init image; rejected by the guard below
        reference_images: Optional[list[str]] = None,  # GPU/diffusers-only (FLUX.2)
        # LoRA (id, weight) pairs; resolved up front then applied per path: prompt tags for one-shot sd-cli, structured `lora` for sd-server.
        loras: Optional[list[tuple[str, float]]] = None,
        # ControlNet is diffusers-only; rejected by the guard below (accepted for parity).
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
    ) -> dict[str, Any]:
        import tempfile

        from PIL import Image

        from core.inference import diffusion_lora

        if (
            init_image is not None
            or mask_image is not None
            or reference_images
            or (upscale is not None and upscale > 1)
        ):
            raise ValueError(
                "img2img / inpaint / reference / upscale are not yet supported on the native "
                "sd.cpp engine; run on a GPU (diffusers) for image-conditioned workflows."
            )
        if prompts is not None or seeds is not None:
            raise ValueError(
                "Batched prompt/seed lists are not supported on the native sd.cpp engine "
                "(it renders serially); run on a GPU (diffusers) for batched generation, "
                "or use batch_size for a serial native batch."
            )
        # strength 0/None disables ControlNet (matches diffusers), so no-op it rather than 400.
        if controlnet is not None and controlnet[3] in (None, 0, 0.0):
            controlnet = None
        if controlnet is not None:
            raise ValueError(
                "ControlNet is not yet supported on the native sd.cpp engine; run on a GPU "
                "(diffusers) for ControlNet conditioning."
            )

        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                # A resident server can exit while idle; drop stale state and report not-loaded so the client gets the reload path, not a 500.
                if (
                    state.mode == "server"
                    and state.server is not None
                    and not state.server.is_alive()
                ):
                    self._state = None
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                self._active_generate_cancel = cancel
                # Publish an active (step 0) state before the slow pre-generate setup so a reload probe does not read idle while this holds _generate_lock.
                self._gen = _SdGen(total_steps = int(steps))
            try:
                if seed is None:
                    seed = int.from_bytes(os.urandom(6), "big") & ((1 << 53) - 1)
                else:
                    seed = int(seed)
                cfg_scale, flux_guidance = _map_guidance(state.family, guidance)
                # Resolve selected LoRAs up front (a bad id gives a clear 400). Drop weight-0 rows BEFORE the support gate so an only-disabled request stays a no-op.
                lora_resolved: list = []
                active_loras = [(i, w) for (i, w) in (loras or []) if w != 0]
                if active_loras:
                    if not diffusion_lora.supports_lora(
                        engine = "sd_cpp",
                        family = state.family.name,
                        model_kind = "gguf",
                        transformer_quant = None,
                    ):
                        raise ValueError(
                            f"LoRA is not supported for {state.family.name} on the native "
                            "sd.cpp engine."
                        )
                    lora_resolved = diffusion_lora.resolve_specs(
                        active_loras,
                        family = state.family.name,
                        hf_token = state.hf_token,
                        cancel_event = cancel,
                    )
                if state.mode == "server" and state.server is not None:
                    images, seeds = self._generate_server(
                        state,
                        prompt = prompt,
                        negative_prompt = negative_prompt,
                        width = width,
                        height = height,
                        steps = steps,
                        seed = seed,
                        batch_size = batch_size,
                        cfg_scale = cfg_scale,
                        flux_guidance = flux_guidance,
                        lora_resolved = lora_resolved,
                        cancel = cancel,
                    )
                else:
                    images, seeds = self._generate_oneshot(
                        state,
                        prompt = prompt,
                        negative_prompt = negative_prompt,
                        width = width,
                        height = height,
                        steps = steps,
                        seed = seed,
                        batch_size = batch_size,
                        cfg_scale = cfg_scale,
                        flux_guidance = flux_guidance,
                        lora_resolved = lora_resolved,
                        cancel = cancel,
                    )
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # ``seeds`` is the per-image seed (image i used seed+i) for the route to persist.
                return {
                    "images": images,
                    "seed": int(seed),
                    "seeds": seeds,
                    "repo_id": state.repo_id,
                    # The BUILD, for the recipe: the repo id alone does not say WHICH GGUF quant ran, and two quants make different pixels.
                    "model_kind": "gguf",
                    "gguf_filename": state.gguf_filename,
                    # The rest of the build the recipe records. The native engine has no dense
                    # quant path and no memory-mode planner, so those two are honestly null -- but
                    # the offload it ran under is real (sd-cli flags) and status() already derives
                    # it the same way. Omitting them here persisted null for every native image and
                    # left the recipe unable to say how the picture was produced.
                    "transformer_quant": None,
                    "text_encoder_quant": None,
                    "memory_mode": None,
                    "offload_policy": "active" if state.offload_flags else "none",
                }
            except SdCppCancelled as exc:
                raise RuntimeError(DIFFUSION_CANCELLED_MSG) from exc
            finally:
                self._gen = None
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None

    def _generate_server(
        self,
        state: _SdState,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        steps: int,
        seed: int,
        batch_size: int,
        cfg_scale: Optional[float],
        flux_guidance: Optional[float],
        lora_resolved: list,
        cancel: threading.Event,
    ) -> tuple[list, list[int]]:
        """Generate via the resident sd-server (no model reload).

        A batch larger than the server's per-job limit is split into chunks: the server
        rejects a batch_count above _MAX_SERVER_BATCH, and the one-shot path served large
        batches image-by-image, so preserve that. The base seed is masked to sd.cpp's
        signed-int64 range (the request model / diffusers accept larger seeds), and each
        chunk is submitted at base+offset so the per-image seeds stay reproducible. Each
        chunk gets a timeout proportional to its image count so a slow CPU batch is not
        cancelled partway through on one fixed deadline.

        LoRA on the server goes through the structured ``lora`` request field, NOT prompt
        tags (the sdcpp API intentionally ignores ``<lora:>`` in the prompt). Selected
        adapters are staged into the server's ``--lora-model-dir`` scratch dir, which the
        server rescans per request, and referenced by their staged filename."""
        import io
        import os
        import shutil

        from PIL import Image

        from core.inference import diffusion_lora

        assert state.server is not None
        total = max(1, int(batch_size))
        # sd.cpp's image seed is signed int64; mask base and derived seeds to that range.
        base_seed = int(seed) & ((1 << 63) - 1)
        images: list = []
        seeds: list[int] = []
        # Stage LoRAs into a per-request subdir of the server lora-model-dir (so a prior request's adapters cannot leak in); removed after.
        lora_payload: Optional[list[dict]] = None
        lora_stage: Optional[Path] = None
        if lora_resolved:
            server_lora_dir = state.server.lora_dir
            if server_lora_dir:
                lora_stage = Path(server_lora_dir) / f"gen_{os.urandom(6).hex()}"
                materialized = diffusion_lora.materialize_native_dir(lora_resolved, lora_stage)
                lora_payload = [
                    {
                        "path": f"{lora_stage.name}/{Path(m.path).name}",
                        "multiplier": float(m.weight),
                    }
                    for m in materialized
                ]
        # One deadline for the whole request, shared by its chunks: a batch is chunked only because the server caps images per
        # job, so each chunk gets whatever is left rather than its own full budget and a long batch still ends on time.
        deadline = time.monotonic() + NATIVE_GENERATION_TIMEOUT_S
        try:
            for offset in range(0, total, _MAX_SERVER_BATCH):
                if cancel.is_set():
                    raise SdCppCancelled("sd-server generation was cancelled.")
                count = min(_MAX_SERVER_BATCH, total - offset)
                chunk_seed = (base_seed + offset) & ((1 << 63) - 1)
                payload = build_img_gen_request(
                    prompt = prompt,
                    negative_prompt = negative_prompt or None,
                    width = int(width),
                    height = int(height),
                    steps = int(steps),
                    seed = chunk_seed,
                    batch_count = count,
                    sample_method = state.sampling_method,
                    flow_shift = state.flow_shift,
                    cfg_scale = cfg_scale,
                    distilled_guidance = flux_guidance,
                    lora = lora_payload,
                )
                try:
                    blobs = state.server.img_gen(
                        payload,
                        on_step = self._on_log,
                        cancel_event = cancel,
                        total_timeout = max(deadline - time.monotonic(), 1.0),
                    )
                except RuntimeError as exc:
                    # A ggml unsupported-op abort killed the server: this graph cannot run on the GPU backend at all, so restart the model on the CPU backend once and retry this chunk. Any other death propagates.
                    server = self._restart_server_on_cpu_backend(state, str(exc), cancel)
                    if server is None:
                        raise
                    state = replace(state, server = server)
                    with self._lock:
                        if self._state is not None and self._state.server is not None:
                            self._state = state
                    blobs = server.img_gen(
                        payload,
                        on_step = self._on_log,
                        cancel_event = cancel,
                        total_timeout = max(deadline - time.monotonic(), 1.0),
                    )
                # All-or-nothing per chunk: fail rather than silently drop images from the batch.
                if not cancel.is_set() and len(blobs) != count:
                    raise RuntimeError(
                        f"sd-server returned {len(blobs)} of {count} requested images in the batch."
                    )
                images.extend(Image.open(io.BytesIO(b)).convert("RGB") for b in blobs)
                # sd.cpp advances the seed per image within a job, so report chunk_seed+i.
                seeds.extend((chunk_seed + i) & ((1 << 63) - 1) for i in range(len(blobs)))
        finally:
            if lora_stage is not None:
                shutil.rmtree(lora_stage, ignore_errors = True)
        return images, seeds

    def _restart_server_on_cpu_backend(
        self, state: _SdState, error_text: str, cancel: threading.Event
    ) -> Optional[SdCppServer]:
        """Relaunch this checkpoint's sd-server with ``--backend cpu``; None if that does not apply.

        ggml's Metal backend checks every node against ``ggml_metal_device_supports_op`` and calls
        ``GGML_ABORT`` when one is not implemented for that device, because a single-backend graph
        has nowhere else to put the node -- there is no per-op CPU fallback. The whole sd-server
        dies with SIGABRT mid-generation, so the user sees "the native image renderer stopped
        unexpectedly" with no way forward. Observed on macos-14 arm64 with FLUX.2-klein-4B Q2_K:
        the encoder is already pinned to CPU (see metal_text_encoder_flags), and the abort then
        moves into the denoise loop:

            ggml_metal_op_encode_impl: error: unsupported op 'MUL_MAT' -> ggml_abort
            StableDiffusionGGML::sample -> sample_k_diffusion

        ``--backend cpu`` is the only flag that changes which backend EXECUTES the graph
        (``--offload-to-cpu`` moves parameters, not compute), so the restart runs the same
        checkpoint slower rather than not at all. Done once per load: the second abort, or any
        other cause of death, is surfaced to the caller."""
        if not is_ggml_unsupported_op_abort(error_text):
            return None
        if self._cpu_backend_forced or state.device == "cpu":
            return None  # already on CPU: the abort is not a backend-placement problem
        if state.server is None or cancel.is_set():
            return None
        server_binary = find_sd_server_binary()
        if not server_binary:
            return None
        logger.warning(
            "sd-server aborted on an op the '%s' backend cannot run; restarting on the CPU "
            "backend (slower, but it completes). Details: %s",
            state.device,
            error_text[:300],
        )
        self._cpu_backend_forced = True
        state.server.stop()
        server = SdCppServer(server_binary)
        with self._lock:
            self._pending_server = server
        try:
            server.start(
                state.files,
                vae_format = state.vae_format,
                offload = list(state.offload_flags),
                native_speed = state.native_speed,
                threads = state.threads,
                extra_args = list(CPU_BACKEND_FLAGS),
            )
        except Exception:  # noqa: BLE001 -- the original abort is the more useful error
            server.stop()
            return None
        finally:
            with self._lock:
                if self._pending_server is server:
                    self._pending_server = None
        return server

    def _generate_oneshot(
        self,
        state: _SdState,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        steps: int,
        seed: int,
        batch_size: int,
        cfg_scale: Optional[float],
        flux_guidance: Optional[float],
        lora_resolved: list,
        cancel: threading.Event,
    ) -> tuple[list, list[int]]:
        """Fallback path: re-run one-shot sd-cli per image (reloads the model each time).

        LoRA on the one-shot path uses sd-cli's own mechanism: materialize the selected
        adapters into a ``--lora-model-dir`` and inject matching ``<lora:ALIAS:w>`` tags
        into the prompt (sd-cli parses and strips them). supports_lora already gated the
        family upstream, so a non-empty ``lora_resolved`` is safe to apply here."""
        import tempfile

        from PIL import Image

        from core.inference import diffusion_lora

        engine = self._resolve_engine()
        extra_args: list[str] = []
        if state.vae_format:
            extra_args += ["--vae-format", state.vae_format]
        if state.flow_shift is not None:
            extra_args += ["--flow-shift", repr(float(state.flow_shift))]

        images = []
        seeds: list[int] = []
        with tempfile.TemporaryDirectory(prefix = "sdcpp_gen_") as tmpdir:
            # Materialize LoRAs into a scan dir and inject <lora:ALIAS:w> tags (deduped). Empty -> unchanged.
            eff_prompt = prompt
            lora_dir: Optional[str] = None
            if lora_resolved:
                materialized = diffusion_lora.materialize_native_dir(
                    lora_resolved, Path(tmpdir) / "loras"
                )
                eff_prompt = diffusion_lora.inject_prompt_tags(prompt, materialized)
                lora_dir = str(Path(tmpdir) / "loras")
            for index in range(max(1, int(batch_size))):
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # Distinct reproducible seed per image; mask to int64 (53 bits would truncate large explicit seeds and collide them).
                seed_i = (seed + index) & ((1 << 63) - 1)
                out_path = str(Path(tmpdir) / f"img_{index}.png")
                params = SdCppGenParams(
                    prompt = eff_prompt,
                    negative_prompt = negative_prompt or None,
                    width = int(width),
                    height = int(height),
                    steps = int(steps),
                    cfg_scale = cfg_scale,
                    guidance = flux_guidance,
                    seed = seed_i,
                    sampling_method = state.sampling_method,
                    batch_count = 1,
                    lora_dir = lora_dir,
                    lora_apply_mode = "auto" if lora_dir else None,
                )
                engine.generate(
                    state.files,
                    params,
                    output_path = out_path,
                    offload = list(state.offload_flags) or None,
                    native_speed = state.native_speed,
                    threads = state.threads,
                    extra_args = extra_args or None,
                    on_log = self._on_log,
                    cancel_event = cancel,
                )
                with Image.open(out_path) as im:
                    images.append(im.copy())
                seeds.append(seed_i)
        return images, seeds

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
        with self._lock:
            # Under the lock: begin_load rebinds this attribute, so an unlocked read could set an event the current load no longer watches.
            self._cancel_event.set()
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            state = self._state
            self._state = None
            self._load_token += 1
            self._loading = None
            # Grab a mid-start() uncommitted server too so we can stop it (startup is abortable).
            pending = self._pending_server
            self._pending_server = None
        # Stop the resident server outside the lock (terminate can take seconds); a mid-flight generation unwinds as the process goes away.
        if state is not None and state.server is not None:
            state.server.stop()
        if pending is not None and pending is not (state.server if state else None):
            pending.stop()
        # Barrier: wait for a signalled one-shot generation to exit before reporting unloaded, since callers treat this return as "device is free".
        with self._generate_lock:
            pass
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self._state
        # A resident sd-server can exit after load (OOM/crash while idle); drop stale state so clients reload instead of 500ing per generation.
        if (
            state is not None
            and state.mode == "server"
            and state.server is not None
            and not state.server.is_alive()
        ):
            logger.warning("sd-server exited after load; clearing loaded state")
            with self._lock:
                if self._state is state:
                    self._state = None
            state = None
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
                "native_mode": None,
                "supports_lora": False,
                "supports_controlnet": False,
                "workflows": [],
            }
        from core.inference import diffusion_lora

        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": "gguf",
            # Reflect the offload flags actually passed to sd-cli (empty on CPU -> "none").
            "cpu_offload": bool(state.offload_flags),
            "offload_policy": "active" if state.offload_flags else "none",
            "vae_tiling": False,
            "memory_mode": None,
            "speed_mode": state.native_speed,
            "speed_optims": [],
            "text_encoder_quant": None,
            "transformer_quant": None,
            "attention_backend": None,
            "transformer_cache": None,
            "engine": "sd_cpp",
            "supports_lora": diffusion_lora.supports_lora(
                engine = "sd_cpp",
                family = state.family.name,
                model_kind = "gguf",
                transformer_quant = None,
            ),
            # ControlNet is diffusers-only; the native engine's generate() rejects it.
            "supports_controlnet": False,
            # "server" = resident sd-server (load once); "oneshot" = legacy per-image sd-cli.
            "native_mode": state.mode,
            # Native supports txt2img only; advertise it so the UI doesn't disable the Create tab.
            "workflows": ["txt2img"],
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
