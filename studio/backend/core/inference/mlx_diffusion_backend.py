# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in MLX (mflux) image engine for Apple Silicon (``UNSLOTH_DIFFUSION_MLX=1``) on Studio's GGUF.

Same duck-typed surface as the other image engines; selected only by ``diffusion_engine_router``.
"""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from core.inference.diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    DiffusionFamily,
    DiffusionModelReplacedError,
    LoadIdentity,
    _is_local_path,
    canonical_base,
    detect_family_for_pick,
    family_mlx_supported,
    load_identity,
    mirror_repo,
    mlx_model_config_for,
    named_variant_base,
    prefer_ungated_mirror,
    resolve_base_repo,
    resolve_local_gguf_child,
    supported_family_names,
)
from core.inference import mlx_diffusion_convert as convert
from core.inference.diffusion_batched import SEED_MASK
from loggers import get_logger
from utils.account_context import account_thread, current_account_id
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory

logger = get_logger(__name__)

ENGINE_NAME = "mlx"

BASE_ALLOW_PATTERNS = (
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "tokenizer/*",
    "vae/*.safetensors",
    "vae/*.json",
)
_LINKED_SUBDIRS = ("text_encoder", "tokenizer", "vae")
_TE_GROUP_SIZE = 64
_ENCODING_CACHE_SIZE = 8
_LOW_RAM_BYTES = 16 * 1024**3


class MlxCancelled(Exception):
    """Raised in the mflux step callback to stop at the next step."""


def clear_exception_locals(exc: Optional[BaseException]) -> None:
    """Drop the locals of every finished frame in an exception chain, keeping the traceback text."""
    import traceback

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


def drop_exception_frames(exc: Optional[BaseException]) -> None:
    """Clear frame locals along an exception chain: a retained traceback otherwise keeps the model alive."""
    import traceback

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None
        nxt = exc.__cause__ or exc.__context__
        exc.__cause__ = None
        exc.__context__ = None
        exc = nxt


def _total_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return 0


def _env_int(name: str, allowed: tuple[int, ...]) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring it", name, raw)
        return None
    if value not in allowed:
        logger.warning("%s=%s is not one of %s; ignoring it", name, value, allowed)
        return None
    return value


def text_encoder_bits(ram_bytes: Optional[int] = None) -> Optional[int]:
    forced = _env_int("UNSLOTH_DIFFUSION_MLX_TE_BITS", (4, 8, 16))
    if forced is not None:
        return None if forced == 16 else forced
    ram = _total_ram_bytes() if ram_bytes is None else ram_bytes
    return 4 if 0 < ram <= _LOW_RAM_BYTES else 8


def dense_bits(ram_bytes: Optional[int] = None) -> int:
    forced = _env_int("UNSLOTH_DIFFUSION_MLX_DENSE_BITS", (0, 4, 8))
    if forced is not None:
        return forced
    ram = _total_ram_bytes() if ram_bytes is None else ram_bytes
    return 4 if 0 < ram <= _LOW_RAM_BYTES else 0


def _chip_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
        return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def compute_precision_name(chip: Optional[str] = None) -> str:
    """M1 / M2 lack native bf16, so fp16 there."""
    raw = os.environ.get("UNSLOTH_DIFFUSION_MLX_PRECISION", "auto").strip().lower()
    if raw in ("fp16", "float16", "half"):
        return "float16"
    if raw in ("bf16", "bfloat16"):
        return "bfloat16"
    name = (chip if chip is not None else _chip_name()).lower()
    return "float16" if ("apple m1" in name or "apple m2" in name) else "bfloat16"


class MfluxRuntime:
    """The only code that imports mlx / mflux. Tests replace it with a fake (``backend.runtime_factory``)."""

    def build(
        self, model_path: str, *, model_config: str, precision: str, te_bits: Optional[int]
    ) -> Any:
        import mlx.core as mx
        import mlx.nn as nn
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image import ZImage

        config = getattr(ModelConfig, model_config.replace("-", "_"))()
        model = ZImage(
            model_config = config,
            model_path = model_path,
            quantize = None,
            compute_precision = mx.float16 if precision == "float16" else None,
        )
        if te_bits:

            def _quantizable(_path, module):
                weight = getattr(module, "weight", None)
                return (
                    hasattr(module, "to_quantized")
                    and weight is not None
                    and weight.shape[-1] % _TE_GROUP_SIZE == 0
                )

            nn.quantize(
                model.text_encoder,
                group_size = _TE_GROUP_SIZE,
                bits = te_bits,
                class_predicate = _quantizable,
            )
        # Materialize in slices so peak memory stays near the quantized size, not the bf16 checkpoint.
        from mlx.utils import tree_flatten

        for component in (model.transformer, model.vae, model.text_encoder):
            arrays = [a for _name, a in tree_flatten(component.parameters())]
            for start in range(0, len(arrays), 32):
                mx.eval(arrays[start : start + 32])
        mx.eval(model.parameters())
        return model

    def generate(
        self,
        model: Any,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        seed: int,
        steps: int,
        width: int,
        height: int,
        guidance: Optional[float],
        on_step: Callable[[int], None],
    ) -> Any:
        import mlx.core as mx
        from mflux.utils.exceptions import StopImageGenerationException

        class _StepCallback:
            def call_in_loop(self, t, seed, prompt, latents, config, time_steps):  # noqa: ARG002
                mx.eval(latents)
                try:
                    on_step(int(t) + 1)
                except MlxCancelled as exc:
                    raise StopImageGenerationException(str(exc)) from exc

        callback = _StepCallback()
        model.callbacks.register(callback)
        cancelled = False
        try:
            result = model.generate_image(
                seed = int(seed),
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = int(steps),
                width = int(width),
                height = int(height),
                guidance = guidance,
            )
        except StopImageGenerationException as exc:
            # The unwound mflux frames hold the model; uncleared, it survives unload().
            drop_exception_frames(exc)
            cancelled = True
        except BaseException as exc:
            # Same for a real failure, but keep the traceback for the caller's log.
            clear_exception_locals(exc)
            raise
        finally:
            try:
                model.callbacks.in_loop.remove(callback)
            except (AttributeError, ValueError):
                pass
        if cancelled:
            raise MlxCancelled("cancelled")
        return result.image

    def cache_text_encodings(
        self,
        model: Any,
        size: int = _ENCODING_CACHE_SIZE,
    ) -> None:
        original = model._encode_prompts
        cache: OrderedDict = OrderedDict()

        def _cached(*, prompt, negative_prompt, guidance):
            cfg = float(guidance or 0.0) > 1.0
            key = (prompt, negative_prompt if cfg else None, cfg)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            value = original(prompt = prompt, negative_prompt = negative_prompt, guidance = guidance)
            cache[key] = value
            while len(cache) > size:
                cache.popitem(last = False)
            return value

        model._encode_prompts = _cached

    def free(self) -> None:
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    def supports_guidance(self, model: Any) -> bool:
        return bool(getattr(getattr(model, "model_config", None), "supports_guidance", False))


@dataclass
class _MlxState:
    repo_id: str
    display_repo_id: Optional[str]
    gguf_filename: str
    base_repo: str
    family: DiffusionFamily
    model: Any
    model_config: str
    precision: str
    te_bits: Optional[int]
    dense_bits: int
    convert_dir: str
    timings: dict = field(default_factory = dict)


@dataclass
class _MlxLoading:
    repo_id: str
    base_repo: str
    fetch_repo: Optional[str] = None
    phase: str = "downloading"
    downloaded_bytes: int = 0
    expected_bytes: int = 0
    fraction: float = 0.0
    error: Optional[str] = None


@dataclass
class _MlxGen:
    total_steps: int
    step: int = 0
    first_step_at: float = 0.0
    eta_seconds: Optional[float] = None


def _snapshot_revision(path: str) -> Optional[str]:
    parts = Path(path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def fetch_gguf(
    repo_id: str,
    gguf_filename: str,
    hf_token: Optional[str],
    *,
    cancel_event: threading.Event,
    local_files_only: bool,
) -> str:
    local_root = Path(repo_id).expanduser()
    if local_root.exists():
        return str(resolve_local_gguf_child(local_root, gguf_filename))
    from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

    return hf_hub_download_with_xet_fallback(
        repo_id,
        gguf_filename,
        hf_token,
        cancel_event = cancel_event,
        reuse_other_cache_root = True,
        local_files_only = local_files_only,
    )


def _holds_base_components(snapshot: str) -> bool:
    root = Path(snapshot)
    if not all((root / sub).is_dir() for sub in _LINKED_SUBDIRS):
        return False
    if not all(
        (root / "tokenizer" / f).is_file() for f in ("tokenizer.json", "tokenizer_config.json")
    ):
        return False
    for sub in ("text_encoder", "vae"):
        folder = root / sub
        shards: set[str] = set()
        for index in folder.glob("*.safetensors.index.json"):
            try:
                shards.update(json.loads(index.read_text(encoding = "utf-8"))["weight_map"].values())
            except (OSError, ValueError, KeyError, AttributeError):
                return False
        if shards:
            if not all((folder / shard).is_file() for shard in shards):
                return False
        elif not any(folder.glob("*.safetensors")):
            return False
    return True


def fetch_base_snapshot(
    base_repo: str,
    hf_token: Optional[str],
    *,
    local_files_only: bool,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """The base repo's text encoder / tokenizer / VAE snapshot dir. Online, each file goes through the
    same cancellable downloader as the GGUF, so an unload stops the transfer between and within files."""
    local_root = Path(base_repo).expanduser()
    if local_root.is_dir():
        return str(local_root)
    from huggingface_hub import snapshot_download

    if local_files_only:
        from huggingface_hub import constants
        from huggingface_hub.errors import LocalEntryNotFoundError
        from utils.hf_cache_settings import active_hf_hub_cache

        missing: Optional[Exception] = None
        incomplete: Optional[Exception] = None
        # The plan may have staged the mirror even when the upstream holds some weights: try both.
        repos = [prefer_ungated_mirror(base_repo, hf_token), base_repo]
        if not os.environ.get("UNSLOTH_DIFFUSION_NO_MIRROR", "").strip():
            repos.append(mirror_repo(base_repo))
        for repo in dict.fromkeys(r for r in repos if r):
            for cache_dir in dict.fromkeys((active_hf_hub_cache(), constants.HF_HUB_CACHE)):
                try:
                    snapshot = snapshot_download(
                        repo,
                        allow_patterns = list(BASE_ALLOW_PATTERNS),
                        token = hf_token,
                        local_files_only = True,
                        cache_dir = cache_dir,
                    )
                except LocalEntryNotFoundError as exc:
                    missing = exc
                    continue
                if _holds_base_components(snapshot):
                    return snapshot
                incomplete = incomplete or LocalEntryNotFoundError(
                    f"'{repo}' is cached without a complete text_encoder / tokenizer / vae"
                )
        raise incomplete or missing
    import fnmatch

    from huggingface_hub import HfApi
    from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

    def _listing(repo: str) -> tuple[Any, list[str]]:
        info = HfApi().model_info(repo, token = hf_token, files_metadata = False)
        return info, [
            s.rfilename
            for s in info.siblings or ()
            if any(fnmatch.fnmatch(s.rfilename, pattern) for pattern in BASE_ALLOW_PATTERNS)
        ]

    info, wanted = _listing(base_repo)
    # The repo the delegated download plan staged from: the ungated mirror unless the upstream is cached.
    fetch_repo = prefer_ungated_mirror(base_repo, hf_token, files = wanted) if wanted else base_repo
    if fetch_repo != base_repo:
        info, wanted = _listing(fetch_repo)
    if not wanted:
        raise FileNotFoundError(f"'{fetch_repo}' has no text_encoder / tokenizer / vae files")
    from huggingface_hub import constants, try_to_load_from_cache
    from utils.hf_cache_settings import active_hf_hub_cache

    def _holds_all(root: str) -> bool:
        return all(
            isinstance(
                try_to_load_from_cache(fetch_repo, f, revision = info.sha, cache_dir = root), str
            )
            for f in wanted
        )

    # One root for every file: mflux reads the three directories out of a single snapshot.
    cache_dir = active_hf_hub_cache()
    if not _holds_all(cache_dir) and _holds_all(constants.HF_HUB_CACHE):
        cache_dir = constants.HF_HUB_CACHE
    snapshot_dir: Optional[str] = None
    for filename in wanted:
        if cancel_event is not None and cancel_event.is_set():
            raise MlxCancelled("load cancelled")
        path = hf_hub_download_with_xet_fallback(
            fetch_repo,
            filename,
            hf_token,
            cancel_event = cancel_event,
            revision = info.sha,
            cache_dir = cache_dir,
        )
        if snapshot_dir is None:
            snapshot_dir = str(Path(path).parents[len(Path(filename).parts) - 1])
    return snapshot_dir


def resolve_variant_base(
    repo_id: str,
    fam: DiffusionFamily,
    hf_token: Optional[str],
    gguf_filename: Optional[str],
    *,
    fallback: str,
    local_files_only: bool = False,
) -> str:
    """The diffusers loader's base resolution (trusted ``base_model`` card tag, then a named variant,
    then the family default), so both engines read the same base for one pick. Cache-only reads no
    card: the base an earlier online load recorded for this repo, else the name, else the default."""
    if local_files_only:
        return _cached_variant_base(repo_id, fam, gguf_filename, fallback)
    if Path(repo_id).expanduser().exists():
        named = _named_mlx_base(fam, repo_id, gguf_filename)
        if named:
            return named
    try:
        from core.inference.diffusion import _hf_base_model, _resolve_base_repo
    except Exception as exc:  # noqa: BLE001 -- no diffusers stack: the family default stands
        logger.debug("mlx_diffusion.base_resolver_unavailable: %s", exc)
        return fallback
    named = _named_mlx_base(fam, repo_id, gguf_filename)
    try:
        # No card tag (none set, or the Hub read failed): the named variant decides, as offline.
        explicit = named if named and _hf_base_model(repo_id, hf_token) is None else None
        return _resolve_base_repo(repo_id, explicit, fam, hf_token, gguf_filename)
    except Exception as exc:  # noqa: BLE001 -- a metadata failure keeps the named variant or default
        logger.debug("mlx_diffusion.base_resolve_failed: %s", exc)
        return named or fallback


def _cached_variant_base(
    repo_id: str, fam: DiffusionFamily, gguf_filename: Optional[str], fallback: str
) -> str:
    candidates = {
        b.lower()
        for b in (
            fam.base_repo,
            *(getattr(fam, "named_variant_bases", ()) or ()),
            *(b for b, _name in getattr(fam, "mlx_model_config_by_base", ()) or ()),
        )
        if b
    }
    try:
        from hub.utils.companion_assets import read_companion_links
        recorded = read_companion_links().get(repo_id.strip().lower(), ())
    except Exception as exc:  # noqa: BLE001 -- no record: the name decides
        logger.debug("mlx_diffusion.companion_links_unreadable: %s", exc)
        recorded = ()
    named = _named_mlx_base(fam, repo_id, gguf_filename)
    for base in reversed(recorded):
        upstream = canonical_base(base)
        # A recorded family default may be an earlier offline fallback, so it cannot overrule a name.
        if upstream.lower() in candidates and not (
            named and upstream.lower() == fam.base_repo.lower()
        ):
            return upstream
    return named or named_variant_base(fam, repo_id, gguf_filename) or fallback


def _named_mlx_base(fam: DiffusionFamily, *names: Optional[str]) -> Optional[str]:
    """The ``mlx_model_config_by_base`` base the names spell, or None. Per name the longest spelling wins, so a Turbo
    name (which also spells the undistilled ``Z-Image``) keeps the family default, and any name spelling the family
    default (a Turbo repo holding a short file name) keeps it too."""

    def fold(text: str) -> str:
        return "".join(c for c in text.lower() if c.isalnum())

    keyed = [
        (b, fold(b.rsplit("/", 1)[-1]))
        for b in (
            fam.base_repo,
            *(b for b, _name in getattr(fam, "mlx_model_config_by_base", ()) or ()),
        )
        if b
    ]
    found = None
    for name in reversed([n for n in names if n]):
        identity = fold(name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
        hits = [(len(key), b) for b, key in keyed if key and key in identity]
        if hits:
            best = max(hits)[1]
            if best == fam.base_repo:
                return None
            found = found or best
    return found


def _companion_mirror(base: str) -> Optional[str]:
    if os.environ.get("UNSLOTH_DIFFUSION_NO_MIRROR", "").strip():
        return None
    return mirror_repo(base)


def _link_base_components(model_dir: Path, base_dir: str) -> None:
    """Relinked every load: a new revision moves the snapshot."""
    for sub in _LINKED_SUBDIRS:
        source = Path(base_dir) / sub
        if not source.is_dir():
            raise FileNotFoundError(f"base repo snapshot has no {sub}/ directory")
        link = model_dir / sub
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            import shutil
            shutil.rmtree(link)
        os.symlink(str(source.resolve()), str(link), target_is_directory = True)


def _eta(total: int, step: int, first_at: float, now: float) -> Optional[float]:
    if step <= 1 or first_at <= 0 or total <= step:
        return 0.0 if total and step >= total else None
    per_step = (now - first_at) / (step - 1)
    return max(0.0, per_step * (total - step))


def _round16(value: Optional[int], default: int) -> int:
    v = int(value or default)
    return max(64, (v // 16) * 16)


class MlxDiffusionBackend:
    runs_off_torch_device = False

    def __init__(self, runtime_factory: Optional[Callable[[], Any]] = None) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.runtime_factory = runtime_factory or MfluxRuntime
        self._runtime: Any = None
        self._state: Optional[_MlxState] = None
        self._loading: Optional[_MlxLoading] = None
        self._load_token = 0
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        self._active_generate_account: Optional[str] = None
        self._gen: Optional[_MlxGen] = None
        self._load_thread: Optional[threading.Thread] = None
        self._unloading = 0
        self._teardown_lock = threading.Lock()
        self._free_pending = False
        # Repos a cancelled load may still be reading until its worker is joined; see draining_repo_ids().
        self._draining_repos: dict[int, tuple[str, ...]] = {}
        self.last_timings: dict[str, Any] = {}

    def _rt(self) -> Any:
        if self._runtime is None:
            self._runtime = self.runtime_factory()
        return self._runtime

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    # ── Load ─────────────────────────────────────────────────────────────────

    def begin_load(
        self,
        repo_id: str,
        *,
        local_files_only: bool = False,
        display_repo_id: Optional[str] = None,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[Union[str, bool]] = None,
        # Accepted and ignored: uniform route call.
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
        gpu_ids: Optional[list[int]] = None,
        gpu_ordinal: Optional[int] = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if isinstance(hf_token, str):
            hf_token = hf_token.strip() or None
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the MLX engine loads GGUF checkpoints only."
            )
        if model_kind and model_kind != "gguf":
            raise ValueError("The MLX engine loads GGUF checkpoints only.")
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}."
            )
        if not family_mlx_supported(fam):
            raise ValueError(f"Family '{fam.name}' has no MLX implementation.")
        convertible, why = convert.quant_convertible(gguf_filename)
        if not convertible:
            raise ValueError(why or "This GGUF quant is not convertible to MLX.")
        if loras:
            raise ValueError("LoRA adapters are not supported on the MLX engine yet.")
        base = resolve_base_repo(fam, base_repo)
        named = _named_mlx_base(fam, base) if _is_local_path(base) else None
        model_config = mlx_model_config_for(fam, named or base)
        with self._lock:
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            if self._unloading:
                raise RuntimeError("A diffusion unload is in progress.")
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._load_token += 1
            token = self._load_token
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._loading = _MlxLoading(
                repo_id = repo_id, base_repo = base, fetch_repo = _companion_mirror(base)
            )
            worker = account_thread(
                target = self._run_load,
                kwargs = dict(
                    repo_id = repo_id,
                    display_repo_id = display_repo_id,
                    gguf_filename = gguf_filename,
                    base = base,
                    fam = fam,
                    model_config = model_config,
                    hf_token = hf_token,
                    local_files_only = local_files_only,
                    _load_token = token,
                    _cancel_event = cancel_event,
                    explicit_base = bool((base_repo or "").strip()),
                ),
                daemon = True,
            )
            self._load_thread = worker
            worker.start()
        return self.status()

    def _set_phase(
        self,
        token: int,
        phase: str,
        fraction: float = 0.0,
    ) -> None:
        with self._lock:
            if self._load_token == token and self._loading is not None:
                self._loading.phase = phase
                self._loading.fraction = fraction

    @_invalidates_gpu_memory("mlx diffusion load")
    def _run_load(
        self,
        *,
        repo_id: str,
        display_repo_id: Optional[str],
        gguf_filename: str,
        base: str,
        fam: DiffusionFamily,
        model_config: str,
        hf_token: Optional[str],
        local_files_only: bool,
        _load_token: int,
        _cancel_event: threading.Event,
        explicit_base: bool = True,
    ) -> None:
        timings: dict[str, Any] = {}
        model = state = displaced = None
        building = published = False
        try:
            if not explicit_base:
                # The repo's base_model card tag can name the undistilled base: resolve it the way the
                # diffusers loader and the delegated download plan do, before picking the mflux config.
                base = resolve_variant_base(
                    repo_id,
                    fam,
                    hf_token,
                    gguf_filename,
                    fallback = base,
                    local_files_only = local_files_only,
                )
                model_config = mlx_model_config_for(fam, base)
                with self._lock:
                    if self._load_token == _load_token and self._loading is not None:
                        self._loading.base_repo = base
                        self._loading.fetch_repo = _companion_mirror(base)
            t0 = time.perf_counter()
            gguf_path = fetch_gguf(
                repo_id,
                gguf_filename,
                hf_token,
                cancel_event = _cancel_event,
                local_files_only = local_files_only,
            )
            base_dir = fetch_base_snapshot(
                base,
                hf_token,
                local_files_only = local_files_only,
                cancel_event = _cancel_event,
            )
            timings["fetch_s"] = round(time.perf_counter() - t0, 3)
            if _cancel_event.is_set():
                return
            bits = dense_bits()
            revision = _snapshot_revision(gguf_path)
            if revision is None:
                st = os.stat(gguf_path)
                revision = f"local-{st.st_size}-{int(st.st_mtime)}"
            out_dir = convert.cache_dir_for(repo_id, revision, gguf_filename, dense_bits = bits)
            self._set_phase(_load_token, "finalizing", 0.0)
            t1 = time.perf_counter()
            if convert.conversion_complete(out_dir, source_path = gguf_path):
                timings["convert_cached"] = True
            else:
                out_dir.mkdir(parents = True, exist_ok = True)

                def _progress(done: int, total: int) -> None:
                    self._set_phase(_load_token, "finalizing", (done / total) if total else 0.0)

                summary = convert.convert_gguf_to_mflux(
                    gguf_path,
                    out_dir,
                    dense_bits = bits,
                    cancel_event = _cancel_event,
                    progress = _progress,
                )
                timings["convert_cached"] = False
                timings["convert_summary"] = {
                    k: v
                    for k, v in (summary or {}).items()
                    if isinstance(v, (int, float, str, dict))
                }
            timings["convert_s"] = round(time.perf_counter() - t1, 3)
            _link_base_components(Path(out_dir), base_dir)
            if _cancel_event.is_set():
                return
            precision = compute_precision_name()
            te_bits = text_encoder_bits()
            t2 = time.perf_counter()
            runtime = self._rt()
            # Free the replaced model first: two resident DiTs do not fit a small Mac.
            with self._lock:
                displaced = self._state if self._load_token == _load_token else None
                if displaced is not None:
                    self._state = None
            if displaced is not None:
                del displaced
                with self._generate_lock:
                    pass
                runtime.free()
            if _cancel_event.is_set():
                return
            building = True
            model = runtime.build(
                str(out_dir),
                model_config = model_config,
                precision = precision,
                te_bits = te_bits,
            )
            runtime.cache_text_encodings(model)
            timings["build_s"] = round(time.perf_counter() - t2, 3)
            state = _MlxState(
                repo_id = repo_id,
                display_repo_id = display_repo_id,
                gguf_filename = gguf_filename,
                base_repo = base,
                family = fam,
                model = model,
                model_config = model_config,
                precision = precision,
                te_bits = te_bits,
                dense_bits = bits,
                convert_dir = str(out_dir),
                timings = timings,
            )
            with self._lock:
                if self._load_token != _load_token:
                    superseded = True
                else:
                    superseded = False
                    self._state = state
                    self._loading = None
            published = True
            if superseded:
                del state, model
                runtime.free()
                return
            logger.info(
                "mlx_diffusion.loaded: repo=%s gguf=%s config=%s precision=%s te_bits=%s dense_bits=%s "
                "timings=%s",
                repo_id,
                gguf_filename,
                model_config,
                precision,
                te_bits,
                bits,
                timings,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            clear_exception_locals(exc)
            if self._load_token != _load_token or _cancel_event.is_set():
                return
            logger.error("mlx_diffusion.load_failed: %s", exc, exc_info = True)
            if self._state is not None:
                from .gpu_arbiter import DIFFUSION, restore_owner_account
                from hub.services.models.account_access import restore_resident_metadata

                restore_owner_account(DIFFUSION)
                restore_resident_metadata("diffusion")
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == _load_token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))
        finally:
            # A cached exception below (Hub helpers memoize import failures) can retain this frame;
            # a model left in its locals would then survive unload().
            model = state = displaced = None
            if building and not published and self._runtime is not None:
                self._runtime.free()

    def download_plan(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
        """The diffusers GGUF plan stages exactly the files this engine opens; its memory verdict does not apply."""
        from core.inference.diffusion import get_diffusion_backend

        kwargs["memory_verdict"] = False
        if not (kwargs.get("base_repo") or "").strip():
            gguf_filename = kwargs.get("gguf_filename")
            fam = detect_family_for_pick(repo_id, gguf_filename, kwargs.get("family_override"))
            if fam is not None and gguf_filename and family_mlx_supported(fam):
                kwargs["base_repo"] = resolve_variant_base(
                    repo_id,
                    fam,
                    kwargs.get("hf_token"),
                    gguf_filename,
                    fallback = fam.base_repo,
                    local_files_only = bool(kwargs.get("local_files_only")),
                )
        # The loader always opens the base repo's text_encoder/ and vae/, never a single-file component.
        kwargs.pop("text_encoder_files", None)
        kwargs.pop("vae_file", None)
        return get_diffusion_backend().download_plan(
            repo_id, discover_text_encoders = False, **kwargs
        )

    def preflight_base_access(
        self, repo_id: str, fam: Optional[DiffusionFamily], **kwargs: Any
    ) -> None:
        from core.inference.diffusion import get_diffusion_backend
        get_diffusion_backend().preflight_base_access(repo_id, fam, **kwargs)

    def load_progress(self) -> dict[str, Any]:
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)
        return _progress(
            loading.phase,
            loading.downloaded_bytes,
            loading.expected_bytes,
            loading.fraction,
        )

    def loading_repo_ids(self) -> tuple[str, ...]:
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            return tuple(
                dict.fromkeys(
                    r for r in (loading.repo_id, loading.base_repo, loading.fetch_repo) if r
                )
            )

    def draining_repo_ids(self) -> tuple[str, ...]:
        """Repos a cancelled load is still reading until unload() has joined its worker, for the cache-deletion guards
        only (``loading_repo_ids()`` reads as ownership elsewhere)."""
        with self._lock:
            return tuple(dict.fromkeys(r for repos in self._draining_repos.values() for r in repos))

    def loaded_repo_ids(self) -> tuple[str, ...]:
        with self._lock:
            state = self._state
            if state is None:
                return ()
            return tuple(dict.fromkeys((state.repo_id, state.base_repo)))

    # ── Generate ─────────────────────────────────────────────────────────────

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        prompts: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        upscale: Optional[float] = None,
        reference_images: Optional[list[str]] = None,
        workflow: Optional[str] = None,
        reference_resolution: Optional[int] = None,
        localized_edit: Any = None,
        loras: Optional[list[tuple[str, float]]] = None,
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
        expected_load: Optional[LoadIdentity] = None,
        allow_oversized: bool = False,
    ) -> dict[str, Any]:
        if (
            init_image is not None
            or mask_image is not None
            or reference_images
            or workflow not in (None, "txt2img")
            or reference_resolution is not None
            or localized_edit is not None
            or (upscale is not None and upscale > 1)
        ):
            raise ValueError(
                "The MLX engine runs text-to-image only; unset UNSLOTH_DIFFUSION_MLX for "
                "img2img / inpaint / edit / reference / upscale."
            )
        if prompts is not None or seeds is not None:
            raise ValueError(
                "Batched prompt/seed lists are not supported on the MLX engine; use batch_size."
            )
        if controlnet is not None and controlnet[3] not in (None, 0, 0.0):
            raise ValueError("ControlNet is not supported on the MLX engine.")
        if loras:
            raise ValueError("LoRA adapters are not supported on the MLX engine yet.")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be at least 1")
        batch = max(1, int(batch_size or 1))
        base_seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        w = _round16(width, 1024)
        h = _round16(height, 1024)
        family = getattr(self._state, "family", None)
        if family is not None:
            from core.inference.diffusion_conditioning import check_output_size
            check_output_size(family, w, h)

        cancel = threading.Event()
        from core.inference.generation_ticket import bind_generation_cancel

        bind_generation_cancel(cancel)
        from hub.services.models.account_access import media_generation_slot

        with self._generate_lock, media_generation_slot("diffusion"):
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                loaded_id = load_identity(state.repo_id, state.base_repo, state.family.name)
                if expected_load is not None and expected_load != loaded_id:
                    raise DiffusionModelReplacedError(expected_load, loaded_id)
                self._active_generate_cancel = cancel
                self._active_generate_account = current_account_id()
                self._gen = _MlxGen(total_steps = steps * batch)
            try:
                runtime = self._rt()
                cfg = float(guidance or 0.0) if runtime.supports_guidance(state.model) else None
                images = []
                seeds_out = []
                per_image: list[dict[str, float]] = []
                for index in range(batch):
                    image_seed = (base_seed + index) & SEED_MASK
                    offset = index * steps
                    started = time.perf_counter()
                    marks: list[float] = []

                    def _on_step(
                        done: int,
                        offset = offset,
                        marks = marks,
                    ) -> None:
                        marks.append(time.perf_counter())
                        gen = self._gen
                        if gen is not None:
                            now = time.time()
                            gen.step = min(offset + done, gen.total_steps)
                            if gen.first_step_at == 0.0:
                                gen.first_step_at = now
                            gen.eta_seconds = _eta(
                                gen.total_steps, gen.step, gen.first_step_at, now
                            )
                        if cancel.is_set():
                            raise MlxCancelled("cancelled")

                    if cancel.is_set():
                        raise MlxCancelled("cancelled")
                    image = runtime.generate(
                        state.model,
                        prompt = prompt,
                        negative_prompt = negative_prompt,
                        seed = image_seed,
                        steps = steps,
                        width = w,
                        height = h,
                        guidance = cfg,
                        on_step = _on_step,
                    )
                    total = time.perf_counter() - started
                    timing = {"total_s": round(total, 3)}
                    if len(marks) >= 2:
                        # Exclude the first step: it includes prompt encoding.
                        timing["s_per_step"] = round((marks[-1] - marks[0]) / (len(marks) - 1), 4)
                        timing["first_step_s"] = round(marks[0] - started, 3)
                        timing["decode_s"] = round(time.perf_counter() - marks[-1], 3)
                    per_image.append(timing)
                    images.append(image)
                    seeds_out.append(image_seed)
                with self._lock:
                    if cancel.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None
                self.last_timings = {"width": w, "height": h, "steps": steps, "images": per_image}
                logger.info(
                    "mlx_diffusion.generated: %dx%d steps=%d batch=%d timings=%s",
                    w,
                    h,
                    steps,
                    batch,
                    per_image,
                )
                return {
                    "images": images,
                    "seed": base_seed,
                    "seeds": seeds_out,
                    "negative_prompt": (negative_prompt or None) if cfg and cfg > 1.0 else None,
                    "repo_id": state.display_repo_id or state.repo_id,
                    "model_kind": "gguf",
                    "gguf_filename": state.gguf_filename,
                    # The MLX build on top of the GGUF: on-load quantization varies with RAM and env.
                    "transformer_quant": _dense_quant_name(state.dense_bits),
                    "text_encoder_quant": _te_quant_name(state.te_bits),
                    "memory_mode": None,
                    "offload_policy": "none",
                    "speed_mode": None,
                    "cpu_offload": False,
                    "workflow": "txt2img",
                    "reference_resolution": None,
                    "localized_edit": None,
                }
            except MlxCancelled as exc:
                drop_exception_frames(exc)
                raise RuntimeError(DIFFUSION_CANCELLED_MSG) from None
            except BaseException as exc:
                clear_exception_locals(exc)
                raise
            finally:
                state = runtime = None
                self._gen = None
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None

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

    def cancel_generate(self, expected_account: Optional[str] = None) -> bool:
        with self._lock:
            cancel = self._active_generate_cancel
            if cancel is None:
                return False
            if expected_account is not None and self._active_generate_account not in (
                None,
                expected_account,
            ):
                return False
            cancel.set()
            return True

    # ── Unload / status ──────────────────────────────────────────────────────

    @_invalidates_gpu_memory("mlx diffusion unload")
    def unload(self, *, expected_account: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            if expected_account is not None:
                from .gpu_arbiter import DIFFUSION, GpuBusyForAnotherAccountError
                from hub.services.models.account_access import require_resident_control

                if (
                    self._active_generate_cancel is not None
                    and self._active_generate_account != expected_account
                ):
                    raise GpuBusyForAnotherAccountError(DIFFUSION, 1)
                require_resident_control(
                    DIFFUSION, self._state.repo_id if self._state is not None else None
                )
            self._cancel_event.set()
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            if self._state is not None:
                self._free_pending = True
            self._state = None
            self._load_token += 1
            # Left visible until joined: a concurrent unload must wait for the same worker.
            worker = self._load_thread
            if self._loading is not None and worker is not None:
                self._draining_repos[id(worker)] = tuple(
                    dict.fromkeys(
                        r
                        for r in (
                            self._loading.repo_id,
                            self._loading.base_repo,
                            self._loading.fetch_repo,
                        )
                        if r
                    )
                )
            self._loading = None
            # Held until the join and free below: a load started meanwhile would build beside the superseded one.
            self._unloading += 1
        try:
            # Serialized so every unload returns only after the whole teardown, whichever caller runs it.
            with self._teardown_lock:
                # Barrier: on return the cancelled generation has released the model.
                with self._generate_lock:
                    pass
                # And a superseded load has stopped: a build in progress cannot be interrupted, and the
                # caller treats this return as "memory is free" before activating another engine or load.
                if worker is not None and worker is not threading.current_thread():
                    worker.join()
                with self._lock:
                    if self._load_thread is worker:
                        self._load_thread = None
                    if worker is not None:
                        self._draining_repos.pop(id(worker), None)
                    free, self._free_pending = self._free_pending, False
                if free and self._runtime is not None:
                    self._runtime.free()
        finally:
            with self._lock:
                self._unloading -= 1
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "loaded": False,
                "repo_id": None,
                "display_repo_id": None,
                "family": None,
                "base_repo": None,
                "device": None,
                "dtype": None,
                "gguf_variant": None,
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
                "resolved": None,
                "engine": ENGINE_NAME,
                "native_mode": None,
                "supports_lora": False,
                "supports_controlnet": False,
                "workflows": [],
                "conditioning": None,
            }
        from core.inference.diffusion_conditioning import conditioning_capabilities
        from hub.utils.gguf import extract_quant_token

        conditioning = conditioning_capabilities(state.family, ["txt2img"])
        conditioning["reference_resolutions"] = []
        conditioning["notes"] = []
        try:
            negative = bool(self._rt().supports_guidance(state.model))
        except Exception:  # noqa: BLE001
            negative = False
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "display_repo_id": state.display_repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": "mps",
            "dtype": state.precision,
            "model_kind": "gguf",
            "gguf_filename": state.gguf_filename,
            "gguf_variant": extract_quant_token(state.gguf_filename),
            "cpu_offload": False,
            "offload_policy": "none",
            "vae_tiling": False,
            "memory_mode": None,
            "speed_mode": None,
            "speed_optims": [],
            "text_encoder_quant": _te_quant_name(state.te_bits),
            "transformer_quant": _dense_quant_name(state.dense_bits),
            "attention_backend": None,
            "transformer_cache": None,
            "resolved": None,
            "engine": ENGINE_NAME,
            "native_mode": None,
            "mlx": {
                "model_config": state.model_config,
                "precision": state.precision,
                "text_encoder_bits": state.te_bits,
                "dense_bits": state.dense_bits,
                "timings": dict(state.timings),
            },
            "supports_lora": False,
            "supports_controlnet": False,
            "supports_negative_prompt": negative,
            "prompt_hint": state.family.prompt_hint,
            "workflows": ["txt2img"],
            "conditioning": conditioning,
        }


def _dense_quant_name(bits: Optional[int]) -> Optional[str]:
    return f"mlx_int{bits}_dense" if bits else None


def _te_quant_name(bits: Optional[int]) -> Optional[str]:
    return f"mlx_int{bits}" if bits else None


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


_mlx_diffusion_backend: Optional[MlxDiffusionBackend] = None


def get_mlx_diffusion_backend() -> MlxDiffusionBackend:
    global _mlx_diffusion_backend
    if _mlx_diffusion_backend is None:
        _mlx_diffusion_backend = MlxDiffusionBackend()
    return _mlx_diffusion_backend


def generation_in_flight() -> bool:
    backend = _mlx_diffusion_backend
    return backend is not None and backend._gen is not None
