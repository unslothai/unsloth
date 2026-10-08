# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Release the text encoders between prompts on unified memory.

On Apple Silicon (and integrated GPUs) the CPU and GPU share one memory pool, so offloading a component to the CPU
frees nothing, and the memory planner keeps every weight resident. The text encoders run once, before step 0, yet
they stay resident through the whole denoise and VAE decode. For Qwen-Image-2.1 that is the 17.5 GB Qwen3-VL-8B
next to a 4.2 GB GGUF denoiser.

When the planned resident set (weights plus generation headroom) does not fit the device budget, this releases the
encoders' weights as soon as the denoise loop starts and loads them back the next time an encoder runs. The first
release writes the encoders' tensors to a local snapshot (once per load), and every reload reads that snapshot, so a
reload is exact and needs no network. Studio's prompt cache answers a repeated prompt without running the encoders,
so the same prompt with a new seed never reloads.

``UNSLOTH_DIFFUSION_RELEASE_TEXT_ENCODER=1`` forces it on, ``=0`` forces it off. torch is imported lazily.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Any, Optional

RELEASE_ENV = "UNSLOTH_DIFFUSION_RELEASE_TEXT_ENCODER"
TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")
# Tensors below this stay resident: norms, biases and rotary tables are a rounding error next to the projections.
_MIN_RELEASE_BYTES = 1 << 20
# Snapshot shard size: bounds the host copy a write holds at once (unified memory has no second pool to spill to).
_SHARD_BYTES = 512 << 20
_SNAPSHOT_DIR = "text_encoder_release"


def release_override() -> Optional[bool]:
    """``True`` / ``False`` when ``UNSLOTH_DIFFUSION_RELEASE_TEXT_ENCODER`` forces it, else None (auto)."""
    value = (os.environ.get(RELEASE_ENV) or "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return None


def release_wanted(plan: Any) -> tuple[bool, str]:
    """Whether a load committed to ``plan`` should release its text encoders between prompts, and why.

    Auto engages only on unified memory, where nothing else can make room, and only when the planned resident
    requirement (every weight plus the generation headroom and base overhead) exceeds the safe device budget. A
    load that fits keeps its encoders resident and pays no reload."""
    forced = release_override()
    if forced is not None:
        return forced, f"{RELEASE_ENV}={'1' if forced else '0'}"
    try:
        memory = plan.device_memory
        if getattr(memory, "memory_kind", None) != "unified_memory":
            return False, "not unified memory"
        estimates = plan.estimates
        budget = estimates.get("safe_device_budget_mib")
        required = estimates.get("resident_required_mib")
        encoders = estimates.get("text_encoder_dense_mib")
    except Exception:  # noqa: BLE001 - no plan, no release
        return False, "no memory plan"
    if budget is None or required is None or not encoders:
        return False, "budget or text-encoder size unknown"
    if int(required) <= int(budget):
        return False, f"weights and generation headroom ({required} MiB) fit the {budget} MiB budget"
    return True, (
        f"unified memory: weights and generation headroom ({required} MiB) exceed the {budget} MiB budget, "
        f"and the text encoders ({encoders} MiB) run once per prompt"
    )


def _snapshot_root() -> Path:
    try:
        from utils.paths.storage_roots import cache_root

        return Path(cache_root()) / _SNAPSHOT_DIR
    except Exception:  # noqa: BLE001 - no Studio home (tests, bare scripts): the system temp dir
        import tempfile

        return Path(tempfile.gettempdir()) / f"unsloth_{_SNAPSHOT_DIR}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001 - permission errors mean the process exists
        return True
    return True


def _sweep_stale(root: Path) -> None:
    """Delete snapshots a crashed backend left behind (the directory name carries its pid)."""
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        pid_text = entry.name.split("-", 1)[0]
        if pid_text.isdigit() and int(pid_text) != os.getpid() and not _pid_alive(int(pid_text)):
            shutil.rmtree(entry, ignore_errors = True)


def _offload_hooked(module: Any) -> bool:
    """accelerate / diffusers offload hooks move weights themselves; never release under them."""
    for sub in module.modules():
        if getattr(sub, "_hf_hook", None) is not None or getattr(sub, "_diffusers_hook", None) is not None:
            return True
    return False


def _plain_tensor(tensor: Any) -> bool:
    import torch

    return type(tensor) in (torch.Tensor, torch.nn.Parameter) and tensor.device.type != "meta"


class TextEncoderReleaser:
    """Frees a pipeline's text-encoder weights during denoise and restores them, bit for bit, before they run again."""

    def __init__(self, encoders: list[tuple[str, Any]], *, logger: Any = None) -> None:
        self._encoders = encoders
        self._logger = logger
        self._lock = threading.RLock()
        self._released = False
        self._snapshot: Optional[Path] = None
        self._shards: list[Path] = []
        # key -> (owner module, leaf name, is_parameter)
        self._slots: dict[str, tuple[Any, str, bool]] = {}
        self._released_bytes = 0
        self._hooks: list[Any] = []
        self.reloads = 0
        self.releases = 0
        self._collect()
        self._install_hooks()
        self._finalizer = weakref.finalize(self, _remove_dir, None)

    # -- discovery -------------------------------------------------------------------------------------------------

    def _collect(self) -> None:
        seen: set[int] = set()
        for attr, encoder in self._encoders:
            for module_name, module in encoder.named_modules():
                for leaf, param in list(module._parameters.items()):
                    if param is None or id(param) in seen:
                        continue
                    seen.add(id(param))
                    if param.numel() * param.element_size() >= _MIN_RELEASE_BYTES:
                        self._slots[f"{attr}.{module_name}.{leaf}"] = (module, leaf, True)
                for leaf, buf in list(module._buffers.items()):
                    if buf is None or id(buf) in seen or leaf in module._non_persistent_buffers_set:
                        continue
                    seen.add(id(buf))
                    if buf.numel() * buf.element_size() >= _MIN_RELEASE_BYTES:
                        self._slots[f"{attr}.{module_name}.{leaf}"] = (module, leaf, False)

    def _install_hooks(self) -> None:
        owners = {id(owner): owner for owner, _, _ in self._slots.values()}
        for _, encoder in self._encoders:
            owners.setdefault(id(encoder), encoder)

        def _pre_hook(module: Any, args: Any) -> None:
            if self._released:
                self.ensure_loaded()

        for owner in owners.values():
            self._hooks.append(owner.register_forward_pre_hook(_pre_hook))

    # -- release / reload ------------------------------------------------------------------------------------------

    @property
    def released(self) -> bool:
        return self._released

    @property
    def releasable_bytes(self) -> int:
        return sum(
            getattr(owner, "_parameters" if is_param else "_buffers")[leaf].numel()
            * getattr(owner, "_parameters" if is_param else "_buffers")[leaf].element_size()
            for owner, leaf, is_param in self._slots.values()
        )

    def _tensor(self, key: str) -> Any:
        owner, leaf, is_param = self._slots[key]
        return (owner._parameters if is_param else owner._buffers)[leaf]

    def _write_snapshot(self) -> None:
        import torch
        from safetensors.torch import save_file

        root = _snapshot_root()
        root.mkdir(parents = True, exist_ok = True)
        _sweep_stale(root)
        need = self.releasable_bytes
        free = shutil.disk_usage(root).free
        if free < need + (1 << 30):
            raise RuntimeError(
                f"not enough free disk for the text-encoder snapshot ({need / 1e9:.1f} GB needed, "
                f"{free / 1e9:.1f} GB free under {root})"
            )
        snapshot = root / f"{os.getpid()}-{uuid.uuid4().hex[:10]}"
        snapshot.mkdir()
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, _remove_dir, str(snapshot))
        shards: list[Path] = []
        batch: dict[str, Any] = {}
        size = 0

        def _flush() -> None:
            nonlocal batch, size
            if not batch:
                return
            path = snapshot / f"shard-{len(shards):05d}.safetensors"
            save_file(batch, str(path))
            shards.append(path)
            batch, size = {}, 0

        with torch.no_grad():
            for key in self._slots:
                tensor = self._tensor(key).detach()
                batch[key] = tensor.to("cpu", copy = True).contiguous()
                size += tensor.numel() * tensor.element_size()
                if size >= _SHARD_BYTES:
                    _flush()
            _flush()
        self._snapshot = snapshot
        self._shards = shards

    def release(self) -> int:
        """Free the encoders' large tensors; returns the bytes released (0 when already released or refused)."""
        import torch

        with self._lock:
            if self._released or not self._slots:
                return 0
            started = time.monotonic()
            try:
                for _, encoder in self._encoders:
                    if _offload_hooked(encoder):
                        raise RuntimeError("an offload hook manages these weights")
                for key in self._slots:
                    if not _plain_tensor(self._tensor(key)):
                        raise RuntimeError(f"{key} is not a plain dense tensor")
                if self._snapshot is None:
                    self._write_snapshot()
            except Exception as exc:  # noqa: BLE001 - releasing is an optimisation: stay resident
                self._note("warning", "diffusion.text_encoder_release: kept resident: %s", exc)
                self._slots = {}
                return 0
            released = 0
            with torch.no_grad():
                for key, (owner, leaf, is_param) in self._slots.items():
                    tensor = self._tensor(key)
                    released += tensor.numel() * tensor.element_size()
                    empty = torch.empty(0, dtype = tensor.dtype, device = tensor.device)
                    if is_param:
                        owner._parameters[leaf].data = empty
                    else:
                        owner._buffers[leaf] = empty
            self._released = True
            self._released_bytes = released
            self.releases += 1
            _empty_device_cache()
            self._note(
                "info",
                "diffusion.text_encoder_release: released %.2f GB of text-encoder weights for the denoise (%.2f s)",
                released / 1e9,
                time.monotonic() - started,
            )
            return released

    def ensure_loaded(self) -> None:
        """Restore every released tensor from the snapshot, on its original device. No-op when resident."""
        import torch
        from safetensors import safe_open

        with self._lock:
            if not self._released:
                return
            started = time.monotonic()
            with torch.no_grad():
                for shard in self._shards:
                    with safe_open(str(shard), framework = "pt", device = "cpu") as handle:
                        for key in handle.keys():
                            owner, leaf, is_param = self._slots[key]
                            current = (owner._parameters if is_param else owner._buffers)[leaf]
                            restored = handle.get_tensor(key).to(current.device)
                            if is_param:
                                owner._parameters[leaf].data = restored
                            else:
                                owner._buffers[leaf] = restored
            self._released = False
            self.reloads += 1
            self._note(
                "info",
                "diffusion.text_encoder_release: reloaded %.2f GB of text-encoder weights in %.2f s",
                self._released_bytes / 1e9,
                time.monotonic() - started,
            )

    def close(self) -> None:
        """Drop the hooks and the snapshot. The weights stay as they are (the pipeline is being dropped)."""
        with self._lock:
            for hook in self._hooks:
                try:
                    hook.remove()
                except Exception:  # noqa: BLE001
                    pass
            self._hooks = []
            self._finalizer()

    def _note(self, level: str, msg: str, *args: Any) -> None:
        if self._logger is not None:
            getattr(self._logger, level)(msg, *args)


def _remove_dir(path: Optional[str]) -> None:
    if path:
        shutil.rmtree(path, ignore_errors = True)


def _empty_device_cache() -> None:
    import torch

    for backend in ("mps", "cuda"):
        module = getattr(torch, backend, None)
        try:
            if module is not None and module.is_available():
                module.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def maybe_install(pipe: Any, plan: Any, *, logger: Any = None) -> Optional[TextEncoderReleaser]:
    """A releaser for ``pipe`` when ``plan`` calls for one (see ``release_wanted``), else None. Never raises."""
    try:
        wanted, reason = release_wanted(plan)
        if not wanted:
            return None
        encoders = [
            (attr, getattr(pipe, attr))
            for attr in TEXT_ENCODER_ATTRS
            if _is_module(getattr(pipe, attr, None))
        ]
        if not encoders:
            return None
        releaser = TextEncoderReleaser(encoders, logger = logger)
        if not releaser._slots:
            releaser.close()
            return None
        if logger is not None:
            logger.info(
                "diffusion.text_encoder_release: on for %s (%.2f GB): %s",
                ", ".join(a for a, _ in encoders),
                releaser.releasable_bytes / 1e9,
                reason,
            )
        return releaser
    except Exception as exc:  # noqa: BLE001 - an optimisation: the encoders stay resident
        if logger is not None:
            logger.warning("diffusion.text_encoder_release: not installed: %s", exc)
        return None


def _is_module(value: Any) -> bool:
    try:
        import torch

        return isinstance(value, torch.nn.Module)
    except Exception:  # noqa: BLE001
        return False

