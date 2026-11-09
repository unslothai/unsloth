# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Releases text encoders during denoise on unified memory (bit-exact reload from the encoder's own safetensors files,
else from a disk snapshot)."""

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
# 0: snapshot every released tensor, as before, instead of reloading the ones the encoder's own files hold byte for byte.
SOURCE_REUSE_ENV = "UNSLOTH_DIFFUSION_TE_RELEASE_FROM_SOURCE"
# Set by a loader that knows the safetensors files an encoder was built from; else ``name_or_path`` when it is a folder.
WEIGHT_FILES_ATTR = "_unsloth_weight_files"
# A folder with more files than this is not one encoder's checkpoint.
_MAX_SOURCE_FILES = 64
_MAX_CANDIDATES = 4
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


def source_reuse_enabled() -> bool:
    return (os.environ.get(SOURCE_REUSE_ENV) or "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _source_files(encoder: Any) -> list[Path]:
    tagged = getattr(encoder, WEIGHT_FILES_ATTR, None)
    if tagged:
        files = [Path(str(f)) for f in tagged]
    else:
        files = []
        for name in (
            getattr(encoder, "name_or_path", None),
            getattr(getattr(encoder, "config", None), "_name_or_path", None),
        ):
            try:
                folder = Path(str(name)) if name else None
                if folder is not None and folder.is_dir():
                    files = sorted(folder.glob("*.safetensors"))
                    break
            except OSError:
                continue
    files = [f for f in files if f.suffix == ".safetensors" and f.is_file()]
    return files if len(files) <= _MAX_SOURCE_FILES else []


def _safetensors_dtype(name: str) -> Any:
    import torch
    return {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
    }.get(str(name))


def _same_bytes(a: Any, b: Any) -> bool:
    import torch
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)
    )


def release_wanted(plan: Any) -> tuple[bool, str]:
    """``(wanted, reason)``: auto engages only on unified memory when weights plus generation headroom exceed the
    safe budget."""
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
        return (
            False,
            f"weights and generation headroom ({required} MiB) fit the {budget} MiB budget",
        )
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
        if (
            getattr(sub, "_hf_hook", None) is not None
            or getattr(sub, "_diffusers_hook", None) is not None
        ):
            return True
    return False


def _plain_tensor(tensor: Any) -> bool:
    import torch
    return type(tensor) in (torch.Tensor, torch.nn.Parameter) and tensor.device.type != "meta"


class TextEncoderReleaser:
    """Frees a pipeline's text-encoder weights during denoise and restores them, bit for bit, before they run again."""

    def __init__(
        self,
        encoders: list[tuple[str, Any]],
        *,
        logger: Any = None,
    ) -> None:
        self._encoders = encoders
        self._logger = logger
        self._lock = threading.RLock()
        self._released = False
        self._snapshot: Optional[Path] = None
        self._shards: list[Path] = []
        self._prepared = False
        # Open handles keep a source file's bytes readable even if the path is replaced or deleted meanwhile.
        self._source_handles: list[Any] = []
        # key -> (index into _source_handles, tensor name in that file)
        self._from_source: dict[str, tuple[int, str]] = {}
        # key -> (owner module, leaf name, is_parameter)
        self._slots: dict[str, tuple[Any, str, bool]] = {}
        self._released_bytes = 0
        self._bytes: dict[str, int] = {}
        self._hooks: list[Any] = []
        self.reloads = 0
        self.releases = 0
        self._collect()
        self._install_hooks()
        self._finalizer = weakref.finalize(self, _remove_dir, None)

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

    def _tensor_bytes(self, key: str) -> int:
        return self._bytes.get(key, 0)

    def _tensor(self, key: str) -> Any:
        owner, leaf, is_param = self._slots[key]
        return (owner._parameters if is_param else owner._buffers)[leaf]

    def _match_sources(self) -> None:
        """Map tensors whose bytes match the encoder's own files to them; the rest are snapshotted."""
        import torch
        from safetensors import safe_open

        for attr, encoder in self._encoders:
            keys = [
                k for k in self._slots if k.startswith(attr + ".") and k not in self._from_source
            ]
            if not keys:
                continue
            by_suffix: dict[str, list[tuple[int, str]]] = {}
            for path in _source_files(encoder):
                try:
                    handle = safe_open(str(path), framework = "pt", device = "cpu")
                    names = list(handle.keys())
                except Exception:  # noqa: BLE001 - an unreadable file is just not a source
                    continue
                self._source_handles.append(handle)
                index = len(self._source_handles) - 1
                for name in names:
                    parts = name.split(".")
                    for i in range(len(parts)):
                        by_suffix.setdefault(".".join(parts[i:]), []).append((index, name))
            if not by_suffix:
                continue
            for key in keys:
                parts = [p for p in key[len(attr) + 1 :].split(".") if p]
                candidates: list[tuple[int, str]] = []
                # Longest matching name suffix first: transformers may nest the checkpoint's names one level deeper.
                for i in range(len(parts)):
                    candidates = by_suffix.get(".".join(parts[i:]), [])
                    if candidates:
                        break
                tensor = self._tensor(key).detach()
                for index, name in candidates[:_MAX_CANDIDATES]:
                    handle = self._source_handles[index]
                    try:
                        meta = handle.get_slice(name)
                        if _safetensors_dtype(meta.get_dtype()) != tensor.dtype or tuple(
                            meta.get_shape()
                        ) != tuple(tensor.shape):
                            continue
                        with torch.no_grad():
                            if _same_bytes(handle.get_tensor(name), tensor.to("cpu")):
                                self._from_source[key] = (index, name)
                                break
                    except Exception:  # noqa: BLE001 - unreadable: snapshot this tensor instead
                        continue
        used = sorted({index for index, _ in self._from_source.values()})
        remap = {old: new for new, old in enumerate(used)}
        self._source_handles = [self._source_handles[i] for i in used]
        self._from_source = {k: (remap[i], name) for k, (i, name) in self._from_source.items()}

    def _prepare(self) -> None:
        if source_reuse_enabled():
            try:
                self._match_sources()
            except Exception as exc:  # noqa: BLE001 - reuse is an optimisation: snapshot everything
                self._from_source = {}
                self._source_handles = []
                self._note(
                    "info", "diffusion.text_encoder_release: snapshotting every tensor: %s", exc
                )
        if any(key not in self._from_source for key in self._slots):
            self._write_snapshot()
        self._prepared = True

    def _write_snapshot(self) -> None:
        import torch
        from safetensors.torch import save_file

        root = _snapshot_root()
        root.mkdir(parents = True, exist_ok = True)
        _sweep_stale(root)
        need = sum(
            self._tensor(key).numel() * self._tensor(key).element_size()
            for key in self._slots
            if key not in self._from_source
        )
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
                if key in self._from_source:
                    continue
                tensor = self._tensor(key).detach()
                batch[key] = tensor.to("cpu", copy = True).contiguous()
                size += tensor.numel() * tensor.element_size()
                if size >= _SHARD_BYTES:
                    _flush()
            _flush()
        self._snapshot = snapshot
        self._shards = shards

    def release(self) -> int:
        """Bytes freed; 0 when already released or refused."""
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
                if not self._prepared:
                    self._prepare()
            except Exception as exc:  # noqa: BLE001 - releasing is an optimisation: stay resident
                self._note("warning", "diffusion.text_encoder_release: kept resident: %s", exc)
                self._slots = {}
                self._from_source = {}
                self._source_handles = []
                return 0
            released = 0
            with torch.no_grad():
                for key, (owner, leaf, is_param) in self._slots.items():
                    tensor = self._tensor(key)
                    self._bytes[key] = tensor.numel() * tensor.element_size()
                    released += self._bytes[key]
                    empty = torch.empty(0, dtype = tensor.dtype, device = tensor.device)
                    if is_param:
                        owner._parameters[leaf].data = empty
                    else:
                        owner._buffers[leaf] = empty
            self._released = True
            self._released_bytes = released
            self.releases += 1
            _empty_device_cache()
            from_source = sum(
                self._tensor_bytes(key) for key in self._from_source if key in self._slots
            )
            self._note(
                "info",
                "diffusion.text_encoder_release: released %.2f GB of text-encoder weights for the denoise (%.2f s; "
                "%.2f GB reload from the encoder's own files, %.2f GB from a snapshot)",
                released / 1e9,
                time.monotonic() - started,
                from_source / 1e9,
                (released - from_source) / 1e9,
            )
            return released

    def ensure_loaded(self) -> None:
        """Restore every released tensor from the snapshot onto its original device."""
        import torch
        from safetensors import safe_open

        with self._lock:
            if not self._released:
                return
            started = time.monotonic()
            with torch.no_grad():
                for key, (index, name) in self._from_source.items():
                    owner, leaf, is_param = self._slots[key]
                    current = (owner._parameters if is_param else owner._buffers)[leaf]
                    restored = self._source_handles[index].get_tensor(name).to(current.device)
                    if is_param:
                        owner._parameters[leaf].data = restored
                    else:
                        owner._buffers[leaf] = restored
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
        """Drop the hooks and the snapshot; the weights stay as they are."""
        with self._lock:
            for hook in self._hooks:
                try:
                    hook.remove()
                except Exception:  # noqa: BLE001
                    pass
            self._hooks = []
            self._from_source = {}
            self._source_handles = []
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


def maybe_install(
    pipe: Any,
    plan: Any,
    *,
    logger: Any = None,
) -> Optional[TextEncoderReleaser]:
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
