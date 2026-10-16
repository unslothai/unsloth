# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, slot save and restore."""

import contextlib
import httpx
import os
import shutil
import uuid

from pathlib import Path
from typing import (
    Callable,
    Optional,
)
from core.inference.llama_server_args import _flag_name

from .core_types import (
    _SLOT_SAVE_HTTP_TIMEOUT,
    _SLOT_SAVE_MAX_BYTES,
)
from .gguf_cache import _SHARD_FULL_RE
from .launch_flags import _LLAMA_ARG_FALSE_VALUES
from .log import logger
from .memory_policy import _kv_bytes_per_elem


class _SlotsMixin:
    def _probe_mtp_decode(self, timeout: float = 60.0) -> bool:
        """One tiny /completion to confirm MTP survives the first decode.

        MTP-draft can pass /health yet crash the flash-attn kernel only once
        tokens generate (e.g. under --split-mode tensor). False on any error so
        the caller can drop MTP and retry.
        """
        url = f"{self.base_url}/completion"
        payload = {"prompt": "Hi", "n_predict": 4, "temperature": 0.0, "stream": False}
        try:
            resp = httpx.post(
                url,
                json = payload,
                timeout = timeout,
                headers = self._auth_headers,
                trust_env = False,
            )
        except Exception as e:
            logger.debug(f"MTP decode probe failed: {e}")
            return False
        if resp.status_code != 200:
            logger.debug(f"MTP decode probe returned HTTP {resp.status_code}")
            return False
        # A crash can drop the connection or kill the process right after a reply.
        if self._process is not None and self._process.poll() is not None:
            return False
        return True

    def _slot_launch_fingerprint(self) -> tuple:
        # KV validity keys on extra args, stat'd sidecar weights, effective ctx.
        sidecars = []
        for path in self._sidecar_weight_files():
            try:
                st = os.stat(path)
                sidecars.append((path, st.st_size, st.st_mtime_ns))
            except OSError:
                sidecars.append((path, None, None))
        return (
            tuple(self._extra_args or ()),
            tuple(sidecars),
            self._requested_n_ctx,
            self._effective_context_length,
            self._effective_cache_types,
            self.effective_parallel_slots,
            self._swa_full,
            self._kv_cache_unified,
            self._n_ubatch,
            self._flash_attn_enabled,
        )

    @staticmethod
    def _gguf_load_source_identity(path: str, mmproj_path: Optional[str] = None) -> Optional[tuple]:
        """Identity of the exact GGUF inode(s) handed to the resident process."""
        p = Path(path)
        paths = [p]
        match = _SHARD_FULL_RE.match(p.name)
        if match:
            prefix, _first, total = match.groups()
            paths = [
                p.with_name(f"{prefix}-{index:05d}-of-{total}{p.suffix}")
                for index in range(1, int(total) + 1)
            ]
        try:
            identity = []
            for shard in paths:
                resolved = shard.resolve()
                stat = shard.stat()
                identity.append(
                    (
                        str(resolved),
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
            if mmproj_path:
                projector = Path(mmproj_path)
                resolved = projector.resolve()
                stat = projector.stat()
                identity.append(
                    (
                        "mmproj",
                        str(resolved),
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
            return tuple(identity)
        except (OSError, RuntimeError):
            return None

    def _gguf_file_identity(self, path) -> Optional[tuple]:
        # (size, mtime_ns) per shard: a split GGUF keys KV validity on every sibling.
        p = Path(path)
        paths = [p]
        m = _SHARD_FULL_RE.match(p.name)
        if m:
            prefix, _first, total = m.groups()
            paths = [
                p.with_name(f"{prefix}-{i:05d}-of-{total}{p.suffix}")
                for i in range(1, int(total) + 1)
            ]
        try:
            return tuple((sp.stat().st_size, sp.stat().st_mtime_ns) for sp in paths)
        except OSError:
            return None

    _SIDECAR_WEIGHT_FLAGS = (
        "--mmproj",
        "-mm",
        "--lora",
        "--lora-scaled",
        "--control-vector",
        "--control-vector-scaled",
    )

    def _sidecar_weight_files(self) -> list[str]:
        # llama.cpp: comma-separated paths, FNAME:SCALE on -scaled (older builds: FNAME SCALE).
        args = [str(a).strip() for a in (self._extra_args or ())]
        files: list[str] = []
        for i, arg in enumerate(args):
            flag = _flag_name(arg)
            _, sep, inline = arg.partition("=")
            if flag not in self._SIDECAR_WEIGHT_FLAGS:
                continue
            operand = inline if sep else (args[i + 1] if i + 1 < len(args) else "")
            if not operand:
                continue
            candidates = [operand]
            pieces = [p for p in operand.split(",") if p]
            if len(pieces) > 1:
                candidates.extend(pieces)
            if flag.endswith("-scaled"):
                for item in list(candidates):
                    # ":<number>" tail is a scale; rpartition spares drive letters.
                    head, colon, tail = item.rpartition(":")
                    if not (colon and head):
                        continue
                    try:
                        float(tail)
                    except ValueError:
                        continue
                    candidates.append(head)
            for cand in candidates:
                if cand not in files:
                    files.append(cand)
        return files

    def _prompt_cache_off(self) -> bool:
        # Caching off makes restores useless; last prompt-cache flag wins, env only when unset.
        last = None
        for arg in self._extra_args or ():
            flag = arg.strip().split("=", 1)[0]
            if flag in ("--cache-prompt", "--no-cache-prompt"):
                last = flag
        if last is not None:
            return last == "--no-cache-prompt"
        if self._prompt_cache_disabled:
            return True
        if os.environ.get("LLAMA_ARG_NO_CACHE_PROMPT") is not None:
            return True
        env = (os.environ.get("LLAMA_ARG_CACHE_PROMPT") or "").strip().lower()
        return env in _LLAMA_ARG_FALSE_VALUES

    def save_slots_for_resume(
        self, should_abort: Optional[Callable[[], bool]] = None
    ) -> Optional[dict]:
        if (
            not self.is_loaded
            or not self._slot_save_dir
            or not self._gguf_path
            or self._prompt_cache_off()
        ):
            return None
        # Same predicate as the estimator's SWA path: a window alone is not enough.
        # phi3 GGUFs carry attention.sliding_window but no key/value length, and
        # llama.cpp forces them back to a non-SWA cache, so their slots do restore.
        if (
            (self._sliding_window or 0) > 0
            and self._kv_key_length is not None
            and self._kv_value_length is not None
            and not self._swa_full
        ):
            logger.debug("Skipping slot save: compact SWA cache cannot be reused after restart")
            return None
        save_dir = Path(self._slot_save_dir)
        gguf_stat = self._gguf_file_identity(self._gguf_path)
        if gguf_stat is None:
            return None
        launch = self._slot_launch_fingerprint()
        # If the GGUF or a sidecar was swapped on disk while the original weights
        # stayed mapped, the live KV belongs to the old weights but a reload would
        # load the new file. Persisting it would let restore misapply stale KV.
        if self._slot_loaded_identity is not None and self._slot_loaded_identity != (
            gguf_stat,
            launch,
        ):
            logger.debug("Skipping slot save: model files changed on disk since load")
            return None
        try:
            estimate = self._estimate_kv_cache_bytes(
                self._kv_cache_context_total
                or self._effective_context_length
                or self._context_length
                or 0,
                max(self._effective_cache_types, key = _kv_bytes_per_elem),
                n_parallel = self.effective_parallel_slots,
                swa_full = self._swa_full,
                kv_unified = self._kv_cache_unified,
                n_ubatch = self._n_ubatch,
                flash_attn = self._flash_attn_enabled,
            )
            # Skip before writing anything when the estimate alone blows the cap,
            # rather than fully writing a slot and discarding it afterwards.
            if estimate > _SLOT_SAVE_MAX_BYTES:
                logger.debug(
                    "Skipping slot save: estimated %d bytes exceeds cap %d",
                    estimate,
                    _SLOT_SAVE_MAX_BYTES,
                )
                return None
            # A 0 estimate means metadata was insufficient, not a zero-byte cache:
            # a slot can still be many GiB, so demand room for the whole cap before
            # trusting the post-write check.
            required = (estimate if estimate > 0 else _SLOT_SAVE_MAX_BYTES) + (1 << 30)
            if shutil.disk_usage(save_dir).free < required:
                logger.debug("Skipping slot save: insufficient free disk")
                return None
        except Exception:
            pass
        token = uuid.uuid4().hex[:8]
        entries: list[dict] = []
        total_bytes = 0
        for slot in range(self.effective_parallel_slots):
            # A request pending mid-save waits on the gate; stop wasting its time.
            if should_abort is not None and should_abort():
                break
            filename = f"resume-{token}-slot{slot}.bin"
            path = save_dir / filename
            try:
                resp = httpx.post(
                    f"{self.base_url}/slots/{slot}",
                    params = {"action": "save"},
                    json = {"filename": filename},
                    headers = self._auth_headers,
                    timeout = _SLOT_SAVE_HTTP_TIMEOUT,
                    trust_env = False,
                )
            except Exception as e:
                logger.debug(f"slot {slot} save failed: {e}")
                with contextlib.suppress(OSError):
                    path.unlink()
                break
            if resp.status_code != 200:
                logger.debug(f"slot {slot} save returned HTTP {resp.status_code}")
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            try:
                body = resp.json()
                if not isinstance(body, dict):
                    raise ValueError("slot save response was not a JSON object")
                n_saved = int(body.get("n_saved") or 0)
            except Exception as e:
                # A 200 that still wrote a file but returns a malformed body must
                # clean up like the transport/HTTP error paths above, or the file
                # (which holds chat KV) is orphaned until the next startup sweep.
                logger.debug(f"slot {slot} save returned an invalid response: {e}")
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            if n_saved <= 0:
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            # Account by the bytes actually on disk, not the server-reported
            # count, so the cap holds even if a custom binary under-reports.
            try:
                n_written = path.stat().st_size
            except OSError:
                n_written = 0
            total_bytes += n_written
            entries.append({"id": slot, "filename": filename, "n_saved": n_saved})
            if total_bytes > _SLOT_SAVE_MAX_BYTES:
                break  # already over the cap; the discard below cleans up
        if not entries:
            return None
        if total_bytes > _SLOT_SAVE_MAX_BYTES:
            logger.debug(
                "Discarding slot save: %d bytes exceeds cap %d",
                total_bytes,
                _SLOT_SAVE_MAX_BYTES,
            )
            for entry in entries:
                with contextlib.suppress(OSError):
                    (save_dir / entry["filename"]).unlink()
            return None
        return {
            "dir": self._slot_save_dir,
            "binary": self._slot_save_binary,
            "gguf": str(self._gguf_path),
            "gguf_stat": gguf_stat,
            "launch": launch,
            "slots": entries,
        }

    def restore_slots_for_resume(self, manifest: dict) -> None:
        if not self.is_loaded or not self._slot_save_dir:
            return
        for entry in manifest.get("slots") or []:
            try:
                resp = httpx.post(
                    f"{self.base_url}/slots/{int(entry['id'])}",
                    params = {"action": "restore"},
                    json = {"filename": str(entry["filename"])},
                    headers = self._auth_headers,
                    timeout = _SLOT_SAVE_HTTP_TIMEOUT,
                    trust_env = False,
                )
            except Exception as e:
                logger.debug(f"slot restore failed: {e}")
                break
            if resp.status_code != 200:
                logger.debug(f"slot {entry.get('id')} restore returned HTTP {resp.status_code}")
