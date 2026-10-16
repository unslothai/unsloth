# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load-in-flight markers shared with the routes."""

import contextlib
import functools
import os
import threading

from typing import (
    Callable,
    Optional,
)
from core.inference.llama_server_args import extra_args_disable_mmproj

from .binary_runtime import _load_probe_memo_scope
from .core_types import GgufLoadIntent
from .gguf_cache import _hub_download_blocks_gguf_load
from .spec_args import _canonicalize_spec_mode

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.

# Active GGUF loads by normalized repo ID.
_LOADS_IN_FLIGHT: dict[str, int] = {}
_LOADS_IN_FLIGHT_LOCK = threading.Lock()


@contextlib.contextmanager
def gguf_load_in_flight(hf_repo: Optional[str]):
    """Track an HF GGUF load until the context exits."""
    key = (hf_repo or "").strip().lower()
    if not key:
        yield
        return
    with _LOADS_IN_FLIGHT_LOCK:
        _LOADS_IN_FLIGHT[key] = _LOADS_IN_FLIGHT.get(key, 0) + 1
    try:
        yield
    finally:
        with _LOADS_IN_FLIGHT_LOCK:
            remaining = _LOADS_IN_FLIGHT.get(key, 1) - 1
            if remaining <= 0:
                _LOADS_IN_FLIGHT.pop(key, None)
            else:
                _LOADS_IN_FLIGHT[key] = remaining


# Chat loads in flight, repo-agnostic (local paths and safetensors included). The repo-keyed counter above answers the download manager; this one answers the GPU arbiter, which must know before llama-server spawns.
_CHAT_LOADS_IN_FLIGHT = 0


@contextlib.contextmanager
def chat_load_in_flight():
    """Track a chat load from before its download until the load call returns."""
    global _CHAT_LOADS_IN_FLIGHT
    with _LOADS_IN_FLIGHT_LOCK:
        _CHAT_LOADS_IN_FLIGHT += 1
    try:
        yield
    finally:
        with _LOADS_IN_FLIGHT_LOCK:
            _CHAT_LOADS_IN_FLIGHT = max(0, _CHAT_LOADS_IN_FLIGHT - 1)


def chat_load_active() -> bool:
    """Whether a chat load is in flight, spawned or not.

    ``is_active`` only covers a live llama-server process, which an HF load does not have
    until its GGUF finished downloading -- minutes during which the arbiter's evictor would
    find nothing to cancel and grant the GPU to a second workload.
    """
    with _LOADS_IN_FLIGHT_LOCK:
        return _CHAT_LOADS_IN_FLIGHT > 0


def zero_vram_chat_load(
    gpu_memory_mode: str,
    gpu_layers: int,
    extra_args: Optional[list] = None,
    needs_mmproj: bool = False,
    speculative_type: Optional[str] = None,
) -> bool:
    """Whether a GGUF load will hold no VRAM at all, decided from the request.

    A deliberate manual zero-offload load runs on the CPU and is launched with the GPUs hidden
    from the child (see ``_cpu_only_zero_offload``), so it neither needs the GPU arbiter nor
    should evict an image/video pipeline for it. This mirrors that launch-time mask so both
    agree: an mmproj, a GPU drafter, a device pin or a surviving tensor mode all keep the GPUs
    visible, and the mask is skipped on Vulkan. ``--mmproj`` and ``--model-draft`` are added by
    the backend rather than being present in ``extra_args``, so their intent is passed in.
    """
    if gpu_memory_mode != "manual" or gpu_layers != 0:
        return False
    # Any speculative mode may launch a GPU drafter and only the request's own knobs are known here, so treat every selection as GPU-bearing except
    # "off". Canonicalize first: the UI sends the literal "off", which a bare truthiness test read as "speculation requested".
    # ...and an ABSENT mode is not a quiet "off": every consumer resolves None to
    # "auto", where a sidecar can still be discovered and GPU-offloaded. Grouping None
    # with "off" let a defaulted load skip acquire_for(CHAT) while the launch-time mask
    # -- reading the FINISHED argv, drafter included -- kept the GPUs visible: no
    # arbiter, real VRAM, free to OOM a resident image pipeline.
    spec_mode = _canonicalize_spec_mode(speculative_type) or "auto"
    if needs_mmproj or spec_mode != "off":
        return False
    if LlamaCppBackend._is_vulkan_backend():
        return False
    return not LlamaCppBackend._zero_offload_keeps_gpu_visible(
        [str(arg) for arg in (extra_args or [])], os.environ
    )


def hf_gguf_load_in_flight(hf_repo: str) -> bool:
    """Return whether a GGUF load is active for hf_repo."""
    key = (hf_repo or "").strip().lower()
    if not key:
        return False
    with _LOADS_IN_FLIGHT_LOCK:
        return _LOADS_IN_FLIGHT.get(key, 0) > 0


def _with_gguf_load_marker(load: Callable):
    """Keep an HF repo marked for the full synchronous load call."""

    @functools.wraps(load)
    def wrapped(
        self,
        intent: GgufLoadIntent,
        load_cancel_event: Optional[threading.Event] = None,
    ):
        hf_repo = intent.hf_repo
        # Must not leak: polling outside a load needs LIVE VRAM, never this snapshot.
        with _load_probe_memo_scope(), gguf_load_in_flight(hf_repo):
            if hf_repo and _hub_download_blocks_gguf_load(
                hf_repo,
                intent.hf_variant,
                # Mirrors the download gate below EXACTLY, which is the whole point:
                # this load fetches the projector whenever the repo ships one and the
                # extras have not opted out, switch or no switch, because only the
                # file's metadata says whether it is an image tower or an audio
                # encoder. So the switch must not relax the interlock here. It did
                # briefly, and the two together let a vision-off load skip the 409 and
                # then write into the shared Hub cache next to a running download job,
                # which is the race this check exists to stop.
                require_mmproj = bool(
                    intent.is_vision and not extra_args_disable_mmproj(intent.extra_args)
                ),
                hf_token = intent.hf_token,
            ):
                raise RuntimeError(
                    f"'{hf_repo}' is currently being downloaded by the download manager"
                )
            return load(self, intent, load_cancel_event = load_cancel_event)

    return wrapped
