# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opportunistic checkpoint I/O; model loading and dataset preparation stay on the main thread."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextvars import copy_context
from pathlib import Path

from loggers import get_logger
from utils.account_context import account_thread

logger = get_logger(__name__)
_SHARD = re.compile(r"model-[0-9]+-of-[0-9]+\.safetensors\Z")


def prefetch_repo(
    config,
    trainer,
    *,
    load_in_4bit,
    capabilities,
    offline = False,
    formatting_can_fork = False,
):
    """Only routes whose ordinary root checkpoint can be resolved without loading model code."""
    name = config.get("model_name", "")
    if (
        offline
        or formatting_can_fork
        or trainer.should_stop
        or not isinstance(name, str)
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", name)
        or Path(name).expanduser().exists()
        or config.get("training_type", "LoRA/QLoRA") != "LoRA/QLoRA"
        or any(
            config.get(key)
            for key in (
                "model_snapshot_path",
                "model_revision",
                "resume_from_checkpoint",
                "require_exact_resume_resources",
                "require_exact_model_resource",
                "trust_remote_code",
                "is_embedding",
                "is_decision",
                "is_dataset_audio",
            )
        )
        or getattr(trainer, "is_audio", False)
        or getattr(trainer, "is_audio_vlm", False)
        or getattr(trainer, "_audio_type", None)
        or getattr(trainer, "diffusion_lm_profile", None)
        or not getattr(trainer, "_audio_type_known", False)
        or capabilities is None
    ):
        return None
    allow_bnb = getattr(capabilities, "ALLOW_BITSANDBYTES", None)
    allow_prequant = getattr(capabilities, "ALLOW_PREQUANTIZED_MODELS", None)
    if not isinstance(allow_bnb, bool) or allow_prequant is not True:
        # The loader can strip/remap prequants on unsupported stacks; do not speculate.
        return None
    from utils.models.unsloth_mirror import mirror_lookup_available, unsloth_public_mirror

    if not mirror_lookup_available():
        return None
    quantized = load_in_4bit() if callable(load_in_4bit) else load_in_4bit
    return unsloth_public_mirror(name, bool(quantized and allow_bnb)) or name


def checkpoint_files(files, read_json):
    """Pick one standard safetensors representation, never all weights/quantizations in a repo."""
    files = set(files)
    if "adapter_config.json" in files or "config.json" not in files:
        return []
    config = read_json("config.json")
    if (
        not isinstance(config, dict)
        or config.get("auto_map")
        or config.get("base_model_name_or_path")
    ):
        return []
    if "model.safetensors" in files:
        return ["model.safetensors"]
    index_name = "model.safetensors.index.json"
    if index_name not in files:
        return []
    index = read_json(index_name)
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        return []
    if any(not isinstance(name, str) for name in weights.values()):
        return []
    shards = set(weights.values())
    if len(shards) > 1024 or any(
        not isinstance(name, str) or not _SHARD.fullmatch(name) or name not in files
        for name in shards
    ):
        return []
    return sorted(shards)


class ModelWeightPrefetch:
    def __init__(self, repo, *, token, cache_dir, emit, should_stop):
        self.repo, self.token, self.cache_dir = repo, token, str(cache_dir)
        self._emit, self._should_stop = emit, should_stop
        self._cancel = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._watchdog_stop = None

    def _send(self, kind, **kwargs):
        with self._lock:
            if not self._closed.is_set():
                self._emit({"type": kind, **kwargs, "ts": time.time()})

    def _stopped(self):
        return self._cancel.is_set() or self._should_stop()

    def _download(self):
        from huggingface_hub import HfApi, hf_hub_download
        from utils.hf_xet_fallback import start_watchdog

        info = HfApi().model_info(self.repo, token = self.token)
        if self._stopped():
            return
        commit = info.sha
        if not isinstance(commit, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
            return
        self._send("model_load_resolved", repo_id = self.repo)
        self._watchdog_stop = start_watchdog(
            repo_ids = [self.repo],
            on_stall = lambda message: self._send("stall", message = message),
            xet_disabled = os.environ.get("HF_HUB_DISABLE_XET") == "1",
        )

        def fetch(filename):
            if self._stopped():
                raise InterruptedError("Training model prefetch cancelled")
            return hf_hub_download(
                repo_id = self.repo,
                filename = filename,
                revision = commit,
                token = self.token,
                cache_dir = self.cache_dir,
            )

        def read_json(filename):
            path = Path(fetch(filename))
            if path.stat().st_size > 5 * 1024**2:
                raise ValueError(
                    "Unfamiliar checkpoint metadata; leave selection to the model loader"
                )
            return json.loads(path.read_text(encoding = "utf-8"))

        files = checkpoint_files([sibling.rfilename for sibling in info.siblings or ()], read_json)
        for filename in files:
            fetch(filename)

    def _run(self):
        try:
            if not self._stopped():
                self._download()
        except Exception as error:  # noqa: BLE001 -- the authoritative loader handles failures later
            logger.info("Model checkpoint prefetch skipped for %s: %s", self.repo, error)
        finally:
            if self._watchdog_stop is not None:
                self._watchdog_stop.set()

    def start(self):
        # Account identity and other scopes (cache/download/security) must survive the new thread.
        context = copy_context()
        self._send("model_load_started")
        self._thread = account_thread(target = lambda: context.run(self._run), daemon = True)
        self._thread.start()

    def finish(self, *, wait):
        try:
            while wait and self._thread.is_alive() and not self._should_stop():
                self._thread.join(timeout = 0.2)
        finally:
            self._cancel.set()
            if self._watchdog_stop is not None:
                self._watchdog_stop.set()
            # A Hub call still exiting on Stop/error must not emit into the next phase.
            with self._lock:
                self._closed.set()
                self._emit({"type": "model_load_completed", "ts": time.time()})


def prepare_dataset_with_prefetch(load_dataset, prefetch):
    """Do not consume streaming rows, copy datasets, or move tokenizer/GPU work into a thread."""
    if prefetch is None:
        return load_dataset()
    try:
        prefetch.start()
    except Exception as error:  # noqa: BLE001 -- an optimization cannot prevent the normal load
        logger.info("Could not start model checkpoint prefetch: %s", error)
        prefetch.finish(wait = False)
        return load_dataset()
    ready = False
    try:
        result = load_dataset()
        ready = result is not None and result[0] is not None
        return result
    finally:
        prefetch.finish(wait = ready)
