# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""System One decision models Studio can serve on ``POST /v1/systemone``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LAYA_REPO = "convaiinnovations/laya"
# Recipe for Clef folders that have no model defaults of their own (local copies, fine-tunes).
CLEF_DEFAULTS_REPO = "Cloudflare/clef-flash"


@dataclass(frozen = True)
class Checkpoint:
    name: str
    source: str
    subfolder: str | None
    description: str
    download_bytes: int = 0
    # "laya" (rl_agent_config.json + encoder) or "clef" (Qwen3.5 backbone + joint schema head).
    layout: str = "laya"

    @property
    def is_local(self) -> bool:
        return Path(self.source).expanduser().is_dir()


@dataclass(frozen = True)
class ClefCheckpoint(Checkpoint):
    layout: str = "clef"
    revision: str = ""
    files: tuple[str, ...] = ()


def _clef_files(shards: int) -> tuple[str, ...]:
    return (
        "config.json",
        "generation_config.json",
        "joint_head_config.json",
        "joint_head.safetensors",
        "model.safetensors.index.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
        *(f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)),
    )


CHECKPOINTS = {
    c.name: c
    for c in (
        Checkpoint(
            "laya-multilingual",
            LAYA_REPO,
            "multilingual",
            "Laya on mmBERT-base: 100+ languages, 1024-token context.",
            678_201_636,
        ),
        Checkpoint(
            "laya-english",
            LAYA_REPO,
            None,
            "Laya on ModernBERT-large: English, 512-token context.",
            846_195_574,
        ),
        Checkpoint(
            "laya-typed-decisions",
            LAYA_REPO,
            "typed-decisions",
            "Laya English fine-tuned on four typed-decision workflows, 1024-token context.",
            846_195_716,
        ),
        ClefCheckpoint(
            "clef-flash",
            "Cloudflare/clef-flash",
            None,
            "Clef Flash · 9B · text and images. About 20 GB GPU memory or 40 GB CPU RAM (slow). Video and audio are not served.",
            19_083_200_000,
            revision = "17f0b0ad64efb65d273590632833508766b2aae6",
            files = _clef_files(4),
        ),
        ClefCheckpoint(
            "clef",
            "Cloudflare/clef",
            None,
            "Clef · 27B · text and images. About 56 GB GPU memory or 112 GB CPU RAM (slow). Video and audio are not served.",
            54_989_600_000,
            revision = "2f3de3dd85f379784083b0814d997ab627200f0c",
            files = _clef_files(12),
        ),
    )
}

NATIVE_CHECKPOINTS = {
    c.name: c
    for c in (
        ClefCheckpoint(
            "clef-flash",
            "ggml-org/Clef-Flash-GGUF",
            None,
            "Clef Flash · Q8 · native text decisions",
            9_657_260_096,
            revision = "4a7a08c09bc63baf043b62b5ba89dd67a0357d95",
            files = ("Clef-Flash-Q8_0.gguf",),
        ),
        ClefCheckpoint(
            "clef",
            "ggml-org/Clef-GGUF",
            None,
            "Clef · Q8 · native text decisions",
            28_732_215_264,
            revision = "5f70656b6670c65eb85ad07a11efe211b5f211bd",
            files = ("Clef-Q8_0.gguf",),
        ),
    )
}

# SDK default aliases reach the installation's configured model.
DEFAULT_ALIASES = frozenset({"default", "laya", "jev-latest", "jev-preview", "openjev-latest"})
LOCAL_NAME = "laya-local"
CONNECTION_PREFIX = "connection:"
FINE_TUNE_PREFIX = "laya-ft:"
CLEF_FINE_TUNE_PREFIX = "clef-ft:"
FINE_TUNE_PREFIXES = (FINE_TUNE_PREFIX, CLEF_FINE_TUNE_PREFIX)


CLEF_NEEDS_GPU = (
    "Clef models need an NVIDIA or AMD GPU; this machine has none. Use a Laya model instead."
)


def clef_unsupported_reason(wait: bool = True) -> str | None:
    # ROCm reports DeviceType.CUDA too. A failed probe answers None: detection only ever widens.
    # wait=False reads only a finished detection, so a settings read never waits on torch import.
    try:
        from utils.hardware import hardware

        device = hardware.get_device() if wait else hardware.DEVICE
        if device is None:
            return None
        return None if device == hardware.DeviceType.CUDA else CLEF_NEEDS_GPU
    except Exception:
        return None


def is_fine_tune_name(name: object) -> bool:
    return isinstance(name, str) and name.startswith(FINE_TUNE_PREFIXES)


def _owner_outputs() -> Path:
    from utils.account_context import OWNER, run_as
    from utils.paths import outputs_root
    return run_as(OWNER, outputs_root).resolve()


def _fine_tune_in(root: Path, folder_name: str) -> Checkpoint | None:
    from utils.models.model_config import CLEF_MARKERS

    from .laya_runtime import is_cached

    # Dot folders include runs staged for deletion (.<name>.deleting-<id>).
    if not folder_name or folder_name.startswith("."):
        return None
    folder = root / folder_name
    try:
        if folder.resolve().parent != root:
            return None
    except (OSError, RuntimeError, ValueError):
        # A NUL byte or a symlink loop in a caller's name.
        return None
    if all((folder / name).is_file() for name in CLEF_MARKERS):
        checkpoint = Checkpoint(
            CLEF_FINE_TUNE_PREFIX + folder_name,
            str(folder),
            None,
            "Clef fine-tuned in Studio.",
            layout = "clef",
        )
    else:
        checkpoint = Checkpoint(
            FINE_TUNE_PREFIX + folder_name, str(folder), None, "Laya fine-tuned in Studio."
        )
    return checkpoint if is_cached(checkpoint) else None


def fine_tune(name: str) -> Checkpoint | None:
    # Either prefix finds the run; the answer carries the one for the folder's layout.
    if not is_fine_tune_name(name):
        return None
    folder_name = name.partition(":")[2]
    if "/" in folder_name or "\\" in folder_name:
        return None
    return _fine_tune_in(_owner_outputs(), folder_name)


def fine_tunes() -> list[Checkpoint]:
    root = _owner_outputs()
    try:
        folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    return [c for name in folders if (c := _fine_tune_in(root, name)) is not None]


@dataclass(frozen = True)
class Connection:
    provider_id: str
    model: str

    @property
    def name(self) -> str:
        return f"{CONNECTION_PREFIX}{self.provider_id}:{self.model}"


def parse_connection(value: object) -> Connection | None:
    if not isinstance(value, str) or not value.startswith(CONNECTION_PREFIX):
        return None
    provider_id, _, model = value[len(CONNECTION_PREFIX) :].partition(":")
    return Connection(provider_id, model) if provider_id and model else None


LISTED_DECISION_MODELS: dict[tuple[str, str], tuple[float, list[str]]] = {}


def decision_models(row: dict) -> list[str] | None:
    from core.inference.providers import answers_decisions_only

    if answers_decisions_only(row["provider_type"], row.get("api_type")):
        return row["models"]
    if row["provider_type"] == "openrouter":
        return LISTED_DECISION_MODELS.get((row["id"], row["updated_at"]), (0.0, []))[1]
    return None


def decision_connections() -> list[tuple[dict, list[str]]]:
    # Connections are per account, the Decision API is installation-wide: read the owner's.
    from storage.providers_db import list_providers
    from utils.account_context import OWNER, run_as
    return [
        (row, models)
        for row in run_as(OWNER, list_providers)
        if row["is_enabled"] and (models := decision_models(row))
    ]


def default_checkpoint() -> Checkpoint | Connection:
    from utils.systemone_settings import get_model

    configured = get_model()
    if configured in CHECKPOINTS:
        return CHECKPOINTS[configured]
    if clef := next(
        (
            c
            for c in CHECKPOINTS.values()
            if isinstance(c, ClefCheckpoint) and c.source == configured
        ),
        None,
    ):
        return clef
    if connection := parse_connection(configured):
        return connection
    if (checkpoint := fine_tune(configured)) is not None:
        return checkpoint
    subfolder = os.environ.get("UNSLOTH_SYSTEMONE_SUBFOLDER", "").strip() or None
    from utils.models.model_config import CLEF_MARKERS

    folder = Path(configured).expanduser()
    if subfolder is None and all((folder / name).is_file() for name in CLEF_MARKERS):
        return Checkpoint(LOCAL_NAME, configured, None, "Local Clef checkpoint.", layout = "clef")
    return Checkpoint(LOCAL_NAME, configured, subfolder, "Local Laya checkpoint.")


def resolve(model: str) -> Checkpoint | Connection | None:
    name = (model or "").strip()
    if name in DEFAULT_ALIASES:
        return default_checkpoint()
    if name == LOCAL_NAME or name.startswith(CONNECTION_PREFIX):
        checkpoint = default_checkpoint()
        return checkpoint if checkpoint.name == name else None
    checkpoint = CHECKPOINTS.get(name) or next(
        (c for c in CHECKPOINTS.values() if isinstance(c, ClefCheckpoint) and c.source == name),
        None,
    )
    if checkpoint is not None:
        return checkpoint
    from utils.account_context import is_owner_context

    checkpoint = fine_tune(name)
    # Other accounts reach only the fine-tune the owner configured, not the owner's other outputs.
    if checkpoint is None or is_owner_context() or checkpoint == default_checkpoint():
        return checkpoint
    return None
