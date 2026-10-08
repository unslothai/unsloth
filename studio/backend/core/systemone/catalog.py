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
    # "laya" (rl_agent_config.json + encoder), "clef" (Qwen3.5 backbone + joint schema head), or
    # "gguf" (a GGUF_COMPANIONS entry llama.cpp serves, with no PyTorch form).
    layout: str = "laya"
    # "pytorch", or "llama.cpp" for the GGUF a Clef entry is served from (see laya_runtime._native_target).
    backend: str = "pytorch"
    # The served GGUF files of a local export, so a re-export is a different checkpoint to a resident server.
    revision: str | None = None
    # Display name where the frontend has none of its own.
    label: str | None = None

    @property
    def is_local(self) -> bool:
        return Path(self.source).expanduser().is_dir()


def _gguf(name: str, label: str, repo: str, description: str, download_bytes: int) -> Checkpoint:
    return Checkpoint(
        name, repo, None, description, download_bytes, "gguf", "llama.cpp", label = label
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
        Checkpoint(
            "clef-flash",
            "Cloudflare/clef-flash",
            None,
            "Cloudflare Clef-flash on Qwen3.5 9B: fast multimodal decisions, needs a GPU.",
            19_063_259_136,
            "clef",
        ),
        Checkpoint(
            "clef",
            "Cloudflare/clef",
            None,
            "Cloudflare Clef on Qwen3.8 27B: the most accurate decisions, needs a large GPU.",
            54_976_000_000,
            "clef",
        ),
        _gguf(
            "kev-0.8b",
            "Kev 0.8B",
            "ggml-org/Kev-0.8B-GGUF",
            "Kev 0.8B (Q8_0 GGUF, llama.cpp only): small and fast, text only, runs on CPU.",
            812_406_304,
        ),
        _gguf(
            "kev-4b",
            "Kev 4B",
            "ggml-org/Kev-4B-GGUF",
            "Kev 4B (Q8_0 GGUF, llama.cpp only): text only, a GPU is recommended.",
            4_483_801_504,
        ),
        _gguf(
            "kev-9b",
            "Kev 9B",
            "ggml-org/Kev-9B-GGUF",
            "Kev 9B (Q8_0 GGUF, llama.cpp only): text only, needs a GPU.",
            9_529_735_648,
        ),
        _gguf(
            "lev",
            "lev",
            "ggml-org/lev-GGUF",
            "Interfaze lev on Qwen3.5 4B (Q8_0 GGUF, llama.cpp only): text only, a GPU is recommended.",
            4_482_405_280,
        ),
        _gguf(
            "bespoke-nimble-9b-v3",
            "Bespoke Nimble 9B v3",
            "ggml-org/Bespoke-Nimble-9B-v3-GGUF",
            "Bespoke Nimble 9B v3 (Q8_0 GGUF, llama.cpp only): text only, needs a GPU. "
            "Non-commercial license (CC BY-NC 4.0).",
            9_527_503_392,
        ),
        _gguf(
            "openjev",
            "OpenJev",
            "ggml-org/OpenJev-GGUF",
            "OpenJev 27B (Q8_0 GGUF, llama.cpp only): text and images, needs a large GPU. "
            "Non-commercial license (CC BY-NC 4.0).",
            28_595_765_408 + 629_247_232,
        ),
        _gguf(
            "laya-gguf",
            "Laya GGUF",
            "ggml-org/Laya-GGUF",
            "Laya (Q8_0 GGUF, llama.cpp only): small, text only, runs on CPU.",
            449_397_600,
        ),
        _gguf(
            "julia-1",
            "Julia-1",
            "ggml-org/Julia-1-GGUF",
            "Supersonic Julia-1 (Q8_0 GGUF, llama.cpp only): the smallest, text only, runs on CPU.",
            168_166_496,
        ),
    )
}


@dataclass(frozen = True)
class GgufCompanion:
    repo: str
    revision: str
    model: str
    mmproj: str | None
    download_bytes: int
    # LFS sha256 of model, mmproj: the Hub cache names blobs by it, so a copy fetched at another revision is found.
    sha256: tuple[str, ...] = ()


# ggml-org GGUFs served by llama.cpp's /v1/systemone (b11443 and newer).
GGUF_COMPANIONS = {
    "clef-flash": GgufCompanion(
        "ggml-org/Clef-Flash-GGUF",
        "4a192915ef971886004b5b13294f2b4c7a7fc39d",
        "Clef-Flash-Q8_0.gguf",
        "mmproj-Clef-Flash-Q8_0.gguf",
        9_657_260_192 + 624_229_728,
        (
            "d7c352faf1bdd9ea24d0b9347e8eb1eb4bbadeff6c02383bf750215a74f2f1f1",
            "3fbc646617c56c35ba48e06f0fbe8693a83bde49eb31e2a3bf59ba0a308c9e25",
        ),
    ),
    "clef": GgufCompanion(
        "ggml-org/Clef-GGUF",
        "63840a1a68cb7084c88610cffc328509356b04cb",
        "Clef-Q8_0.gguf",
        "mmproj-Clef-Q8_0.gguf",
        28_732_215_360 + 629_247_424,
        (
            "07c6410af7011e0e56873a3b0b3f4ad9e31fb176f0ca80d6fd1525fe6f036548",
            "0d901ea999ae122ba0afeb4a3f4c4ac9a90baa07f1d120df9c350082db1190a8",
        ),
    ),
    **{
        name: GgufCompanion(repo, revision, model, mmproj, CHECKPOINTS[name].download_bytes, sha256)
        for name, repo, revision, model, mmproj, sha256 in (
            (
                "kev-0.8b",
                "ggml-org/Kev-0.8B-GGUF",
                "e551e319d483ff57e1ff208924b349d397cffc1c",
                "Kev-0.8B-Q8_0.gguf",
                None,
                ("27278f34eb3273bceea4c053dc50dd61a5161da21a718c4aacdf8fd5830771d0",),
            ),
            (
                "kev-4b",
                "ggml-org/Kev-4B-GGUF",
                "d924f2e2c3872da8b8aaf3eb4453b4126deceb79",
                "Kev-4B-Q8_0.gguf",
                None,
                ("7c2ebed90560522c2801389db482ac1dc4c36d828f201f6074c1d60e433948da",),
            ),
            (
                "kev-9b",
                "ggml-org/Kev-9B-GGUF",
                "ec2bbfe6620218aee2e01cc93bb78dee2a96ed58",
                "Kev-9B-Q8_0.gguf",
                None,
                ("d30b225bfdc985d1856bb04a990be76008d0cf40dccbf78938e3069744bc614c",),
            ),
            (
                "lev",
                "ggml-org/lev-GGUF",
                "3e9286a79ae857b4e1de051c92fab6dc574581ce",
                "lev-Q8_0.gguf",
                None,
                ("c6b70833a9ec59c67bda2940f4047c2e1bdc066b4a6f8aec3c36d62b6ea34eef",),
            ),
            (
                "bespoke-nimble-9b-v3",
                "ggml-org/Bespoke-Nimble-9B-v3-GGUF",
                "a72bdbadb355ca3014f7f9d2585380917cc59971",
                "Bespoke-Nimble-9B-v3-Q8_0.gguf",
                None,
                ("ad484077ad28c1ba0644509b81e80268f8fd39e9d218415638a988caf3c6e66c",),
            ),
            (
                "openjev",
                "ggml-org/OpenJev-GGUF",
                "10840f375658dea7afc5ff4711127bca8218b560",
                "OpenJev-Q8_0.gguf",
                "mmproj-OpenJev-Q8_0.gguf",
                (
                    "5e5574cfeb9145809d3ebe3e854761d62a947033d3f76af4c4b1dd8562409412",
                    "e372cdbf59fdd6bd2504cb64c988b31c7a42ac406a8f711df4b7a7acd9216f1e",
                ),
            ),
            (
                "laya-gguf",
                "ggml-org/Laya-GGUF",
                "22265007700297ba9e128297e82540cf28c5d7d4",
                "Laya-Q8_0.gguf",
                None,
                ("c06528c5746d3bb8baa72a27938be95abbfd0b226f8471e8a9e365ed0bb066d2",),
            ),
            (
                "julia-1",
                "ggml-org/Julia-1-GGUF",
                "16fee17949206fbf58da9347daea44d792a81211",
                "Julia-1-Q8_0.gguf",
                None,
                ("1ea6a7e87156eeeda88cb7a36a61265b37ba7b993897b7289b99aea5b5e47069",),
            ),
        )
    },
}

# Names TypeSafe's and OpenJev's SDKs send by default, so an unmodified client reaches the configured model.
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


CLEF_NEEDS_GPU_SETTING = (
    "Clef models on the PyTorch runtime run on the GPU only, and the Decision API device is set "
    "to CPU. Switch it to GPU, serve the model's GGUF through llama.cpp, or use a Laya model."
)


def clef_unavailable_reason(wait: bool = True) -> str | None:
    """Why PyTorch Clef cannot serve: no GPU, or a CPU device chosen. Training uses clef_unsupported_reason."""
    if (reason := clef_unsupported_reason(wait)) is not None:
        return reason
    from utils.systemone_settings import device_chosen, get_device

    return None if not device_chosen() or get_device() == "gpu" else CLEF_NEEDS_GPU_SETTING


def is_fine_tune_name(name: object) -> bool:
    return isinstance(name, str) and name.startswith(FINE_TUNE_PREFIXES)


def _owner_outputs() -> Path:
    from utils.account_context import OWNER, run_as
    from utils.paths import outputs_root
    return run_as(OWNER, outputs_root).resolve()


def _fine_tune_in(root: Path, folder_name: str) -> Checkpoint | None:
    from utils.models.model_config import clef_folder_kind

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
    if clef_folder_kind(folder) is not None:
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
    if name in CHECKPOINTS:
        return CHECKPOINTS[name]
    from utils.account_context import is_owner_context

    checkpoint = fine_tune(name)
    # Other accounts reach only the fine-tune the owner configured, not the owner's other outputs.
    if checkpoint is None or is_owner_context() or checkpoint == default_checkpoint():
        return checkpoint
    return None
