# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The provider contract for `unsloth deploy`.

The command layer (commands/deploy.py) talks only to `Provider`, so adding a
cloud means implementing it in a new ``<name>_client.py`` and registering it in
``provider.py``. SSH, pausing, and local-model upload are opt-in capabilities a
provider declares with a flag, so a cloud that lacks them still works.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from unsloth_cli.deploy import DeployError, Gpu, SshTarget, StagedModel


# Whether an option is needed to authenticate (always) or only to upload a local model.
NEEDED_FOR_AUTH = "auth"
NEEDED_FOR_LOCAL_MODEL = "local_model"


@dataclass(frozen = True)
class Option:
    """A credential or setting a provider reads. The command layer resolves it
    from --provider-opt > env var > saved config and passes it to `auth()`."""
    key: str                          # dict key the provider reads, e.g. "api_key"
    env: str                          # env var checked, e.g. "RUNPOD_API_KEY"
    help: str                         # shown when prompting / in error text
    secret: bool = False              # hide prompt input; e.g. tokens, S3 secrets
    required: bool = True             # block the deploy if unresolved
    needed_for: str = NEEDED_FOR_AUTH  # NEEDED_FOR_AUTH or NEEDED_FOR_LOCAL_MODEL


def _unsupported(provider: "Provider", what: str, hint: str = "") -> DeployError:
    msg = f"{provider.name} does not support {what}."
    return DeployError(f"{msg} {hint}".rstrip())


class Provider(ABC):
    """A cloud that `unsloth deploy` can launch Unsloth Studio on.

    Implement the abstract methods and set the capability flags for any optional
    methods you override; the command layer routes on the flags, never the class.
    """

    name: str = "provider"

    # Capabilities. The command layer checks these instead of provider internals.
    supports_ssh: bool = False           # get_ssh() returns a reachable target
    supports_pause: bool = False         # pause() suspends without terminating
    supports_local_model: bool = False   # stage_local_model() uploads local weights
    reports_stock: bool = True           # list_gpus carries a live per-GPU stock band

    # One-line caveat printed after a successful deploy (e.g. an auto-stop limit).
    deploy_note: str = ""

    @classmethod
    def option_schema(cls) -> list[Option]:
        """Options this provider reads. Default: none (auth uses ambient creds)."""
        return []

    @abstractmethod
    def auth(self, options: dict[str, str]) -> None:
        """Authenticate from resolved `options` (keyed by `Option.key`). Raise
        DeployError if a credential is missing or the SDK isn't installed."""

    @abstractmethod
    def list_gpus(self, min_vram_gb: int = 0) -> list[Gpu]:
        """On-demand GPUs with at least `min_vram_gb` of VRAM, cheapest first.
        Enumerate what the cloud actually offers -- no hardcoded allow/denylist."""

    @abstractmethod
    def create_instance(
        self,
        *,
        name: str,
        gpu: Gpu,
        image: str,
        http_port: int,
        env: dict[str, str],
        disk_gb: int,
        ssh_port: Optional[int] = None,
        staged: Optional[StagedModel] = None,
    ) -> str:
        """Start a container running `image`, expose `http_port` for Studio, and
        return an instance id. If `staged` is set, attach that storage and honor
        ``staged.placement``. Providers without SSH ignore `ssh_port`."""

    @abstractmethod
    def wait_ready(self, instance_id: str, timeout_s: int) -> None:
        """Block until the instance is actually serving (not just scheduled)."""

    @abstractmethod
    def endpoint_url(self, instance_id: str, http_port: int) -> str:
        """Base http(s) URL that reaches `http_port` on the instance."""

    @abstractmethod
    def terminate(self, instance_id: str) -> None:
        """Destroy the instance so compute billing stops."""

    def pause(self, instance_id: str) -> None:
        """Suspend without destroying (only when `supports_pause`)."""
        raise _unsupported(self, "pausing an instance", "Use terminate instead.")

    def get_ssh(self, instance_id: str) -> SshTarget:
        """SSH target for the instance (only when `supports_ssh`)."""
        raise _unsupported(self, "SSH access")

    def stage_local_model(
        self,
        local_path: Path,
        *,
        gpu: Gpu,
        log: Callable[[str], None] = lambda _msg: None,
    ) -> StagedModel:
        """Upload local weights to provider storage and return where the instance
        reads them (only when `supports_local_model`). `log` streams progress;
        clean up any created storage before raising on failure."""
        raise _unsupported(
            self, "uploading a local model",
            "Pass a Hugging Face id with --model, or load from the Studio UI.",
        )

    def delete_storage(self, storage_id: str) -> None:
        """Delete storage created by `stage_local_model` so it stops billing."""
        raise _unsupported(self, "deletable storage")
