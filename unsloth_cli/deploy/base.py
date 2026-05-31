# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The provider contract for `unsloth deploy`.

A provider turns "run Unsloth Studio in the cloud" into calls against one cloud's
API. The command layer (commands/deploy.py) talks ONLY to this interface, so
adding a cloud means implementing `Provider` in a new ``<name>_client.py`` and
registering it in ``provider.py`` -- with no changes to the command layer.

Clouds are not all shaped alike. RunPod has a live GPU marketplace, SSH-reachable
pods, and datacenter-pinned network volumes; Modal has fixed-price GPUs, no SSH,
and its own volume API. So the required contract is deliberately small, and the
rest is opt-in: a provider sets a capability flag and implements its method, or
leaves the default that raises "unsupported" and the command layer routes around
it (hides the SSH line, refuses a local-model upload with a helpful message,
etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from unsloth_cli.deploy import DeployError, Gpu, SshTarget, StagedModel


# When the command layer must have an option resolved. Auth-time options (API
# tokens) are always required; local-model options (storage creds) only when the
# user is actually uploading local weights.
NEEDED_FOR_AUTH = "auth"
NEEDED_FOR_LOCAL_MODEL = "local_model"


@dataclass(frozen = True)
class Option:
    """One credential or setting a provider reads.

    The command layer resolves each option from ``--provider-opt`` > env var >
    saved config, prompts for any still missing (when interactive), then persists
    the result so later runs need no env vars. A provider never reads os.environ
    itself -- it declares its options here and receives the resolved values in
    `auth()`."""
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

    Implement the abstract methods (the required core) and set the capability
    flags for any optional methods you override. The command layer checks the
    flags -- it never inspects the concrete class -- so a provider that supports,
    say, neither SSH nor local-model upload still works end to end for a Hugging
    Face model loaded from the UI.
    """

    name: str = "provider"

    # Capabilities. The command layer checks these instead of provider internals.
    supports_ssh: bool = False           # get_ssh() returns a reachable target
    supports_pause: bool = False         # pause() suspends without terminating
    supports_local_model: bool = False   # stage_local_model() uploads local weights

    # --- credentials / configuration -----------------------------------------

    @classmethod
    def option_schema(cls) -> list[Option]:
        """Options this provider reads, for the command layer to resolve, prompt
        for, and persist. Default: none (auth uses ambient credentials)."""
        return []

    @abstractmethod
    def auth(self, options: dict[str, str]) -> None:
        """Authenticate from already-resolved `options` (keyed by `Option.key`).
        Raise DeployError with actionable text if a credential is missing or the
        provider SDK is not installed."""

    # --- GPU catalog ----------------------------------------------------------

    @abstractmethod
    def list_gpus(self, min_vram_gb: int = 0) -> list[Gpu]:
        """On-demand GPUs with at least `min_vram_gb` of VRAM, cheapest first.

        A live marketplace (RunPod) queries current price and stock; a
        fixed-price cloud (Modal) returns its supported GPU types with
        ``stock=None``. Either way, enumerate what the cloud actually offers --
        do not maintain a hardcoded allow/denylist."""

    # --- compute lifecycle ----------------------------------------------------

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
        """Start a container running `image` and exposing `http_port` for Studio.

        If `staged` is set, wire that storage into the instance and honor
        ``staged.placement`` (e.g. pin to the volume's datacenter). Providers
        without SSH ignore `ssh_port`. Return a provider instance id; raise
        DeployError on failure (the command layer will not have anything to
        clean up unless stage_local_model already created storage)."""

    @abstractmethod
    def wait_ready(self, instance_id: str, timeout_s: int) -> None:
        """Block until the instance's container is actually serving (not merely
        scheduled), or raise DeployError on timeout/terminal failure."""

    @abstractmethod
    def endpoint_url(self, instance_id: str, http_port: int) -> str:
        """Base http(s) URL that reaches `http_port` on the instance."""

    @abstractmethod
    def terminate(self, instance_id: str) -> None:
        """Destroy the instance so compute billing stops."""

    def pause(self, instance_id: str) -> None:
        """Suspend the instance without destroying it. Only call when
        `supports_pause` is True."""
        raise _unsupported(self, "pausing an instance", "Use terminate instead.")

    # --- optional: SSH --------------------------------------------------------

    def get_ssh(self, instance_id: str) -> SshTarget:
        """SSH target for the instance. Only call when `supports_ssh` is True."""
        raise _unsupported(self, "SSH access")

    # --- optional: local-model staging ---------------------------------------

    def stage_local_model(
        self,
        local_path: Path,
        *,
        gpu: Gpu,
        log: Callable[[str], None] = lambda _msg: None,
    ) -> StagedModel:
        """Upload local weights to provider storage and return where the instance
        will read them. Only call when `supports_local_model` is True. `log`
        streams human-readable progress (volume creation, upload). On failure,
        clean up any storage created here before raising, so nothing is left
        billing."""
        raise _unsupported(
            self, "uploading a local model",
            "Pass a Hugging Face id with --model, or load from the Studio UI.",
        )

    def delete_storage(self, storage_id: str) -> None:
        """Delete storage created by `stage_local_model` so it stops billing."""
        raise _unsupported(self, "deletable storage")
