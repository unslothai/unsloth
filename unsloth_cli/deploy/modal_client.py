# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Modal provider: GPU sandboxes running Unsloth Studio, with local-model staging
onto a Modal volume. Implements the `Provider` contract in deploy/base.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from unsloth_cli.deploy import DeployError, Gpu, StagedModel
from unsloth_cli.deploy.base import Option, Provider


APP_NAME = "unsloth-studio"          # persisted Modal app sandboxes are created under
MODEL_MOUNT_DIR = "/model"           # where a staged volume is mounted in the sandbox
SANDBOX_TIMEOUT_S = 86400            # Modal's hard cap on a sandbox's lifetime (24h)
POLL_INTERVAL_S = 3

# The official prebaked Studio image (pulled, not built). Runs as user `unsloth` with
# `unsloth` on PATH (/opt/venv); its built frontend is whited out, so we serve the API
# only (--api-only) -- all the deploy needs. The web UI would need the image entrypoint.
STUDIO_IMAGE = "unsloth/unsloth"
STUDIO_HOME = "/home/unsloth/.unsloth/studio"   # UNSLOTH_STUDIO_HOME we pin (writable)

# Modal has no live pricing/stock API, so this is its published catalog.
MODAL_GPUS = [  # (gpu id / spec, display name, vram_gb, usd/hr)
    ("T4",        "NVIDIA T4",        16, 0.59),
    ("L4",        "NVIDIA L4",        24, 0.80),
    ("A10G",      "NVIDIA A10G",      24, 1.10),
    ("L40S",      "NVIDIA L40S",      48, 1.95),
    ("A100",      "NVIDIA A100 40GB", 40, 2.10),
    ("A100-80GB", "NVIDIA A100 80GB", 80, 2.50),
    ("H100",      "NVIDIA H100",      80, 3.95),
    ("H200",      "NVIDIA H200",     141, 4.54),
    ("B200",      "NVIDIA B200",     180, 6.25),
]


class Modal(Provider):
    name = "modal"

    supports_ssh = False             # Modal exposes sb.exec, not an SSH target
    supports_pause = False           # no suspend/resume
    supports_local_model = True      # uploads onto a Modal volume
    reports_stock = False            # fixed-capacity cloud; no live stock signal
    deploy_note = "Modal stops this instance automatically after 24h (max sandbox lifetime)."

    def __init__(self):
        self._modal = None
        self._app = None

    @classmethod
    def option_schema(cls) -> list[Option]:
        # None: the Modal SDK discovers credentials itself (MODAL_TOKEN_ID/SECRET env
        # vars, else ~/.modal.toml written by `modal token new`).
        return []

    def auth(self, options: dict[str, str]) -> None:
        try:
            import modal
        except ImportError as e:
            raise DeployError(
                "The 'modal' package is required for `unsloth deploy --provider modal`.\n"
                "Install it with:\n"
                "    pip install unsloth[deploy]"
            ) from e
        try:
            # App.lookup authenticates with the SDK's ambient credentials and returns a
            # persisted app, so sandboxes outlive this CLI process (ephemeral app.run()
            # would tear them down). Raises if no valid credentials are found.
            self._app = modal.App.lookup(APP_NAME, create_if_missing = True)
        except Exception as e:
            raise DeployError(
                "Modal authentication failed. Run `modal token new`, or set "
                "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET."
            ) from e
        self._modal = modal

    def list_gpus(self, min_vram_gb: int = 0) -> list[Gpu]:
        gpus = [
            Gpu(id = gpu_id, name = name, vram_gb = vram_gb, cost_per_hour_usd = price, stock = None)
            for gpu_id, name, vram_gb, price in MODAL_GPUS if vram_gb >= min_vram_gb
        ]
        gpus.sort(key = lambda g: (g.cost_per_hour_usd, g.vram_gb))
        return gpus

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
        # We pull the prebaked unsloth/unsloth (see _studio_image) and start Studio
        # ourselves, so the passed `image` tag, disk_gb, and ssh_port don't apply.
        sdk = self._sdk()
        volumes = {}
        if staged and staged.storage_id:
            volumes[MODEL_MOUNT_DIR] = sdk.Volume.from_name(staged.storage_id)
        try:
            # Stream the first-time image pull (~13 GB) so the deploy isn't silent.
            with sdk.enable_output():
                sb = sdk.Sandbox.create(
                    "/bin/sh", "-lc", self._start_command(http_port),
                    app = self._app,
                    name = name,
                    image = self._studio_image(),
                    gpu = gpu.id,
                    # TLS tunnel: the admin password travels over this URL, so never plaintext.
                    encrypted_ports = [http_port],
                    secrets = [sdk.Secret.from_dict(env)],   # admin password, not a plaintext arg
                    volumes = volumes,
                    timeout = SANDBOX_TIMEOUT_S,
                )
        except Exception as e:
            raise DeployError(f"Modal sandbox create failed: {e}") from e
        return sb.object_id

    def _start_command(self, http_port: int) -> str:
        """Shell to launch Studio. Studio ignores UNSLOTH_ADMIN_PASSWORD and seeds its
        admin user from a .bootstrap_password file (like the RunPod image's start.sh),
        so we pin UNSLOTH_STUDIO_HOME and write the password (a Modal secret) there
        first. --api-only because the image whites out the built web frontend."""
        pw_file = f"{STUDIO_HOME}/auth/.bootstrap_password"
        return (
            f"export UNSLOTH_STUDIO_HOME={STUDIO_HOME} && "
            f"mkdir -p {STUDIO_HOME}/auth && "
            f'printf %s "$UNSLOTH_ADMIN_PASSWORD" > {pw_file} && '
            f"chmod 600 {pw_file} && "
            f"unsloth studio --api-only -p {http_port} -H 0.0.0.0"
        )

    def _studio_image(self):
        # Clear the image entrypoint so our command runs directly (the entrypoint also
        # launches Jupyter/SSH, which a Studio deploy doesn't need).
        return self._sdk().Image.from_registry(STUDIO_IMAGE).entrypoint([])

    def wait_ready(self, instance_id: str, timeout_s: int) -> None:
        sdk = self._sdk()
        sb = sdk.Sandbox.from_id(instance_id)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            code = sb.poll()
            if code is not None:
                # A non-None exit code means the container stopped before it served.
                raise DeployError(
                    f"Modal sandbox {instance_id} exited before serving (code {code})."
                )
            try:
                if sb.tunnels(timeout = POLL_INTERVAL_S):
                    return  # tunnel resolved -> the network path is up
            except Exception:
                pass  # tunnel not ready yet; keep polling until the deadline
            time.sleep(POLL_INTERVAL_S)
        raise DeployError(f"Modal sandbox {instance_id} did not start within {timeout_s}s")

    def endpoint_url(self, instance_id: str, http_port: int) -> str:
        sb = self._sdk().Sandbox.from_id(instance_id)
        try:
            return sb.tunnels()[http_port].url
        except Exception as e:
            raise DeployError(
                f"Modal tunnel for port {http_port} is unavailable: {e}"
            ) from e

    def terminate(self, instance_id: str) -> None:
        try:
            self._sdk().Sandbox.from_id(instance_id).terminate()
        except Exception as e:
            raise DeployError(f"Modal terminate failed: {e}") from e

    def stage_local_model(
        self,
        local_path: Path,
        *,
        gpu: Gpu,
        log: Callable[[str], None] = lambda _msg: None,
    ) -> StagedModel:
        """Create a Modal volume, upload `local_path` onto it, and return the
        on-volume path the sandbox loads from."""
        sdk = self._sdk()
        volume_name = f"unsloth-model-{int(time.time())}"
        model_name = local_path.name or local_path.resolve().name or "model"

        log(f"Creating Modal volume {volume_name}...")
        vol = sdk.Volume.from_name(volume_name, create_if_missing = True)

        log(f"Uploading {local_path} to the volume (stays in Modal)...")
        try:
            with vol.batch_upload() as batch:
                if local_path.is_dir():
                    batch.put_directory(str(local_path), f"/{model_name}")
                else:
                    batch.put_file(str(local_path), f"/{model_name}")
        except BaseException as e:
            # Any failure -- including a Ctrl-C mid-upload -- must not leave the
            # billing volume behind. Delete it, then propagate.
            try:
                sdk.Volume.delete(volume_name)
            except Exception:
                log(
                    f"  warning: couldn't delete volume {volume_name} after a failed "
                    f"upload; it may keep billing. Remove it with:\n"
                    f"      unsloth deploy delete-storage {volume_name} --provider modal"
                )
            # Wrap a plain SDK error so the command layer catches it; let interrupts pass.
            if isinstance(e, (KeyboardInterrupt, SystemExit, DeployError)):
                raise
            raise DeployError(f"Modal volume upload failed: {e}") from e

        return StagedModel(
            model_path = f"{MODEL_MOUNT_DIR}/{model_name}",
            storage_id = volume_name,
            summary = f"Modal volume {volume_name}",
            placement = None,  # Modal volumes are region-agnostic; no pinning
        )

    def delete_storage(self, storage_id: str) -> None:
        try:
            self._sdk().Volume.delete(storage_id)
        except Exception as e:
            raise DeployError(f"Modal volume delete failed: {e}") from e

    def _sdk(self):
        if self._modal is None:
            raise DeployError("Modal is not authenticated. Call auth() first.")
        return self._modal
