# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Modal provider for `unsloth deploy`: GPU sandboxes running Unsloth Studio."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from unsloth_cli.deploy import DeployError, Gpu, StagedModel
from unsloth_cli.deploy.base import Option, Provider


APP_NAME = "unsloth-studio"
MODEL_MOUNT_DIR = "/model"
SANDBOX_TIMEOUT_S = 86400
POLL_INTERVAL_S = 3
STUDIO_IMAGE = "unsloth/unsloth"
STUDIO_HOME = "/home/unsloth/.unsloth/studio"

MODAL_GPUS = [  # (gpu id, display name, vram_gb, usd/hr)
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

    supports_ssh = False
    supports_pause = False
    supports_local_model = True
    reports_stock = False
    deploy_note = "Modal stops this instance automatically after 24h (max sandbox lifetime)."

    def __init__(self):
        self._modal = None
        self._app = None

    @classmethod
    def option_schema(cls) -> list[Option]:
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
        sdk = self._sdk()
        volumes = {}
        if staged and staged.storage_id:
            volumes[MODEL_MOUNT_DIR] = sdk.Volume.from_name(staged.storage_id)
        try:
            with sdk.enable_output():
                sb = sdk.Sandbox.create(
                    "/bin/sh", "-lc", self._start_command(http_port),
                    app = self._app,
                    name = name,
                    image = self._studio_image(),
                    gpu = gpu.id,
                    encrypted_ports = [http_port],
                    secrets = [sdk.Secret.from_dict(env)],
                    volumes = volumes,
                    timeout = SANDBOX_TIMEOUT_S,
                )
        except Exception as e:
            raise DeployError(f"Modal sandbox create failed: {e}") from e
        return sb.object_id

    def _start_command(self, http_port: int) -> str:
        # Studio seeds its admin user from a .bootstrap_password file, not from
        # UNSLOTH_ADMIN_PASSWORD, so write the password there before launch.
        # --api-only: the image's built web frontend is whited out.
        pw_file = f"{STUDIO_HOME}/auth/.bootstrap_password"
        return (
            f"export UNSLOTH_STUDIO_HOME={STUDIO_HOME} && "
            f"mkdir -p {STUDIO_HOME}/auth && "
            f'printf %s "$UNSLOTH_ADMIN_PASSWORD" > {pw_file} && '
            f"chmod 600 {pw_file} && "
            f"unsloth studio --api-only -p {http_port} -H 0.0.0.0"
        )

    def _studio_image(self):
        return self._sdk().Image.from_registry(STUDIO_IMAGE).entrypoint([])

    def wait_ready(self, instance_id: str, timeout_s: int) -> None:
        sdk = self._sdk()
        sb = sdk.Sandbox.from_id(instance_id)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            code = sb.poll()
            if code is not None:
                raise DeployError(
                    f"Modal sandbox {instance_id} exited before serving (code {code})."
                )
            try:
                if sb.tunnels(timeout = POLL_INTERVAL_S):
                    return
            except Exception:
                pass
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
            # Delete the volume on any failure (incl. Ctrl-C) so it stops billing.
            try:
                sdk.Volume.delete(volume_name)
            except Exception:
                log(
                    f"  warning: couldn't delete volume {volume_name} after a failed "
                    f"upload; remove it with:\n"
                    f"      unsloth deploy delete-storage {volume_name} --provider modal"
                )
            if isinstance(e, (KeyboardInterrupt, SystemExit, DeployError)):
                raise
            raise DeployError(f"Modal volume upload failed: {e}") from e

        return StagedModel(
            model_path = f"{MODEL_MOUNT_DIR}/{model_name}",
            storage_id = volume_name,
            summary = f"Modal volume {volume_name}",
            placement = None,
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
