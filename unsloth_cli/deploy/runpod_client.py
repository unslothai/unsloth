# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from unsloth_cli.deploy import DeployError, Gpu, SshTarget, StagedModel
from unsloth_cli.deploy import runpod_storage
from unsloth_cli.deploy.base import (
    NEEDED_FOR_AUTH,
    NEEDED_FOR_LOCAL_MODEL,
    Option,
    Provider,
)

VOLUME_MOUNT_PATH = "/workspace"
POD_UPLOADS_DIR = "/workspace/uploads"
TERMINAL_STATUSES = ("EXITED", "FAILED", "TERMINATED", "DEAD")
POLL_INTERVAL_S = 3

STOCK_RANK = {"High": 3, "Medium": 2, "Low": 1}
AVAILABILITY_WORKERS = 16

VOLUME_HEADROOM_FACTOR = 1.3
VOLUME_MIN_GB = 20


class RunPod(Provider):
    name = "runpod"

    supports_ssh = True
    supports_pause = True
    supports_local_model = True

    def __init__(self):
        self._sdk_mod = None
        self._api_key: Optional[str] = None
        self._s3_access_key: Optional[str] = None
        self._s3_secret_key: Optional[str] = None
        self._datacenter: Optional[str] = None

    @classmethod
    def option_schema(cls) -> list[Option]:
        return [
            Option(
                key = "api_key", env = "RUNPOD_API_KEY", secret = True,
                needed_for = NEEDED_FOR_AUTH,
                help = "RunPod API key (Settings > API Keys, rpa_...)",
            ),
            Option(
                key = "s3_access_key", env = "RUNPOD_S3_ACCESS_KEY_ID", secret = True,
                needed_for = NEEDED_FOR_LOCAL_MODEL,
                help = "RunPod S3 access key (Settings > S3 API Keys, user_...) "
                       "-- used to upload a local model",
            ),
            Option(
                key = "s3_secret_key", env = "RUNPOD_S3_SECRET_ACCESS_KEY", secret = True,
                needed_for = NEEDED_FOR_LOCAL_MODEL,
                help = "RunPod S3 secret access key (rps_...)",
            ),
            Option(
                key = "datacenter", env = "RUNPOD_DATACENTER", required = False,
                needed_for = NEEDED_FOR_LOCAL_MODEL,
                help = "Datacenter for the network volume "
                       "(blank = auto-pick one with capacity for the chosen GPU)",
            ),
        ]

    @property
    def api_key(self) -> str:
        if not self._api_key:
            raise DeployError("RunPod is not authenticated. Call auth() first.")
        return self._api_key

    def auth(self, options: dict[str, str]) -> None:
        key = options.get("api_key")
        if not key:
            raise DeployError(
                "Missing RUNPOD_API_KEY.\n"
                "Get one at https://www.runpod.io/console/user/settings, then:\n"
                "    export RUNPOD_API_KEY=rpa_..."
            )
        try:
            import runpod
        except ImportError as e:
            raise DeployError(
                "The 'runpod' package is required for `unsloth deploy`.\n"
                "Install it with:\n"
                "    pip install unsloth[deploy]"
            ) from e
        runpod.api_key = key
        self._sdk_mod = runpod
        self._api_key = key
        self._s3_access_key = options.get("s3_access_key")
        self._s3_secret_key = options.get("s3_secret_key")
        self._datacenter = options.get("datacenter") or None

    def list_gpus(self, min_vram_gb: int = 0) -> list[Gpu]:
        sdk = self._sdk()
        try:
            catalog = sdk.get_gpus() or []
        except Exception as e:
            raise DeployError(f"RunPod GPU listing failed: {e}") from e

        stock = self._global_stock()
        out: list[Gpu] = []
        for g in catalog:
            vram = int(g.get("memoryInGb") or 0)
            if vram < min_vram_gb:
                continue
            try:
                detail = sdk.get_gpu(g["id"]) or {}
            except Exception:
                continue
            price = (detail.get("lowestPrice") or {}).get("uninterruptablePrice")
            if not isinstance(price, (int, float)) or price <= 0:
                continue
            out.append(Gpu(
                id = g["id"],
                name = g.get("displayName") or g["id"],
                vram_gb = vram,
                cost_per_hour_usd = float(price),
                stock = stock.get(g["id"]),
            ))
        out.sort(key = lambda o: (o.cost_per_hour_usd, o.vram_gb))
        return out

    def _global_stock(self) -> dict[str, str]:
        try:
            from runpod.api.graphql import run_graphql_query

            rows = run_graphql_query(
                "{ gpuTypes { id lowestPrice(input: {gpuCount: 1, secureCloud: true})"
                " { stockStatus } } }"
            )["data"]["gpuTypes"]
        except Exception:
            return {}
        out: dict[str, str] = {}
        for r in rows:
            band = (r.get("lowestPrice") or {}).get("stockStatus")
            if r.get("id") and band in STOCK_RANK:
                out[r["id"]] = band
        return out

    def datacenters_for_gpu(self, gpu_id: str) -> list[tuple[str, str]]:
        self._sdk()
        try:
            from runpod.api.graphql import run_graphql_query
        except Exception as e:
            raise DeployError(f"RunPod availability lookup is unavailable: {e}") from e

        try:
            data = run_graphql_query(
                "{ dataCenters { id storageSupport } }"
            )["data"]["dataCenters"]
        except Exception as e:
            raise DeployError(f"RunPod datacenter listing failed: {e}") from e
        dc_ids = [d["id"] for d in data if d.get("id") and d.get("storageSupport")]

        def stock(dc: str) -> Optional[str]:
            query = (
                '{ gpuTypes(input: {id: "%s"}) { lowestPrice('
                'input: {gpuCount: 1, secureCloud: true, dataCenterId: "%s"}'
                ') { stockStatus } } }' % (gpu_id, dc)
            )
            try:
                rows = run_graphql_query(query)["data"]["gpuTypes"]
            except Exception:
                return None
            price = (rows[0].get("lowestPrice") or {}) if rows else {}
            return price.get("stockStatus")

        with ThreadPoolExecutor(max_workers = AVAILABILITY_WORKERS) as pool:
            ranked = [
                (dc, status)
                for dc, status in zip(dc_ids, pool.map(stock, dc_ids))
                if status in STOCK_RANK
            ]
        ranked.sort(key = lambda pair: (-STOCK_RANK[pair[1]], pair[0]))
        return ranked

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
        ports = [f"{http_port}/http"]
        if ssh_port:
            ports.append(f"{ssh_port}/tcp")
        kwargs = dict(
            name = name,
            image_name = image,
            gpu_type_id = gpu.id,
            ports = ",".join(ports),
            container_disk_in_gb = disk_gb,
            volume_mount_path = VOLUME_MOUNT_PATH,
            support_public_ip = True,
            env = env,
        )
        if staged and staged.storage_id:
            kwargs["network_volume_id"] = staged.storage_id
            kwargs["data_center_id"] = staged.placement
            kwargs["volume_in_gb"] = 0
        else:
            kwargs["volume_in_gb"] = disk_gb
        try:
            pod = self._sdk().create_pod(**kwargs)
        except Exception as e:
            raise DeployError(f"RunPod create_pod failed: {e}") from e
        if not pod or "id" not in pod:
            raise DeployError(f"RunPod create_pod returned unexpected payload: {pod!r}")
        return pod["id"]

    def wait_ready(self, instance_id: str, timeout_s: int) -> None:
        sdk = self._sdk()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                pod = sdk.get_pod(instance_id) or {}
            except Exception:
                time.sleep(POLL_INTERVAL_S)
                continue
            status = pod.get("desiredStatus")
            if status in TERMINAL_STATUSES:
                raise DeployError(f"Pod {instance_id} reached terminal status: {status}")
            if status == "RUNNING" and pod.get("runtime"):
                return
            time.sleep(POLL_INTERVAL_S)
        raise DeployError(f"Pod {instance_id} did not start within {timeout_s}s")

    def get_ssh(self, instance_id: str) -> SshTarget:
        pod = self._sdk().get_pod(instance_id) or {}
        for p in (pod.get("runtime") or {}).get("ports") or []:
            if (
                p.get("privatePort") == 22
                and p.get("isIpPublic")
                and p.get("ip")
                and p.get("publicPort")
            ):
                return SshTarget(
                    user = "root",
                    host = p["ip"],
                    port = int(p["publicPort"]),
                )
        return SshTarget(user = instance_id, host = "ssh.runpod.io", port = 22)

    def endpoint_url(self, instance_id: str, http_port: int) -> str:
        return f"https://{instance_id}-{http_port}.proxy.runpod.net"

    def pause(self, instance_id: str) -> None:
        try:
            self._sdk().stop_pod(instance_id)
        except Exception as e:
            raise DeployError(f"RunPod stop_pod failed: {e}") from e

    def terminate(self, instance_id: str) -> None:
        try:
            self._sdk().terminate_pod(instance_id)
        except Exception as e:
            raise DeployError(f"RunPod terminate_pod failed: {e}") from e

    def stage_local_model(
        self,
        local_path: Path,
        *,
        gpu: Gpu,
        log: Callable[[str], None] = lambda _msg: None,
    ) -> StagedModel:
        if not (self._s3_access_key and self._s3_secret_key):
            raise DeployError(
                "Uploading a local model needs RunPod S3 credentials.\n"
                "Create them at https://www.runpod.io/console/user/settings "
                "(S3 API Keys), then set RUNPOD_S3_ACCESS_KEY_ID / "
                "RUNPOD_S3_SECRET_ACCESS_KEY."
            )

        datacenter = self._datacenter or self._auto_datacenter(gpu)
        size_gb = _volume_size_gb(local_path)
        model_name = local_path.name or local_path.resolve().name or "model"

        volume_id = runpod_storage.create_network_volume(
            self,
            name = f"unsloth-{int(time.time())}",
            size_gb = size_gb,
            datacenter_id = datacenter,
        )

        prefix = f"uploads/{model_name}" if local_path.is_dir() else "uploads"
        log(f"Uploading {local_path} to RunPod storage (this can take a while)...")
        try:
            runpod_storage.upload_path(
                local_path,
                volume_id = volume_id,
                datacenter = datacenter,
                access_key = self._s3_access_key,
                secret_key = self._s3_secret_key,
                prefix = prefix,
            )
        except BaseException:
            try:
                runpod_storage.delete_network_volume(self, volume_id)
            except Exception:
                log(
                    f"  warning: couldn't delete network volume {volume_id} after a "
                    f"failed upload; it may keep billing. Remove it with:\n"
                    f"      unsloth deploy delete-storage {volume_id}"
                )
            raise

        return StagedModel(
            model_path = f"{POD_UPLOADS_DIR}/{model_name}",
            storage_id = volume_id,
            summary = f"network volume {volume_id} in {datacenter} (S3)",
            placement = datacenter,
        )

    def delete_storage(self, storage_id: str) -> None:
        runpod_storage.delete_network_volume(self, storage_id)

    def _auto_datacenter(self, gpu: Gpu) -> str:
        ranked = self.datacenters_for_gpu(gpu.id)
        if not ranked:
            raise DeployError(
                f"No datacenter currently has secure-cloud capacity for {gpu.name}.\n"
                "Pick a different GPU (the picker shows stock) or retry shortly."
            )
        return ranked[0][0]

    def _sdk(self):
        if self._sdk_mod is None:
            raise DeployError("RunPod is not authenticated. Call auth() first.")
        self._sdk_mod.api_key = self._api_key
        return self._sdk_mod


def _volume_size_gb(local: Path) -> int:
    total = 0
    if local.is_dir():
        for p in local.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    else:
        total = local.stat().st_size
    return max(VOLUME_MIN_GB, math.ceil(total / 1e9 * VOLUME_HEADROOM_FACTOR) + 5)
