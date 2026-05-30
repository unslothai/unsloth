# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RunPod compute client: GPU catalog and pod lifecycle.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from unsloth_cli.deploy import DeployError, Gpu, SshTarget

VOLUME_MOUNT_PATH = "/workspace"
TERMINAL_STATUSES = ("EXITED", "FAILED", "TERMINATED", "DEAD")
POLL_INTERVAL_S = 3

# RunPod reports availability as a coarse band; higher is more likely to schedule.
STOCK_RANK = {"High": 3, "Medium": 2, "Low": 1}
AVAILABILITY_WORKERS = 16


class RunPod:
    name = "runpod"

    def __init__(self):
        self._sdk_mod = None
        self._api_key: Optional[str] = None

    @property
    def api_key(self) -> str:
        if not self._api_key:
            raise DeployError("RunPod is not authenticated. Call auth() first.")
        return self._api_key

    def auth(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("RUNPOD_API_KEY")
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

    def list_gpus(self, min_vram_gb: int = 0) -> list[Gpu]:
        sdk = self._sdk()
        try:
            catalog = sdk.get_gpus() or []
        except Exception as e:
            raise DeployError(f"RunPod GPU listing failed: {e}") from e

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
            ))
        out.sort(key = lambda o: (o.cost_per_hour_usd, o.vram_gb))
        return out

    def datacenters_for_gpu(self, gpu_id: str) -> list[tuple[str, str]]:
        """`(datacenter_id, stock_status)` for datacenters that currently have
        secure-cloud capacity for `gpu_id`, best availability first. Empty if the
        GPU is unschedulable everywhere right now. Used to place a network-volume
        deploy where the pod can actually start, instead of a fixed datacenter
        whose stock comes and goes.

        Network volumes only exist in secure cloud, so we ask per datacenter with
        `secureCloud: true`. The catalog (`get_gpus`) is global and says nothing
        about per-datacenter stock, hence the sweep."""
        self._sdk()  # ensure authenticated
        try:
            from runpod.api.graphql import run_graphql_query
        except Exception as e:
            raise DeployError(f"RunPod availability lookup is unavailable: {e}") from e

        try:
            data = run_graphql_query("{ dataCenters { id } }")["data"]["dataCenters"]
        except Exception as e:
            raise DeployError(f"RunPod datacenter listing failed: {e}") from e
        dc_ids = [d["id"] for d in data if d.get("id")]

        def stock(dc: str) -> Optional[str]:
            query = (
                '{ gpuTypes(input: {id: "%s"}) { lowestPrice('
                'input: {gpuCount: 1, secureCloud: true, dataCenterId: "%s"}'
                ') { stockStatus } } }' % (gpu_id, dc)
            )
            try:
                rows = run_graphql_query(query)["data"]["gpuTypes"]
            except Exception:
                return None  # a single datacenter erroring shouldn't sink the sweep
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

    def create_pod(
        self,
        *,
        name: str,
        gpu_id: str,
        image: str,
        ports: list[str],
        ssh_port: int,
        disk_gb: int,
        env: dict[str, str],
        network_volume_id: Optional[str] = None,
        data_center_id: Optional[str] = None,
    ) -> str:
        # A network volume *is* the /workspace volume and is pinned to a
        # datacenter, so when one is attached we don't also request a per-pod
        # volume, and we force the pod into that volume's datacenter.
        kwargs = dict(
            name = name,
            image_name = image,
            gpu_type_id = gpu_id,
            ports = ",".join([*ports, f"{ssh_port}/tcp"]),
            container_disk_in_gb = disk_gb,
            volume_mount_path = VOLUME_MOUNT_PATH,
            support_public_ip = True,
            env = env,
        )
        if network_volume_id:
            kwargs["network_volume_id"] = network_volume_id
            kwargs["data_center_id"] = data_center_id
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

    def wait_running(self, pod_id: str, timeout_s: int) -> None:
        # `desiredStatus` flips to RUNNING right after create_pod returns; the
        # container is actually up only once `runtime` is populated.
        sdk = self._sdk()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            pod = sdk.get_pod(pod_id) or {}
            status = pod.get("desiredStatus")
            if status in TERMINAL_STATUSES:
                raise DeployError(f"Pod {pod_id} reached terminal status: {status}")
            if status == "RUNNING" and pod.get("runtime"):
                return
            time.sleep(POLL_INTERVAL_S)
        raise DeployError(f"Pod {pod_id} did not start within {timeout_s}s")

    def get_ssh(self, pod_id: str) -> SshTarget:
        pod = self._sdk().get_pod(pod_id) or {}
        for p in (pod.get("runtime") or {}).get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("ip"):
                return SshTarget(
                    user = "root",
                    host = p["ip"],
                    port = int(p["publicPort"]),
                )
        # Proxy fallback: `ssh <pod_id>@ssh.runpod.io`. Requires a pubkey on
        # the user's account.
        return SshTarget(user = pod_id, host = "ssh.runpod.io", port = 22)

    def endpoint_url(self, pod_id: str, http_port: int) -> str:
        if self._sdk_mod is not None:
            try:
                pod = self._sdk().get_pod(pod_id) or {}
                for p in (pod.get("runtime") or {}).get("ports") or []:
                    if (
                        p.get("privatePort") == http_port
                        and p.get("isIpPublic")
                        and p.get("ip")
                        and p.get("publicPort")
                    ):
                        return f"http://{p['ip']}:{int(p['publicPort'])}"
            except Exception:
                pass
        return f"https://{pod_id}-{http_port}.proxy.runpod.net"

    def stop_pod(self, pod_id: str) -> None:
        try:
            self._sdk().stop_pod(pod_id)
        except Exception as e:
            raise DeployError(f"RunPod stop_pod failed: {e}") from e

    def terminate_pod(self, pod_id: str) -> None:
        try:
            self._sdk().terminate_pod(pod_id)
        except Exception as e:
            raise DeployError(f"RunPod terminate_pod failed: {e}") from e

    def _sdk(self):
        if self._sdk_mod is None:
            raise DeployError("RunPod is not authenticated. Call auth() first.")
        return self._sdk_mod
