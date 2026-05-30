# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Launch Unsloth Studio on RunPod."""

from __future__ import annotations

import json
import math
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from unsloth_cli.deploy import DeployError, Gpu, SshTarget
from unsloth_cli.deploy import runpod_storage
from unsloth_cli.deploy.provider import get_provider
from unsloth_cli.deploy.studio_client import StudioClient


IMAGE_TAG = "ghcr.io/nilayyadav/unsloth-base:cu128"

STUDIO_PORT = 8000
SSH_PORT = 22
DEFAULT_MIN_VRAM_GB = 24
DEFAULT_DISK_GB = 100
POD_RUNNING_TIMEOUT_S = 900
STUDIO_HEALTH_TIMEOUT_S = 300

MAX_GPU_CHOICES = 10

MIN_ADMIN_PASSWORD_LENGTH = 8
DEFAULT_ADMIN_USERNAME = "unsloth"

MODEL_SEARCH_DIRS = ("outputs", "lora_model", "merged_model", "model")

DEFAULT_PROVIDER = "runpod"

POD_UPLOADS_DIR = "/workspace/uploads"
# Network volumes are datacenter-pinned and the pod can only schedule in the
# volume's datacenter, so we don't hardcode one -- a fixed datacenter's stock
# comes and goes. We place the volume where the chosen GPU has live capacity.
VOLUME_HEADROOM_FACTOR = 1.3
VOLUME_MIN_GB = 20


deploy_app = typer.Typer(
    help = "Launch Unsloth Studio on RunPod.",
    no_args_is_help = True,
)


@dataclass(frozen = True)
class Pod:
    """A running pod and the endpoints we reach it on."""
    id: str
    studio_url: str
    ssh: SshTarget


@deploy_app.command("run")
def run(
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER, "--provider",
        help = "Cloud provider to deploy to.",
    ),
    gpu: Optional[str] = typer.Option(
        None, "--gpu", help = "Skip the picker; use this provider's GPU id.",
    ),
    min_vram: int = typer.Option(
        DEFAULT_MIN_VRAM_GB, "--min-vram",
        help = "Drop GPUs with less than this many GB of VRAM.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help = "Auto-pick the cheapest fitting GPU; skip confirmation.",
    ),
    admin_password: Optional[str] = typer.Option(
        None, "--admin-password", envvar = "UNSLOTH_ADMIN_PASSWORD",
        help = "Studio admin password. Prefer the env var; CLI args show in `ps`.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help = "Hugging Face id or local path. If set, auto-loads after boot.",
    ),
    gguf_variant: Optional[str] = typer.Option(
        None, "--gguf-variant",
        help = "GGUF quantization (e.g. Q4_K_M). Only used with --model.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN",
        help = "Hugging Face token for gated repos. Only used with --model.",
    ),
    datacenter: Optional[str] = typer.Option(
        None, "--datacenter", envvar = "RUNPOD_DATACENTER",
        help = "Datacenter for the network volume a local --model is uploaded to. "
               "Default: auto-pick one with live capacity for the chosen GPU.",
    ),
    s3_access_key: Optional[str] = typer.Option(
        None, "--s3-access-key", envvar = "RUNPOD_S3_ACCESS_KEY_ID",
        help = "RunPod S3 access key (Settings > S3 API Keys). For local --model uploads.",
    ),
    s3_secret_key: Optional[str] = typer.Option(
        None, "--s3-secret-key", envvar = "RUNPOD_S3_SECRET_ACCESS_KEY",
        help = "RunPod S3 secret. Prefer the env var; CLI args show in `ps`.",
    ),
):
    """Launch Unsloth Studio on RunPod."""
    if not admin_password:
        _fail("Set UNSLOTH_ADMIN_PASSWORD or pass --admin-password.", code = 2)
    if len(admin_password) < MIN_ADMIN_PASSWORD_LENGTH:
        _fail(
            f"Admin password must be at least {MIN_ADMIN_PASSWORD_LENGTH} characters.",
            code = 2,
        )

    provider = _provider(provider_name)

    if model is None and not yes:
        model = _pick_model()

    # A local model rides a RunPod network volume (uploaded over the S3 API),
    # which needs S3 credentials. Check before we create a billing pod.
    if _is_local_model(model) and not (s3_access_key and s3_secret_key):
        _fail(
            "Uploading a local model needs RunPod S3 credentials.\n"
            "Create them at https://www.runpod.io/console/user/settings (S3 API Keys), then:\n"
            "    export RUNPOD_S3_ACCESS_KEY_ID=user_...\n"
            "    export RUNPOD_S3_SECRET_ACCESS_KEY=rps_...",
            code = 2,
        )

    chosen = _pick_gpu(provider, override = gpu, min_vram_gb = min_vram, yes = yes)

    # A local model rides a datacenter-pinned network volume, and the pod can
    # only start in that volume's datacenter -- so place it where the GPU has
    # live capacity rather than a fixed default that may be sold out.
    if _is_local_model(model):
        datacenter = _resolve_datacenter(provider, chosen, datacenter)

    typer.echo("")
    typer.echo(f"  Provider: {provider.name}")
    typer.echo(f"  Image:   {IMAGE_TAG}")
    typer.echo(f"  Studio:  port {STUDIO_PORT}")
    typer.echo(f"  GPU:     {chosen.name} ({chosen.vram_gb} GB) - ${chosen.cost_per_hour_usd:.3f}/hr")
    if model is not None:
        typer.echo(f"  Model:   {model}")
    if _is_local_model(model):
        typer.echo(f"  Upload:  network volume in {datacenter} (S3)")
    typer.echo("")

    if not yes and not typer.confirm("Continue?", default = False):
        typer.echo("Aborted.")
        raise typer.Exit(0)

    _deploy(
        provider,
        chosen,
        admin_password = admin_password,
        model = model,
        gguf_variant = gguf_variant,
        hf_token = hf_token,
        datacenter = datacenter,
        s3_access_key = s3_access_key,
        s3_secret_key = s3_secret_key,
    )


@deploy_app.command("stop")
def stop(
    pod_id: str = typer.Argument(..., help = "Pod id (printed when deploy succeeds)."),
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER, "--provider", help = "Provider the pod runs on.",
    ),
    keep_volume: bool = typer.Option(
        False, "--keep-volume",
        help = "Pause instead of terminating. Disk storage is still billed.",
    ),
):
    """Terminate the pod so billing stops. Pass --keep-volume to pause it instead."""
    provider = _provider(provider_name)
    try:
        if keep_volume:
            provider.stop_pod(pod_id)
            typer.echo(f"Pod {pod_id} paused (volume preserved, still incurs disk billing).")
        else:
            provider.terminate_pod(pod_id)
            typer.echo(f"Pod {pod_id} terminated.")
    except DeployError as e:
        _fail(str(e))


@deploy_app.command("delete-volume")
def delete_volume(
    volume_id: str = typer.Argument(..., help = "Network volume id (printed when deploy uploads a local model)."),
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER, "--provider", help = "Provider the volume lives on.",
    ),
):
    """Delete a network volume so its storage stops billing. Terminate the pod first."""
    provider = _provider(provider_name)
    try:
        runpod_storage.delete_network_volume(provider, volume_id)
        typer.echo(f"Network volume {volume_id} deleted.")
    except DeployError as e:
        _fail(str(e))


def _model_kind(p: Path) -> Optional[str]:
    if (p / "adapter_config.json").is_file():
        return "LoRA adapter"
    if (p / "config.json").is_file():
        return "HF model"
    try:
        if any(p.glob("*.gguf")):
            return "GGUF"
    except OSError:
        pass
    return None


def _discover_local_models(cwd: Optional[Path] = None) -> list[tuple[Path, str]]:
    """Find model directories under `cwd`, the common output dirs, and the
    Studio output/export roots. Looks one level into each so HF Trainer
    checkpoints and Studio runs show up individually.
    """
    cwd = (cwd or Path.cwd()).resolve()
    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(path: Path) -> bool:
        try:
            path = path.resolve()
        except OSError:
            return False
        if path in seen or not path.is_dir():
            return False
        seen.add(path)
        kind = _model_kind(path)
        if kind is None:
            return False
        found.append((path, kind))
        return True

    add(cwd)
    # Studio's roots are added too, so models trained in the UI also show up.
    roots = [cwd / name for name in MODEL_SEARCH_DIRS] + _studio_output_roots()
    for root in roots:
        if add(root):
            continue
        try:
            for child in sorted(root.iterdir()):
                add(child)
        except OSError:
            continue
    return found


def _studio_output_roots() -> list[Path]:
    """Studio's run and export dirs, which live outside the cwd. We ask the
    backend where they are rather than hardcode a path; [] if it isn't installed."""
    try:
        from unsloth_cli.commands.studio import _find_run_py

        run_py = _find_run_py()
        if run_py is None:
            return []
        # utils.paths uses root-relative imports, so backend must be on the path.
        backend = str(run_py.parent)
        if backend not in sys.path:
            sys.path.insert(0, backend)
        from utils.paths import outputs_root, exports_root

        return [outputs_root(), exports_root()]
    except Exception:
        return []


def _pick_model() -> Optional[str]:
    local = _discover_local_models()
    cwd = Path.cwd()

    typer.echo("")
    typer.echo("What do you want to deploy?")

    for i, (path, kind) in enumerate(local, start = 1):
        try:
            label = f"./{path.relative_to(cwd)}/"
        except ValueError:
            label = str(path)
        typer.echo(f"  {i:2d}. {label:<46} {kind}")

    hf_idx = len(local) + 1
    skip_idx = len(local) + 2
    typer.echo(f"  {hf_idx:2d}. <Hugging Face id>")
    typer.echo(f"  {skip_idx:2d}. <skip>  launch Studio empty; load from the UI")
    typer.echo("")

    raw = typer.prompt("Pick a model (number)", default = str(skip_idx))
    try:
        idx = int(raw)
    except ValueError:
        _fail(f"Invalid selection: {raw!r}", code = 2)

    if 1 <= idx <= len(local):
        return str(local[idx - 1][0])
    if idx == hf_idx:
        hf_id = typer.prompt("Hugging Face model id").strip()
        return hf_id or None
    if idx == skip_idx:
        return None
    _fail(f"Invalid selection: {raw!r}", code = 2)


def _pick_gpu(
    provider, *, override: Optional[str], min_vram_gb: int, yes: bool,
) -> Gpu:
    try:
        options = provider.list_gpus(min_vram_gb = min_vram_gb)
    except DeployError as e:
        _fail(str(e))

    if not options:
        _fail(f"No on-demand GPUs with >= {min_vram_gb} GB VRAM found.")

    if override is not None:
        for gpu in options:
            if gpu.id == override:
                return gpu
        listing = "\n".join(
            f"  {gpu.id} ({gpu.vram_gb} GB, ${gpu.cost_per_hour_usd:.3f}/hr)"
            for gpu in options
        )
        _fail(f"--gpu '{override}' not found. Available:\n{listing}", code = 2)

    if yes:
        return options[0]

    shown = options[:MAX_GPU_CHOICES]
    typer.echo("")
    typer.echo("Available GPUs (cheapest first):")
    for i, gpu in enumerate(shown, start = 1):
        typer.echo(
            f"  {i:2d}. {gpu.name:<28} {gpu.vram_gb:>3} GB   "
            f"${gpu.cost_per_hour_usd:.3f}/hr   ({gpu.id})"
        )
    if len(options) > len(shown):
        typer.echo(
            f"  ... and {len(options) - len(shown)} more "
            "(narrow with --min-vram, or pick one directly with --gpu <id>)."
        )
    typer.echo("")
    raw = typer.prompt("Pick a GPU (number)", default = "1")
    try:
        idx = int(raw) - 1
        if not 0 <= idx < len(shown):
            raise ValueError
    except ValueError:
        _fail(f"Invalid selection: {raw!r}", code = 2)
    return shown[idx]


def _resolve_datacenter(provider, gpu: Gpu, requested: Optional[str]) -> str:
    """Pick the datacenter the network volume (and therefore the pod) lands in.
    Honor an explicit --datacenter; otherwise choose one that currently has
    capacity for `gpu`, so the pod can actually schedule."""
    if requested:
        return requested
    if not hasattr(provider, "datacenters_for_gpu"):
        _fail(
            f"{provider.name} can't auto-pick a datacenter; pass --datacenter.",
            code = 2,
        )

    typer.echo(f"Finding a datacenter with capacity for {gpu.name}...")
    try:
        ranked = provider.datacenters_for_gpu(gpu.id)
    except DeployError as e:
        _fail(str(e))

    if not ranked:
        _fail(
            f"No datacenter currently has secure-cloud capacity for {gpu.name}.\n"
            "Pick a different GPU, or retry shortly -- RunPod stock changes minute "
            "to minute. Live availability: https://www.runpod.io/console/deploy"
        )

    datacenter, stock = ranked[0]
    alts = ", ".join(f"{dc} ({s})" for dc, s in ranked[1:4])
    typer.echo(
        f"  -> {datacenter} ({stock} stock)"
        + (f"; also available: {alts}" if alts else "")
    )
    return datacenter


def _deploy(
    provider,
    gpu: Gpu,
    *,
    admin_password: str,
    model: Optional[str],
    gguf_variant: Optional[str],
    hf_token: Optional[str],
    datacenter: Optional[str],
    s3_access_key: Optional[str],
    s3_secret_key: Optional[str],
) -> None:
    # A local model is shipped ahead of the pod onto a network volume (over S3);
    # an HF id -- or no model -- needs no volume and Studio pulls it directly.
    load_model = model            # what Studio ends up loading: pod path or HF id
    load_kwargs: dict = {}
    network_volume_id: Optional[str] = None

    if _is_local_model(model):
        local = Path(model).expanduser()
        network_volume_id, load_model = _upload_to_volume(
            provider, local,
            datacenter = datacenter,
            access_key = s3_access_key,
            secret_key = s3_secret_key,
        )
        if local.is_dir() and (local / "adapter_config.json").is_file():
            load_kwargs["is_lora"] = True
            typer.echo("  detected adapter_config.json, loading as LoRA adapter")

    if gguf_variant:
        load_kwargs["gguf_variant"] = gguf_variant
    if hf_token:
        load_kwargs["hf_token"] = hf_token

    name = f"unsloth-studio-{int(time.time())}"
    typer.echo(f"Creating pod '{name}'...")
    try:
        pod_id = provider.create_pod(
            name = name,
            gpu_id = gpu.id,
            image = IMAGE_TAG,
            ports = [f"{STUDIO_PORT}/http"],
            ssh_port = SSH_PORT,
            disk_gb = DEFAULT_DISK_GB,
            env = {"UNSLOTH_ADMIN_PASSWORD": admin_password},
            network_volume_id = network_volume_id,
            data_center_id = datacenter if network_volume_id else None,
        )
    except DeployError as e:
        # Capacity in the chosen datacenter can vanish between the availability
        # check and this call; don't leave the volume behind billing for it.
        _fail(_cleanup_volume_after_failure(provider, network_volume_id, str(e)))
    typer.echo(f"  pod id: {pod_id}")

    try:
        typer.echo("Waiting for pod to start...")
        provider.wait_running(pod_id, timeout_s = POD_RUNNING_TIMEOUT_S)
    except DeployError as e:
        _fail(
            f"Deploy failed: {e}\n"
            f"Pod {pod_id} may still be running and billing.\n"
            f"Stop it with: unsloth deploy stop {pod_id}"
            + _volume_billing_note(network_volume_id)
        )

    pod = Pod(
        id = pod_id,
        studio_url = provider.endpoint_url(pod_id, http_port = STUDIO_PORT),
        ssh = provider.get_ssh(pod_id),
    )

    if load_model is None:
        _print_studio_ready(pod, gpu, network_volume_id)
    else:
        _serve_model(
            pod, gpu,
            bootstrap_password = admin_password,
            model = load_model,
            load_kwargs = load_kwargs,
            network_volume_id = network_volume_id,
        )


def _serve_model(
    pod: Pod,
    gpu: Gpu,
    *,
    bootstrap_password: str,
    model: str,
    load_kwargs: dict,
    network_volume_id: Optional[str],
) -> None:
    client = StudioClient(pod.studio_url)
    rotated_password: Optional[str] = None

    try:
        typer.echo("Waiting for Studio API to come online...")
        client.wait_healthy(timeout_s = STUDIO_HEALTH_TIMEOUT_S)

        new_password = secrets.token_urlsafe(18)
        typer.echo("Rotating bootstrap password and minting credential...")
        client.login(username = DEFAULT_ADMIN_USERNAME, password = bootstrap_password)
        client.change_password(current = bootstrap_password, new = new_password)
        rotated_password = new_password
        api_key, credential_note = _mint_credential(client)

        typer.echo(f"Loading '{model}' on the pod (can take several minutes)...")
        client.load_model(model_path = model, **load_kwargs)

        _print_inference_ready(
            pod, gpu,
            api_key = api_key,
            credential_note = credential_note,
            model = model,
            admin_password = new_password,
            network_volume_id = network_volume_id,
        )
    except DeployError as e:
        _fail(_stop_hint(
            e, pod.id,
            admin_password = rotated_password,
            network_volume_id = network_volume_id,
        ))


def _is_local_model(model: Optional[str]) -> bool:
    return model is not None and Path(model).expanduser().exists()


def _volume_size_gb(local: Path) -> int:
    """Volume size for a model: its on-disk size + headroom for what Studio
    writes at runtime on the same /workspace volume."""
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


def _upload_to_volume(
    provider,
    local: Path,
    *,
    datacenter: str,
    access_key: Optional[str],
    secret_key: Optional[str],
) -> tuple[str, str]:
    """Create a network volume, upload `local` onto it over S3, and return
    (volume_id, pod-side model path). No SSH, and nothing leaves RunPod."""
    name = local.resolve().name or "model"
    size_gb = _volume_size_gb(local)

    typer.echo(f"Creating {size_gb} GB network volume in {datacenter}...")
    try:
        volume_id = runpod_storage.create_network_volume(
            provider,
            name = f"unsloth-{int(time.time())}",
            size_gb = size_gb,
            datacenter_id = datacenter,
        )
    except DeployError as e:
        _fail(str(e))
    typer.echo(f"  volume id: {volume_id}")

    prefix = f"uploads/{name}"
    typer.echo(f"Uploading {local} to the volume over S3 (stays in RunPod)...")
    try:
        runpod_storage.upload_path(
            local,
            volume_id = volume_id,
            datacenter = datacenter,
            access_key = access_key,
            secret_key = secret_key,
            prefix = prefix,
            on_file = lambda key, size: typer.echo(f"  {key}  ({size / 1e6:.0f} MB)"),
        )
    except DeployError as e:
        _fail(_with_volume_note(str(e), volume_id))
    return volume_id, f"{POD_UPLOADS_DIR}/{name}"


def _volume_billing_note(volume_id: Optional[str]) -> str:
    if not volume_id:
        return ""
    return (
        f"\nNetwork volume {volume_id} also persists (storage billed). Delete it with:"
        f"\n    unsloth deploy delete-volume {volume_id}"
    )


def _cleanup_volume_after_failure(
    provider, volume_id: Optional[str], msg: str,
) -> str:
    """Delete the just-created volume after a failed pod create so it stops
    billing. If the delete itself fails, fall back to telling the user how."""
    if not volume_id:
        return msg
    try:
        runpod_storage.delete_network_volume(provider, volume_id)
    except DeployError as e:
        return _with_volume_note(
            f"{msg}\n(Could not auto-delete the network volume: {e})", volume_id,
        )
    return f"{msg}\nThe network volume {volume_id} was deleted, so it is not billing."


def _with_volume_note(msg: str, volume_id: Optional[str]) -> str:
    if not volume_id:
        return msg
    return (
        f"{msg}\n"
        f"A network volume ({volume_id}) was created and is billing. Delete it with:\n"
        f"    unsloth deploy delete-volume {volume_id}"
    )


def _mint_credential(client: StudioClient) -> tuple[str, str]:
    """Mint a permanent API key, falling back to the login JWT on older images
    that predate POST /api/auth/api-keys (they 404/405). Any other failure
    propagates to the caller's stop-the-pod handler."""
    try:
        return client.create_api_key(name = f"deploy-{int(time.time())}"), ""
    except DeployError as e:
        if any(code in str(e) for code in (" 404", " 405")):
            note = (
                "  (JWT credential, expires. Rebake the image with a newer\n"
                "   UNSLOTH_COMMIT to get permanent sk-unsloth-... keys.)"
            )
            return client.token, note
        raise


def _print_studio_ready(
    pod: Pod, gpu: Gpu, network_volume_id: Optional[str] = None,
) -> None:
    typer.echo("")
    typer.echo("Unsloth Studio pod is starting (Studio may take ~1 minute).")
    typer.echo(f"  Studio:  {pod.studio_url}")
    typer.echo(f"  SSH:     ssh {pod.ssh.user}@{pod.ssh.host} -p {pod.ssh.port}")
    typer.echo("")
    _print_stop_hint(pod, gpu, network_volume_id)


def _print_inference_ready(
    pod: Pod,
    gpu: Gpu,
    *,
    api_key: str,
    credential_note: str,
    model: str,
    admin_password: str,
    network_volume_id: Optional[str] = None,
) -> None:
    base_url = f"{pod.studio_url}/v1"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]})

    typer.echo("")
    typer.echo("Inference endpoint ready.")
    typer.echo(f"    base_url:  {base_url}")
    typer.echo(f"    api_key:   {api_key}")
    if credential_note:
        typer.echo(credential_note)
    typer.echo(f"    model:     {model}")
    typer.echo(f"    admin_pw:  {admin_password}   (Studio UI at {pod.studio_url})")
    typer.echo("")
    typer.echo("  Test it:")
    typer.echo(f"    curl {base_url}/chat/completions \\")
    typer.echo(f"      -H 'Authorization: Bearer {api_key}' \\")
    typer.echo("      -H 'Content-Type: application/json' \\")
    typer.echo(f"      -d '{body}'")
    typer.echo("")
    _print_stop_hint(pod, gpu, network_volume_id)


def _print_stop_hint(
    pod: Pod, gpu: Gpu, network_volume_id: Optional[str] = None,
) -> None:
    typer.echo(f"  STOP THIS POD WHEN DONE - billed at ${gpu.cost_per_hour_usd:.3f}/hr.")
    typer.echo(f"      unsloth deploy stop {pod.id}")
    if network_volume_id:
        typer.echo(f"  Network volume {network_volume_id} persists (storage billed). Delete when done:")
        typer.echo(f"      unsloth deploy delete-volume {network_volume_id}")
    typer.echo("")


def _stop_hint(
    err: DeployError,
    pod_id: str,
    *,
    admin_password: Optional[str] = None,
    network_volume_id: Optional[str] = None,
) -> str:
    lines = [f"Deploy bootstrap failed: {err}"]
    if admin_password is not None:
        lines.append(
            f"The Studio admin password was already rotated to: {admin_password}\n"
            "  (save it -- the original password no longer works for the Studio UI)."
        )
    lines += [
        f"Pod {pod_id} is still running and billing.",
        f"Stop it with: unsloth deploy stop {pod_id}",
    ]
    note = _volume_billing_note(network_volume_id)
    return "\n".join(lines) + note


def _provider(name: str):
    try:
        provider = get_provider(name)
        provider.auth()
    except DeployError as e:
        _fail(str(e))
    return provider


def _fail(msg: str, code: int = 1):
    typer.echo(msg, err = True)
    raise typer.Exit(code)
