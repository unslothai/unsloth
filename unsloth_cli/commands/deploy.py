# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from unsloth_cli.deploy import DeployError, Gpu, SshTarget, StagedModel
from unsloth_cli.deploy import store
from unsloth_cli.deploy.base import NEEDED_FOR_LOCAL_MODEL, Provider
from unsloth_cli.deploy.provider import PROVIDERS
from unsloth_cli.deploy.studio_client import StudioClient


IMAGE_TAG = "ghcr.io/unslothai/unsloth-base:cu128"

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


deploy_app = typer.Typer(
    help = "Launch Unsloth Studio on a cloud provider.",
    no_args_is_help = True,
)


@dataclass(frozen = True)
class Instance:
    id: str
    studio_url: str
    ssh: Optional[SshTarget]
    provider_name: str


@deploy_app.command("run")
def run(
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER, "--provider",
        help = f"Cloud provider to deploy to. Available: {', '.join(PROVIDERS)}.",
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
    provider_opt: list[str] = typer.Option(
        [], "--provider-opt",
        help = "Provider-specific setting as key=value (repeatable), e.g. "
               "--provider-opt datacenter=US-KS-2. See the provider's options.",
    ),
):
    """Launch Unsloth Studio on the chosen provider."""
    if not admin_password:
        _fail("Set UNSLOTH_ADMIN_PASSWORD or pass --admin-password.", code = 2)
    if len(admin_password) < MIN_ADMIN_PASSWORD_LENGTH:
        _fail(
            f"Admin password must be at least {MIN_ADMIN_PASSWORD_LENGTH} characters.",
            code = 2,
        )

    provider_cls = _provider_cls(provider_name)
    overrides = _parse_provider_opts(provider_opt)

    if model is None and not yes:
        model = _pick_model()
    need_local = _is_local_model(model)

    if need_local and not provider_cls.supports_local_model:
        _fail(
            f"{provider_name} can't upload a local model.\n"
            "Pass a Hugging Face id with --model, or load from the Studio UI "
            "after launch.",
            code = 2,
        )

    provider = _authenticate(
        provider_cls, overrides,
        need_local = need_local, interactive = not yes, persist = True,
    )

    chosen = _pick_gpu(provider, override = gpu, min_vram_gb = min_vram, yes = yes)

    typer.echo("")
    typer.echo(f"  Provider: {provider.name}")
    typer.echo(f"  GPU:      {chosen.name} ({chosen.vram_gb} GB) - ${chosen.cost_per_hour_usd:.3f}/hr")
    if model is not None:
        note = "  (local -- uploaded to provider storage after you confirm)" if need_local else ""
        typer.echo(f"  Model:    {model}{note}")
    typer.echo("")

    if not yes and not typer.confirm("Continue?", default = False):
        typer.echo("Aborted.")
        raise typer.Exit(0)

    staged: Optional[StagedModel] = None
    if need_local:
        staged = _stage_local_model(provider, Path(model).expanduser(), chosen)

    _deploy(
        provider,
        chosen,
        admin_password = admin_password,
        model = model,
        staged = staged,
        gguf_variant = gguf_variant,
        hf_token = hf_token,
    )


@deploy_app.command("stop")
def stop(
    instance_id: str = typer.Argument(..., help = "Instance id (printed when deploy succeeds)."),
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER, "--provider", help = "Provider the instance runs on.",
    ),
    pause: bool = typer.Option(
        False, "--pause",
        help = "Suspend instead of terminating (provider must support it; "
               "attached storage may still bill).",
    ),
):
    """Terminate the instance so billing stops. Pass --pause to suspend it instead."""
    provider = _authenticate(_provider_cls(provider_name), {}, interactive = False)
    try:
        if pause:
            if not provider.supports_pause:
                _fail(f"{provider.name} can't pause an instance; omit --pause to terminate.", code = 2)
            provider.pause(instance_id)
            typer.echo(f"Instance {instance_id} paused (may still incur storage billing).")
        else:
            provider.terminate(instance_id)
            typer.echo(f"Instance {instance_id} terminated.")
    except DeployError as e:
        _fail(str(e))


@deploy_app.command("delete-storage")
def delete_storage(
    storage_id: str = typer.Argument(..., help = "Storage id (printed when deploy uploads a local model)."),
    provider_name: str = typer.Option(
        DEFAULT_PROVIDER, "--provider", help = "Provider the storage lives on.",
    ),
):
    """Delete storage created for a local-model upload so it stops billing. Terminate the instance first."""
    provider = _authenticate(_provider_cls(provider_name), {}, interactive = False)
    try:
        provider.delete_storage(storage_id)
        typer.echo(f"Storage {storage_id} deleted.")
    except DeployError as e:
        _fail(str(e))


def _provider_cls(name: str) -> type[Provider]:
    if name not in PROVIDERS:
        _fail(f"Unknown provider '{name}'. Available: {', '.join(PROVIDERS)}.", code = 2)
    return PROVIDERS[name]


def _parse_provider_opts(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            _fail(f"--provider-opt expects key=value, got {pair!r}.", code = 2)
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _authenticate(
    provider_cls: type[Provider],
    overrides: dict[str, str],
    *,
    need_local: bool = False,
    interactive: bool = False,
    persist: bool = False,
) -> Provider:
    saved = store.load(provider_cls.name)
    resolved: dict[str, str] = dict(saved)

    for opt in provider_cls.option_schema():
        deferred = opt.needed_for == NEEDED_FOR_LOCAL_MODEL and not need_local
        if opt.key in overrides:
            value = overrides[opt.key]
        else:
            value = os.environ.get(opt.env) or saved.get(opt.key) or ""
        if not value and opt.required and not deferred:
            if interactive:
                value = typer.prompt(opt.help, hide_input = opt.secret, default = "").strip()
            if not value:
                _fail(
                    f"Missing {opt.env} -- {opt.help}.\n"
                    f"Set the env var, pass --provider-opt {opt.key}=..., "
                    "or run without --yes to be prompted.",
                    code = 2,
                )
        if value:
            resolved[opt.key] = value

    provider = provider_cls()
    try:
        provider.auth(resolved)
    except DeployError as e:
        _fail(str(e))

    if persist:
        _persist_options(provider_cls, resolved, previously = saved)
    return provider


def _persist_options(
    provider_cls: type[Provider], resolved: dict[str, str], *, previously: dict[str, str],
) -> None:
    keys = {opt.key for opt in provider_cls.option_schema()}
    to_save = {k: v for k, v in resolved.items() if k in keys}
    if not to_save or to_save == previously:
        return
    try:
        path = store.save(provider_cls.name, to_save)
    except OSError:
        return
    if not previously:
        typer.echo(f"  saved {provider_cls.name} settings to {path}")


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
    added: Optional[str] = None
    try:
        from unsloth_cli.commands.studio import _find_run_py

        run_py = _find_run_py()
        if run_py is None:
            return []
        backend = str(run_py.parent)
        if backend not in sys.path:
            sys.path.insert(0, backend)
            added = backend
        from utils.paths import outputs_root, exports_root

        return [outputs_root(), exports_root()]
    except Exception:
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass
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
    provider: Provider, *, override: Optional[str], min_vram_gb: int, yes: bool,
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
        if provider.reports_stock:
            in_stock = next((g for g in options if g.stock), None)
            if in_stock is not None:
                return in_stock
            typer.echo(
                "  warning: couldn't confirm any GPU has stock; trying the cheapest "
                f"({options[0].name}). If it fails to start, retry or pass --gpu.",
                err = True,
            )
        return options[0]

    shown = options[:MAX_GPU_CHOICES]
    typer.echo("")
    typer.echo("Available GPUs (cheapest first):")
    for i, gpu in enumerate(shown, start = 1):
        price = f"${gpu.cost_per_hour_usd:.3f}/hr"
        stock = f"stock: {gpu.stock or 'none':<6}   " if provider.reports_stock else ""
        typer.echo(
            f"  {i:2d}. {gpu.name:<24} {gpu.vram_gb:>3} GB   {price:<11} "
            f"{stock}({gpu.id})"
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


def _stage_local_model(provider: Provider, local: Path, gpu: Gpu) -> StagedModel:
    try:
        return provider.stage_local_model(local, gpu = gpu, log = typer.echo)
    except DeployError as e:
        _fail(str(e))


def _deploy(
    provider: Provider,
    gpu: Gpu,
    *,
    admin_password: str,
    model: Optional[str],
    staged: Optional[StagedModel],
    gguf_variant: Optional[str],
    hf_token: Optional[str],
) -> None:
    storage_id = staged.storage_id if staged else None
    if staged is not None:
        load_model: Optional[str] = staged.model_path
    else:
        load_model = model

    load_kwargs: dict = {}
    if staged is not None and model is not None:
        local = Path(model).expanduser()
        if local.is_dir() and (local / "adapter_config.json").is_file():
            load_kwargs["is_lora"] = True
            typer.echo("  detected adapter_config.json, loading as LoRA adapter")
    if gguf_variant:
        load_kwargs["gguf_variant"] = gguf_variant
    if hf_token:
        load_kwargs["hf_token"] = hf_token

    name = f"unsloth-studio-{int(time.time())}"
    typer.echo(f"Creating instance '{name}'...")
    try:
        instance_id = provider.create_instance(
            name = name,
            gpu = gpu,
            image = IMAGE_TAG,
            http_port = STUDIO_PORT,
            ssh_port = SSH_PORT if provider.supports_ssh else None,
            disk_gb = DEFAULT_DISK_GB,
            env = {"UNSLOTH_ADMIN_PASSWORD": admin_password},
            staged = staged,
        )
    except DeployError as e:
        if storage_id:
            _delete_storage_quietly(provider, storage_id)
            _fail(
                f"{e}\nCapacity changed after upload. Retry, pick a higher-stock "
                "GPU (the picker shows stock), or set the provider's placement."
            )
        _fail(str(e))
    typer.echo(f"  instance id: {instance_id}")

    try:
        typer.echo("Waiting for instance to start...")
        provider.wait_ready(instance_id, timeout_s = POD_RUNNING_TIMEOUT_S)
    except DeployError as e:
        _fail(
            f"Deploy failed: {e}\n"
            f"Instance {instance_id} may still be running and billing.\n"
            f"Stop it with: {_deploy_cmd('stop', instance_id, provider.name)}"
            + _storage_billing_note(storage_id, provider.name)
        )

    instance = Instance(
        id = instance_id,
        studio_url = provider.endpoint_url(instance_id, http_port = STUDIO_PORT),
        ssh = provider.get_ssh(instance_id) if provider.supports_ssh else None,
        provider_name = provider.name,
    )

    if load_model is None:
        _print_studio_ready(instance, gpu, storage_id)
    else:
        _serve_model(
            instance, gpu,
            bootstrap_password = admin_password,
            model = load_model,
            load_kwargs = load_kwargs,
            storage_id = storage_id,
        )

    if provider.deploy_note:
        typer.echo(f"  NOTE: {provider.deploy_note}\n")


def _serve_model(
    instance: Instance,
    gpu: Gpu,
    *,
    bootstrap_password: str,
    model: str,
    load_kwargs: dict,
    storage_id: Optional[str],
) -> None:
    client = StudioClient(instance.studio_url)
    rotated_password: Optional[str] = None

    try:
        typer.echo("Waiting for Studio API to come online...")
        client.wait_healthy(timeout_s = STUDIO_HEALTH_TIMEOUT_S)

        new_password = secrets.token_urlsafe(18)
        typer.echo("Rotating bootstrap password and minting credential...")
        client.login(username = DEFAULT_ADMIN_USERNAME, password = bootstrap_password)
        rotated_password = new_password
        client.change_password(current = bootstrap_password, new = new_password)
        api_key, credential_note = _mint_credential(client)

        typer.echo(f"Loading '{model}' on the instance (can take several minutes)...")
        client.load_model(model_path = model, **load_kwargs)

        _print_inference_ready(
            instance, gpu,
            api_key = api_key,
            credential_note = credential_note,
            model = model,
            admin_password = new_password,
            storage_id = storage_id,
        )
    except Exception as e:
        _fail(_stop_hint(
            e, instance.id,
            provider_name = instance.provider_name,
            admin_password = rotated_password,
            storage_id = storage_id,
        ))


def _is_local_model(model: Optional[str]) -> bool:
    if model is None:
        return False
    path = Path(model).expanduser()
    if not path.exists():
        return False
    if model.startswith(("/", "./", "../", "~")) or model.startswith(os.sep):
        return True
    if path.is_dir():
        return _model_kind(path) is not None
    return path.suffix.lower() == ".gguf"


def _delete_storage_quietly(provider: Provider, storage_id: Optional[str]) -> None:
    if not storage_id:
        return
    try:
        provider.delete_storage(storage_id)
    except DeployError:
        typer.echo(
            f"  warning: couldn't delete storage {storage_id}; "
            f"remove it with: {_deploy_cmd('delete-storage', storage_id, provider.name)}",
            err = True,
        )


def _storage_billing_note(storage_id: Optional[str], provider_name: str) -> str:
    if not storage_id:
        return ""
    return (
        f"\nStorage {storage_id} also persists (billed). Delete it with:"
        f"\n    {_deploy_cmd('delete-storage', storage_id, provider_name)}"
    )


def _mint_credential(client: StudioClient) -> tuple[str, str]:
    try:
        return client.create_api_key(name = f"deploy-{int(time.time())}"), ""
    except DeployError as e:
        if any(f"-> {code}:" in str(e) for code in (404, 405)):
            note = (
                "  (JWT credential, expires. Rebake the image with a newer\n"
                "   UNSLOTH_COMMIT to get permanent sk-unsloth-... keys.)"
            )
            return client.token, note
        raise


def _print_studio_ready(
    instance: Instance, gpu: Gpu, storage_id: Optional[str] = None,
) -> None:
    typer.echo("")
    typer.echo("Unsloth Studio instance is starting (Studio may take ~1 minute).")
    typer.echo(f"  Studio:  {instance.studio_url}")
    if instance.ssh is not None:
        typer.echo(f"  SSH:     ssh {instance.ssh.user}@{instance.ssh.host} -p {instance.ssh.port}")
    typer.echo("")
    _print_stop_hint(instance, gpu, storage_id)


def _print_inference_ready(
    instance: Instance,
    gpu: Gpu,
    *,
    api_key: str,
    credential_note: str,
    model: str,
    admin_password: str,
    storage_id: Optional[str] = None,
) -> None:
    base_url = f"{instance.studio_url}/v1"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]})

    typer.echo("")
    typer.echo("Inference endpoint ready.")
    typer.echo(f"    base_url:  {base_url}")
    typer.echo(f"    api_key:   {api_key}")
    if credential_note:
        typer.echo(credential_note)
    typer.echo(f"    model:     {model}")
    typer.echo(f"    admin_pw:  {admin_password}   (Studio UI at {instance.studio_url})")
    typer.echo("")
    typer.echo("  Test it:")
    typer.echo(f"    curl {base_url}/chat/completions \\")
    typer.echo(f"      -H 'Authorization: Bearer {api_key}' \\")
    typer.echo("      -H 'Content-Type: application/json' \\")
    typer.echo(f"      -d '{body}'")
    typer.echo("")
    _print_stop_hint(instance, gpu, storage_id)


def _print_stop_hint(
    instance: Instance, gpu: Gpu, storage_id: Optional[str] = None,
) -> None:
    typer.echo(f"  STOP THIS INSTANCE WHEN DONE - billed at ${gpu.cost_per_hour_usd:.3f}/hr.")
    typer.echo(f"      {_deploy_cmd('stop', instance.id, instance.provider_name)}")
    if storage_id:
        typer.echo(f"  Storage {storage_id} persists (billed). Delete when done:")
        typer.echo(f"      {_deploy_cmd('delete-storage', storage_id, instance.provider_name)}")
    typer.echo("")


def _stop_hint(
    err: Exception,
    instance_id: str,
    *,
    provider_name: str,
    admin_password: Optional[str] = None,
    storage_id: Optional[str] = None,
) -> str:
    lines = [f"Deploy bootstrap failed: {err}"]
    if admin_password is not None:
        lines.append(
            f"The Studio admin password may have been rotated to: {admin_password}\n"
            "  (save it -- if the original no longer works for the Studio UI, use this)."
        )
    lines += [
        f"Instance {instance_id} is still running and billing.",
        f"Stop it with: {_deploy_cmd('stop', instance_id, provider_name)}",
    ]
    note = _storage_billing_note(storage_id, provider_name)
    return "\n".join(lines) + note


def _deploy_cmd(verb: str, arg: str, provider_name: str) -> str:
    flag = "" if provider_name == DEFAULT_PROVIDER else f" --provider {provider_name}"
    return f"unsloth deploy {verb}{flag} {arg}"


def _fail(msg: str, code: int = 1):
    typer.echo(msg, err = True)
    raise typer.Exit(code)
