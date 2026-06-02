# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth deploy`. The runpod SDK is imported lazily inside the
deploy package, so we stub it via sys.modules to run these offline."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse = True)
def _isolate_deploy_config(monkeypatch, tmp_path):
    """Keep the env->config->reuse credential store out of the user's real
    ~/.config during tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdgconfig"))


# `runpod.get_gpus()` returns only id/displayName/memoryInGb; price data is
# only on `runpod.get_gpu(id)`. The fixtures mirror that split, and include a
# spot-only GPU and a price-less GPU that listing must drop.
_CATALOG = [
    {"id": "A4000",    "displayName": "RTX A4000", "memoryInGb": 16},
    {"id": "A5000",    "displayName": "RTX A5000", "memoryInGb": 24},
    {"id": "4090",     "displayName": "RTX 4090",  "memoryInGb": 24},
    {"id": "A100-80",  "displayName": "A100 80GB", "memoryInGb": 80},
    {"id": "spotonly", "displayName": "Spot only", "memoryInGb": 24},
    {"id": "noprice",  "displayName": "No price",  "memoryInGb": 24},
]
_DETAILS = {
    "A4000":    {"lowestPrice": {"uninterruptablePrice": 0.20}},
    "A5000":    {"lowestPrice": {"uninterruptablePrice": 0.36}},
    "4090":     {"lowestPrice": {"uninterruptablePrice": 0.44}},
    "A100-80":  {"lowestPrice": {"uninterruptablePrice": 1.89}},
    "spotonly": {"lowestPrice": {"minimumBidPrice": 0.10}},
    "noprice":  {"lowestPrice": {}},
}


def _stub_runpod(monkeypatch, **behavior):
    stub = types.ModuleType("runpod")
    stub.api_key = None
    for name, fn in behavior.items():
        setattr(stub, name, fn)
    monkeypatch.setitem(sys.modules, "runpod", stub)
    return stub


def _stub_with_catalog(monkeypatch, **overrides):
    behavior = {
        "get_gpus": lambda: _CATALOG,
        "get_gpu": lambda gid: _DETAILS.get(gid, {}),
    }
    behavior.update(overrides)
    return _stub_runpod(monkeypatch, **behavior)


def _running_pod(pid):
    return {"desiredStatus": "RUNNING", "runtime": {"ports": []}}


def _gpu(gid = "A5000", vram = 24, price = 0.36):
    from unsloth_cli.deploy import Gpu
    return Gpu(id = gid, name = gid, vram_gb = vram, cost_per_hour_usd = price)


class _FakeStudioClient:
    """Records the bootstrap calls so we can assert order without HTTP."""

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.calls = []
        self._token = None

    @property
    def token(self):
        assert self._token, "token accessed before login"
        return self._token

    def wait_healthy(self, timeout_s):
        self.calls.append(("wait_healthy",))

    def login(self, username, password):
        self.calls.append(("login", username, password))
        self._token = "jwt-1"

    def change_password(self, current, new):
        self.calls.append(("change_password", current, new))
        self._token = "jwt-2"

    def create_api_key(self, name):
        self.calls.append(("create_api_key", name))
        return "sk-unsloth-fake-key"

    def load_model(self, model_path, **kwargs):
        self.calls.append(("load_model", model_path, kwargs))


def _fake_studio(store):
    def factory(base_url):
        return store.setdefault("client", _FakeStudioClient(base_url))
    return factory


# ---------------------------------------------------------------------------
# Image + RunPod provider
# ---------------------------------------------------------------------------


def test_image_tag_lives_under_unsloth_org():
    """The deploy image must be the org image, not a personal namespace."""
    from unsloth_cli.commands import deploy
    assert deploy.IMAGE_TAG.startswith("ghcr.io/unslothai/")


def test_auth_requires_api_key(monkeypatch):
    from unsloth_cli.deploy import DeployError
    from unsloth_cli.deploy.runpod_client import RunPod
    with pytest.raises(DeployError, match = "RUNPOD_API_KEY"):
        RunPod().auth({})


def test_list_gpus_drops_spot_and_priceless_and_sorts_by_cost(monkeypatch):
    from unsloth_cli.deploy.runpod_client import RunPod
    _stub_with_catalog(monkeypatch)
    p = RunPod(); p.auth({"api_key": "rpa_x"})

    gpus = p.list_gpus(min_vram_gb = 24)
    ids = [g.id for g in gpus]
    assert "spotonly" not in ids and "noprice" not in ids  # no on-demand price
    assert "A4000" not in ids                               # below min VRAM
    assert ids[0] == "A5000"                                # cheapest first
    assert [g.cost_per_hour_usd for g in gpus] == sorted(g.cost_per_hour_usd for g in gpus)


def test_create_instance_mounts_volume_at_workspace_and_opens_ports(monkeypatch):
    """unsloth-base writes under /workspace, so the volume must mount there;
    both the Studio HTTP port and SSH must be exposed."""
    from unsloth_cli.deploy.runpod_client import RunPod
    captured = {}
    _stub_runpod(monkeypatch, create_pod = lambda **kw: captured.update(kw) or {"id": "pod-abc"})

    p = RunPod(); p.auth({"api_key": "rpa_x"})
    instance_id = p.create_instance(
        name = "test", gpu = _gpu(), image = "img:tag",
        http_port = 8000, ssh_port = 22, disk_gb = 100,
        env = {"UNSLOTH_ADMIN_PASSWORD": "secret"},
    )
    assert instance_id == "pod-abc"
    assert captured["volume_mount_path"] == "/workspace"
    assert "8000/http" in captured["ports"]
    assert "22/tcp" in captured["ports"]


def test_wait_ready_waits_for_runtime(monkeypatch):
    """desiredStatus flips to RUNNING before the container is live; we must
    wait until `runtime` is populated."""
    from unsloth_cli.deploy.runpod_client import RunPod
    states = iter([
        {"desiredStatus": "RUNNING", "runtime": None},
        {"desiredStatus": "RUNNING", "runtime": {"ports": []}},
    ])
    _stub_runpod(monkeypatch, get_pod = lambda pid: next(states))
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = RunPod(); p.auth({"api_key": "rpa_x"})
    p.wait_ready("pod-x", timeout_s = 30)  # returns without raising


def test_endpoint_url_uses_https_proxy_never_plaintext_ip(monkeypatch):
    """The Studio URL must be the TLS proxy, never the pod's plaintext-http direct
    IP: the deploy bootstrap and the user's login send the admin password over it."""
    from unsloth_cli.deploy.runpod_client import RunPod
    assert RunPod().endpoint_url("podxyz", http_port = 8000) == "https://podxyz-8000.proxy.runpod.net"

    # Even when the pod exposes a public direct IP, we still hand back the https proxy.
    fake_pod = {"runtime": {"ports": [
        {"privatePort": 8000, "publicPort": 18000, "ip": "1.2.3.4", "isIpPublic": True},
    ]}}
    _stub_runpod(monkeypatch, get_pod = lambda pid: fake_pod)
    p = RunPod(); p.auth({"api_key": "rpa_x"})
    url = p.endpoint_url("podxyz", http_port = 8000)
    assert url == "https://podxyz-8000.proxy.runpod.net"
    assert url.startswith("https://")


def test_stop_terminates_by_default_pause_suspends(monkeypatch):
    """`stop` fully terminates so users aren't surprised by storage billing;
    `--pause` only suspends."""
    calls = []
    _stub_runpod(
        monkeypatch,
        terminate_pod = lambda pid: calls.append(("terminate", pid)),
        stop_pod = lambda pid: calls.append(("stop", pid)),
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    from unsloth_cli import app

    assert CliRunner().invoke(app, ["deploy", "stop", "pod-a"]).exit_code == 0
    assert CliRunner().invoke(app, ["deploy", "stop", "--pause", "pod-b"]).exit_code == 0
    assert calls == [("terminate", "pod-a"), ("stop", "pod-b")]


# ---------------------------------------------------------------------------
# Provider contract + capabilities
# ---------------------------------------------------------------------------


def test_runpod_declares_its_capabilities():
    from unsloth_cli.deploy.runpod_client import RunPod
    assert RunPod.supports_ssh and RunPod.supports_pause and RunPod.supports_local_model


def test_provider_optional_methods_raise_when_unsupported():
    """A provider that doesn't override an optional method gets a clear
    'unsupported' DeployError instead of an AttributeError."""
    from unsloth_cli.deploy import DeployError
    p = _RealNoStorage()
    with pytest.raises(DeployError, match = "SSH"):
        p.get_ssh("i-1")
    with pytest.raises(DeployError, match = "local model"):
        p.stage_local_model(Path("."), gpu = _gpu())
    with pytest.raises(DeployError, match = "storage"):
        p.delete_storage("s-1")
    with pytest.raises(DeployError, match = "paus"):
        p.pause("i-1")


def _register_nostorage(monkeypatch):
    from unsloth_cli.commands import deploy
    monkeypatch.setitem(deploy.PROVIDERS, "nostorage", _RealNoStorage)


def test_local_model_rejected_when_provider_lacks_storage(monkeypatch, tmp_path):
    """Selecting a local model on a provider without storage fails up front with
    a helpful message -- before authenticating or creating anything."""
    from unsloth_cli.commands import deploy

    local = tmp_path / "m"
    local.mkdir()
    (local / "adapter_config.json").write_text("{}")

    _register_nostorage(monkeypatch)
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "secret-pw")
    reached = []
    monkeypatch.setattr(deploy, "_deploy", lambda *a, **k: reached.append(1))

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--provider", "nostorage", "--model", str(local),
    ])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "can't upload a local model" in combined
    assert not reached


def test_unknown_provider_is_rejected_with_available_list(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "secret-pw")
    from unsloth_cli import app
    result = CliRunner().invoke(app, ["deploy", "run", "--yes", "--provider", "nope"])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "Unknown provider" in combined and "runpod" in combined


def test_get_provider_rejects_unknown_and_lists_available():
    """An unknown --provider fails with the set of registered names."""
    from unsloth_cli.deploy import DeployError
    from unsloth_cli.deploy.base import Provider
    from unsloth_cli.deploy.provider import PROVIDERS, get_provider

    assert "runpod" in PROVIDERS
    assert isinstance(get_provider("runpod"), Provider)
    with pytest.raises(DeployError, match = "Unknown provider"):
        get_provider("nope")


# ---------------------------------------------------------------------------
# Credential store: env -> config -> reuse
# ---------------------------------------------------------------------------


def test_provider_options_resolved_from_env_then_persisted_and_reused(monkeypatch):
    """First run reads the api key from the env and saves it; a later run with no
    env var reads it back from the saved config."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import store
    from unsloth_cli.deploy.runpod_client import RunPod
    _stub_with_catalog(monkeypatch)

    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_from_env")
    p1 = deploy._authenticate(RunPod, {}, need_local = False, interactive = False, persist = True)
    assert p1._api_key == "rpa_from_env"
    assert store.load("runpod").get("api_key") == "rpa_from_env"  # persisted

    # No env this time -- must come from the saved config.
    monkeypatch.delenv("RUNPOD_API_KEY", raising = False)
    p2 = deploy._authenticate(RunPod, {}, need_local = False, interactive = False)
    assert p2._api_key == "rpa_from_env"


def test_provider_opt_override_beats_env(monkeypatch):
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy.runpod_client import RunPod
    _stub_with_catalog(monkeypatch)

    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_from_env")
    p = deploy._authenticate(
        RunPod, {"api_key": "rpa_override"}, need_local = False, interactive = False,
    )
    assert p._api_key == "rpa_override"


# ---------------------------------------------------------------------------
# `deploy run` + pickers
# ---------------------------------------------------------------------------


def test_run_rejects_short_admin_password(monkeypatch):
    """Studio's change-password requires >= 8 chars; a shorter bootstrap
    password would brick the instance, so reject it up front."""
    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "12")
    from unsloth_cli import app
    result = CliRunner().invoke(app, ["deploy", "run", "--yes"])
    assert result.exit_code != 0
    assert "at least 8" in result.output


def test_run_yes_picks_cheapest_and_forwards_model(monkeypatch):
    """`--yes` auto-picks the cheapest fitting GPU and forwards --model and
    friends straight through to the deploy step."""
    from unsloth_cli.commands import deploy
    captured = {}

    def fake_deploy(provider, gpu, **kw):
        captured["gpu"] = gpu
        captured.update(kw)

    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "secret-pw")
    monkeypatch.setattr(deploy, "_deploy", fake_deploy)

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes",
        "--model", "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "--gguf-variant", "Q4_K_M",
        "--hf-token", "hf_xyz",
    ])
    assert result.exit_code == 0, result.output
    assert captured["gpu"].id == "A5000"
    assert captured["model"] == "unsloth/Llama-3.2-1B-Instruct-GGUF"
    assert captured["gguf_variant"] == "Q4_K_M"
    assert captured["hf_token"] == "hf_xyz"
    assert captured["staged"] is None


def test_discover_local_models_finds_finetunes_and_skips_other_dirs(tmp_path):
    """LoRA at the root, HF checkpoints under outputs/, GGUF under merged_model/;
    plain directories are ignored."""
    from unsloth_cli.commands.deploy import _discover_local_models

    (tmp_path / "lora_model").mkdir()
    (tmp_path / "lora_model" / "adapter_config.json").write_text("{}")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "checkpoint-100").mkdir()
    (tmp_path / "outputs" / "checkpoint-100" / "config.json").write_text("{}")
    (tmp_path / "merged_model").mkdir()
    (tmp_path / "merged_model" / "model.gguf").write_bytes(b"")
    (tmp_path / "notes").mkdir()  # not a model

    found = {(p.name, kind) for p, kind in _discover_local_models(tmp_path)}
    assert ("lora_model", "LoRA adapter") in found
    assert ("checkpoint-100", "HF model") in found
    assert ("merged_model", "GGUF") in found
    assert not any(name == "notes" for name, _ in found)


def test_discover_includes_studio_outputs_and_exports(monkeypatch, tmp_path):
    """Models trained or exported via the Studio UI live outside the cwd under
    the studio roots; the picker must surface them too, walking one level in."""
    from unsloth_cli.commands import deploy

    studio_out = tmp_path / "studio" / "outputs"
    studio_exp = tmp_path / "studio" / "exports"
    run = studio_out / "unsloth_Llama-3.2-3B_1771"
    run.mkdir(parents = True)
    (run / "adapter_config.json").write_text("{}")
    export = studio_exp / "my-merged"
    export.mkdir(parents = True)
    (export / "config.json").write_text("{}")

    monkeypatch.setattr(deploy, "_studio_output_roots", lambda: [studio_out, studio_exp])

    empty_cwd = tmp_path / "elsewhere"
    empty_cwd.mkdir()
    found = {(p.name, kind) for p, kind in deploy._discover_local_models(empty_cwd)}
    assert ("unsloth_Llama-3.2-3B_1771", "LoRA adapter") in found
    assert ("my-merged", "HF model") in found


def test_studio_output_roots_reuse_backend_resolver(monkeypatch, tmp_path):
    """deploy locates Studio models via the backend's own path resolvers (not a
    hardcoded path), so they track UNSLOTH_STUDIO_HOME and never drift."""
    from unsloth_cli.commands.deploy import _studio_output_roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    roots = _studio_output_roots()
    if not roots:
        pytest.skip("Studio backend not importable in this environment")
    assert tmp_path.resolve() / "outputs" in roots
    assert tmp_path.resolve() / "exports" in roots


def test_picker_offers_local_model_and_user_selects_it(monkeypatch, tmp_path):
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import StagedModel
    from unsloth_cli.deploy.runpod_client import RunPod

    (tmp_path / "lora_model").mkdir()
    (tmp_path / "lora_model" / "adapter_config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    captured = {}
    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "secret-pw")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY_ID", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_ACCESS_KEY", "rps_y")
    # Stage behind the provider seam -- no network volume / GraphQL in the test.
    monkeypatch.setattr(
        RunPod, "stage_local_model",
        lambda self, local, *, gpu, log = None: StagedModel(
            model_path = "/workspace/uploads/lora_model",
            storage_id = "vol-1", summary = "network volume vol-1", placement = "DC",
        ),
    )
    monkeypatch.setattr(deploy, "_deploy", lambda provider, gpu, **kw: captured.update(kw))

    from unsloth_cli import app
    # pick model 1 (the local LoRA), GPU 1, then confirm
    result = CliRunner().invoke(app, ["deploy", "run"], input = "1\n1\ny\n")
    assert result.exit_code == 0, result.output
    assert captured["model"].endswith("/lora_model")
    assert captured["staged"].storage_id == "vol-1"
    assert "LoRA adapter" in result.output


def test_local_model_not_staged_until_user_confirms(monkeypatch, tmp_path):
    """Creating a volume and uploading multi-GB weights costs time and money, so
    it must not happen before the user confirms."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy.runpod_client import RunPod

    local = tmp_path / "lora"
    local.mkdir()
    (local / "adapter_config.json").write_text("{}")

    staged_calls = []
    reached = []
    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "secret-pw")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY_ID", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_ACCESS_KEY", "rps_y")
    monkeypatch.setattr(RunPod, "stage_local_model", lambda self, *a, **k: staged_calls.append(1))
    monkeypatch.setattr(deploy, "_deploy", lambda *a, **k: reached.append(1))

    from unsloth_cli import app
    # pick GPU 1, then decline the confirmation
    result = CliRunner().invoke(app, ["deploy", "run", "--model", str(local)], input = "1\nn\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert not staged_calls   # nothing uploaded before the user said yes
    assert not reached


def test_stop_does_not_persist_credentials(monkeypatch):
    """Lifecycle commands authenticate but must not write the credential file --
    only `run` persists (the 'set it up once' path)."""
    from unsloth_cli.deploy import store
    _stub_runpod(monkeypatch, terminate_pod = lambda pid: None)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")

    from unsloth_cli import app
    assert CliRunner().invoke(app, ["deploy", "stop", "pod-a"]).exit_code == 0
    assert store.load("runpod") == {}


# ---------------------------------------------------------------------------
# Bootstrap chain
# ---------------------------------------------------------------------------


def test_bootstrap_drives_full_chain_and_prints_endpoint(monkeypatch):
    """`run --model` drives health -> login -> rotate password -> key -> load,
    then prints the endpoint, key, rotated admin password and a stop hint."""
    from unsloth_cli.commands import deploy
    store = {}
    _stub_with_catalog(monkeypatch, create_pod = lambda **kw: {"id": "pod-xyz"}, get_pod = _running_pod)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "bootstrap-pw")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(deploy, "StudioClient", _fake_studio(store))

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--model", "unsloth/Llama-3.2-1B-Instruct",
    ])
    assert result.exit_code == 0, result.output

    client = store["client"]
    assert [c[0] for c in client.calls] == [
        "wait_healthy", "login", "change_password", "create_api_key", "load_model",
    ]
    # Login uses the bootstrap password; the rotation must change it to something else.
    assert client.calls[1] == ("login", "unsloth", "bootstrap-pw")
    assert client.calls[2][1] == "bootstrap-pw" and client.calls[2][2] != "bootstrap-pw"

    assert "sk-unsloth-fake-key" in result.output
    assert "unsloth/Llama-3.2-1B-Instruct" in result.output
    assert "unsloth deploy stop pod-xyz" in result.output


def test_bootstrap_uploads_local_model_via_network_volume(monkeypatch, tmp_path):
    """A local --model is uploaded to a RunPod network volume over S3 (no SSH,
    no Hugging Face); the pod mounts that volume pinned to its datacenter, and
    Studio loads the model by its /workspace path with is_lora set."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import runpod_storage

    local = tmp_path / "my-lora"
    local.mkdir()
    (local / "adapter_config.json").write_text("{}")

    store = {}
    pod_kwargs = {}
    vol_kwargs = {}
    uploads = []

    _stub_with_catalog(
        monkeypatch,
        create_pod = lambda **kw: pod_kwargs.update(kw) or {"id": "p"},
        get_pod = _running_pod,
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "bootstrap-pw")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY_ID", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_ACCESS_KEY", "rps_y")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(
        runpod_storage, "create_network_volume",
        lambda client, **kw: vol_kwargs.update(kw) or "vol-123",
    )
    monkeypatch.setattr(
        runpod_storage, "upload_path",
        lambda local_path, **kw: uploads.append((local_path, kw)),
    )
    monkeypatch.setattr(deploy, "StudioClient", _fake_studio(store))

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--model", str(local),
        "--provider-opt", "datacenter=US-KS-2",
    ])
    assert result.exit_code == 0, result.output

    # Volume created in the chosen datacenter; model uploaded to it over S3.
    assert vol_kwargs["datacenter_id"] == "US-KS-2"
    assert len(uploads) == 1
    _up_local, up_kw = uploads[0]
    assert up_kw["volume_id"] == "vol-123"
    assert up_kw["datacenter"] == "US-KS-2"
    assert up_kw["prefix"] == "uploads/my-lora"
    assert up_kw["access_key"] == "user_x" and up_kw["secret_key"] == "rps_y"

    # The pod mounts that volume, pinned to the volume's datacenter.
    assert pod_kwargs["network_volume_id"] == "vol-123"
    assert pod_kwargs["data_center_id"] == "US-KS-2"

    # Studio loads the model from its on-volume path, flagged as a LoRA adapter.
    load = [c for c in store["client"].calls if c[0] == "load_model"][0]
    assert load[1] == "/workspace/uploads/my-lora"
    assert load[2].get("is_lora") is True

    # The user is told how to delete the storage so it stops billing.
    assert "delete-storage vol-123" in result.output


def test_run_requires_storage_creds_for_local_model(monkeypatch, tmp_path):
    """Deploying a local model needs the provider's storage credentials; without
    them we fail up front -- before creating any billing instance."""
    from unsloth_cli.commands import deploy

    local = tmp_path / "m"
    local.mkdir()
    (local / "adapter_config.json").write_text("{}")

    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "secret-pw")
    monkeypatch.delenv("RUNPOD_S3_ACCESS_KEY_ID", raising = False)
    monkeypatch.delenv("RUNPOD_S3_SECRET_ACCESS_KEY", raising = False)
    reached = []
    monkeypatch.setattr(deploy, "_deploy", lambda *a, **k: reached.append(1))

    from unsloth_cli import app
    result = CliRunner().invoke(app, ["deploy", "run", "--yes", "--model", str(local)])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "RUNPOD_S3_ACCESS_KEY_ID" in combined
    assert not reached  # bailed before creating an instance


def test_s3_upload_targets_per_datacenter_endpoint_and_volume_bucket(monkeypatch, tmp_path):
    """upload_path hits the datacenter's S3 endpoint, uses the volume id as the
    bucket, and keys each file under <prefix>/<relpath>."""
    from unsloth_cli.deploy import runpod_storage as storage

    client_kwargs = {}
    puts = []

    class _FakeS3:
        def upload_file(self, filename, bucket, key):
            puts.append((bucket, key))

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: client_kwargs.update(k) or _FakeS3()
    cfg_mod = types.ModuleType("botocore.config")
    cfg_mod.Config = lambda **k: ("cfg", k)
    exc_mod = types.ModuleType("botocore.exceptions")
    exc_mod.BotoCoreError = type("BotoCoreError", (Exception,), {})
    exc_mod.ClientError = type("ClientError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", types.ModuleType("botocore"))
    monkeypatch.setitem(sys.modules, "botocore.config", cfg_mod)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exc_mod)

    src = tmp_path / "model"
    src.mkdir()
    (src / "a.txt").write_text("x")
    (src / "sub").mkdir()
    (src / "sub" / "b.bin").write_bytes(b"\x00")

    storage.upload_path(
        src, volume_id = "vol-9", datacenter = "US-KS-2",
        access_key = "user_x", secret_key = "rps_y", prefix = "uploads/model",
    )

    assert client_kwargs["endpoint_url"] == "https://s3api-us-ks-2.runpod.io"
    assert client_kwargs["region_name"] == "US-KS-2"
    assert client_kwargs["aws_access_key_id"] == "user_x"
    assert set(puts) == {
        ("vol-9", "uploads/model/a.txt"),
        ("vol-9", "uploads/model/sub/b.bin"),
    }


def test_volume_size_gb_floors_small_models(tmp_path):
    """A tiny model still provisions at least the floor so Studio has room to
    write its HF cache / llama.cpp build on the same volume."""
    from unsloth_cli.deploy.runpod_client import _volume_size_gb, VOLUME_MIN_GB

    small = tmp_path / "m"
    small.mkdir()
    (small / "f").write_bytes(b"0" * 1000)
    assert _volume_size_gb(small) == VOLUME_MIN_GB


def test_bootstrap_failure_shows_stop_hint(monkeypatch):
    """Any bootstrap failure must tell the user how to stop the billing instance."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import DeployError

    class FailingClient(_FakeStudioClient):
        def login(self, username, password):
            raise DeployError("synthetic 401")

    _stub_with_catalog(monkeypatch, create_pod = lambda **kw: {"id": "pod-fail"}, get_pod = _running_pod)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "bootstrap-pw")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(deploy, "StudioClient", lambda base_url: FailingClient(base_url))

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--model", "unsloth/Llama-3.2-1B-Instruct",
    ])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "synthetic 401" in combined
    assert "unsloth deploy stop pod-fail" in combined


def test_bootstrap_failure_after_rotation_surfaces_new_password(monkeypatch):
    """A failure *after* the password is rotated must print the rotated
    password; otherwise the user is locked out of the Studio UI on a billing
    instance (the bootstrap password they passed in no longer works)."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import DeployError

    class FailAfterRotate(_FakeStudioClient):
        def load_model(self, model_path, **kwargs):
            raise DeployError("synthetic load failure")

    store = {}
    _stub_with_catalog(monkeypatch, create_pod = lambda **kw: {"id": "pod-late"}, get_pod = _running_pod)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "bootstrap-pw")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(deploy, "StudioClient", lambda base_url: store.setdefault("client", FailAfterRotate(base_url)))

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--model", "unsloth/Llama-3.2-1B-Instruct",
    ])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    # The rotated password is the `new` arg recorded by change_password.
    rotated = [c for c in store["client"].calls if c[0] == "change_password"][0][2]
    assert rotated and rotated in combined
    assert "rotated to" in combined
    assert "unsloth deploy stop pod-late" in combined


# ---------------------------------------------------------------------------
# Review hardening: billing safety, lockout, credentials, provider hints
# ---------------------------------------------------------------------------


def test_wait_ready_tolerates_transient_poll_errors(monkeypatch):
    """A transient get_pod error mid-poll must not abort the wait and orphan a
    billing pod; we keep polling until it is actually running."""
    from unsloth_cli.deploy.runpod_client import RunPod
    seen = {"n": 0}

    def flaky(pid):
        seen["n"] += 1
        if seen["n"] == 1:
            raise RuntimeError("transient blip")
        return {"desiredStatus": "RUNNING", "runtime": {"ports": []}}

    _stub_runpod(monkeypatch, get_pod = flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = RunPod(); p.auth({"api_key": "rpa_x"})
    p.wait_ready("pod-x", timeout_s = 30)  # returns without raising
    assert seen["n"] >= 2


def test_change_password_tolerates_missing_token(monkeypatch):
    """change-password succeeding but returning no new token must not raise: a
    KeyError here would strand the user on a billing pod whose admin password has
    already rotated, without ever surfacing the new one."""
    from unsloth_cli.deploy.studio_client import StudioClient
    c = StudioClient("http://x")
    c._token = "old"
    monkeypatch.setattr(c, "_post", lambda *a, **k: {})  # 2xx, empty body
    c.change_password(current = "a", new = "b")  # does not raise
    assert c.token == "old"


def test_stage_single_file_model_avoids_double_nesting(monkeypatch, tmp_path):
    """A single-file model (e.g. a .gguf) must upload under uploads/<file> and load
    from /workspace/uploads/<file> -- not uploads/<file>/<file>, which the pod
    can't find."""
    from unsloth_cli.deploy import runpod_storage
    from unsloth_cli.deploy.runpod_client import RunPod

    model = tmp_path / "model.gguf"
    model.write_bytes(b"0" * 10)

    uploads = []
    _stub_runpod(monkeypatch)
    monkeypatch.setattr(runpod_storage, "create_network_volume", lambda client, **kw: "vol-1")
    monkeypatch.setattr(runpod_storage, "upload_path", lambda local_path, **kw: uploads.append(kw))

    p = RunPod()
    p.auth({
        "api_key": "rpa_x", "s3_access_key": "user_x",
        "s3_secret_key": "rps_y", "datacenter": "US-KS-2",
    })
    staged = p.stage_local_model(model, gpu = _gpu())
    assert uploads[0]["prefix"] == "uploads"            # not "uploads/model.gguf"
    assert staged.model_path == "/workspace/uploads/model.gguf"


def test_stage_deletes_volume_when_upload_fails(monkeypatch, tmp_path):
    """An upload failure (even a non-DeployError) must delete the just-created
    network volume so it doesn't keep billing with a half-finished upload."""
    from unsloth_cli.deploy import runpod_storage
    from unsloth_cli.deploy.runpod_client import RunPod

    model = tmp_path / "m"
    model.mkdir()
    (model / "adapter_config.json").write_text("{}")

    deleted = []
    _stub_runpod(monkeypatch)
    monkeypatch.setattr(runpod_storage, "create_network_volume", lambda client, **kw: "vol-9")
    monkeypatch.setattr(runpod_storage, "delete_network_volume", lambda client, vid: deleted.append(vid))

    def boom(local_path, **kw):
        raise RuntimeError("network died mid-upload")

    monkeypatch.setattr(runpod_storage, "upload_path", boom)

    p = RunPod()
    p.auth({
        "api_key": "rpa_x", "s3_access_key": "user_x",
        "s3_secret_key": "rps_y", "datacenter": "US-KS-2",
    })
    with pytest.raises(RuntimeError):
        p.stage_local_model(model, gpu = _gpu())
    assert deleted == ["vol-9"]


def test_saved_config_is_user_only(monkeypatch, tmp_path):
    """The credential file holds tokens/secrets, so it must be 0600 with no
    group/other access -- never world-readable, even briefly."""
    import stat as _stat
    from unsloth_cli.deploy import store
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = store.save("runpod", {"api_key": "rpa_secret"})
    mode = _stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    assert not (mode & 0o077)


def test_local_model_opts_persisted_even_without_local_model(monkeypatch):
    """S3 creds supplied up front (env / --provider-opt) are saved for reuse even
    on a non-local deploy, instead of being silently dropped at auth time."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import store
    from unsloth_cli.deploy.runpod_client import RunPod
    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY_ID", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_ACCESS_KEY", "rps_y")
    deploy._authenticate(RunPod, {}, need_local = False, interactive = False, persist = True)
    saved = store.load("runpod")
    assert saved.get("s3_access_key") == "user_x"
    assert saved.get("s3_secret_key") == "rps_y"


def test_pick_gpu_yes_skips_out_of_stock(monkeypatch):
    """--yes must pick the cheapest GPU that actually has stock, not a cheaper
    sold-out one (stock is None once out-of-stock bands are filtered)."""
    from unsloth_cli.commands.deploy import _pick_gpu
    from unsloth_cli.deploy import Gpu

    class _P:
        name = "runpod"
        reports_stock = True
        def list_gpus(self, min_vram_gb = 0):
            return [
                Gpu(id = "cheap-soldout", name = "c", vram_gb = 24, cost_per_hour_usd = 0.2, stock = None),
                Gpu(id = "instock", name = "i", vram_gb = 24, cost_per_hour_usd = 0.4, stock = "High"),
            ]

    chosen = _pick_gpu(_P(), override = None, min_vram_gb = 24, yes = True)
    assert chosen.id == "instock"


def test_global_stock_drops_out_of_stock_bands(monkeypatch):
    """RunPod returns out-of-stock statuses alongside High/Medium/Low; only the
    real-capacity bands survive, so the picker shows 'none' for the rest and
    --yes won't auto-pick them."""
    from unsloth_cli.deploy.runpod_client import RunPod

    rows = [
        {"id": "A", "lowestPrice": {"stockStatus": "High"}},
        {"id": "B", "lowestPrice": {"stockStatus": "Low"}},
        {"id": "C", "lowestPrice": {"stockStatus": "Out of Stock"}},
        {"id": "D", "lowestPrice": {"stockStatus": None}},
    ]
    api_mod = types.ModuleType("runpod.api")
    graphql_mod = types.ModuleType("runpod.api.graphql")
    graphql_mod.run_graphql_query = lambda q: {"data": {"gpuTypes": rows}}
    _stub_runpod(monkeypatch)
    monkeypatch.setitem(sys.modules, "runpod.api", api_mod)
    monkeypatch.setitem(sys.modules, "runpod.api.graphql", graphql_mod)

    p = RunPod(); p.auth({"api_key": "rpa_x"})
    assert p._global_stock() == {"A": "High", "B": "Low"}


def test_deploy_cmd_carries_nondefault_provider():
    """Lifecycle hints must carry --provider for non-default providers so the
    printed stop / delete-storage commands target the right cloud."""
    from unsloth_cli.commands.deploy import DEFAULT_PROVIDER, _deploy_cmd
    assert _deploy_cmd("stop", "pod-1", DEFAULT_PROVIDER) == "unsloth deploy stop pod-1"
    assert _deploy_cmd("delete-storage", "vol-1", "modal") == (
        "unsloth deploy delete-storage --provider modal vol-1"
    )


def test_is_gateway_timeout_ignores_timeout_word_in_response_body():
    """A real HTTP error whose body merely mentions 'timeout' must not be treated
    as a retryable gateway cut-off -- that would mask a genuine load failure."""
    from unsloth_cli.deploy.studio_client import _is_gateway_timeout
    assert not _is_gateway_timeout(
        "POST /api/inference/load -> 400: {'detail': 'request timeout in config'}"
    )
    # A genuine Cloudflare 524 and a transport-level read timeout still count.
    assert _is_gateway_timeout("POST /api/inference/load -> 524: upstream gone")
    assert _is_gateway_timeout("POST /api/inference/load failed: read timed out")


def test_mint_credential_falls_back_only_on_real_404_405():
    """_mint_credential drops to the JWT only on an actual 404/405 status, not when
    those digits merely appear inside some other error's body."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import DeployError

    class _Old:
        token = "jwt-x"
        def create_api_key(self, name):
            raise DeployError("POST /api/auth/api-keys -> 404: route not found")

    key, note = deploy._mint_credential(_Old())
    assert key == "jwt-x" and note  # fell back to the JWT credential

    class _Other:
        token = "jwt-x"
        def create_api_key(self, name):
            raise DeployError("POST /api/auth/api-keys -> 500: upstream said 404 earlier")

    with pytest.raises(DeployError):
        deploy._mint_credential(_Other())  # 500 must propagate, not silently fall back


def test_get_ssh_falls_back_when_publicport_missing(monkeypatch):
    """A public-IP runtime port with no publicPort must not TypeError on int(None);
    fall back to the ssh.runpod.io proxy target."""
    from unsloth_cli.deploy.runpod_client import RunPod
    pod = {"runtime": {"ports": [
        {"privatePort": 22, "ip": "1.2.3.4", "isIpPublic": True},  # no publicPort
    ]}}
    _stub_runpod(monkeypatch, get_pod = lambda pid: pod)
    p = RunPod(); p.auth({"api_key": "rpa_x"})
    ssh = p.get_ssh("pod-x")
    assert ssh.host == "ssh.runpod.io" and ssh.user == "pod-x"


def test_is_local_model_distinguishes_hf_id_from_path(monkeypatch, tmp_path):
    """A bare HF id isn't hijacked by a coincidentally same-named local dir, but an
    explicit ./path or a recognized model dir is treated as local."""
    from unsloth_cli.commands.deploy import _is_local_model
    monkeypatch.chdir(tmp_path)

    (tmp_path / "org").mkdir()
    (tmp_path / "org" / "name").mkdir()  # exists, but not a model
    assert _is_local_model("org/name") is False        # stays an HF id
    assert _is_local_model("./org/name") is True        # explicit path is local

    (tmp_path / "mymodel").mkdir()
    (tmp_path / "mymodel" / "config.json").write_text("{}")
    assert _is_local_model("mymodel") is True           # recognized model dir
    assert _is_local_model("unsloth/Llama-3.2-1B") is False  # non-existent id


def test_rest_non_json_response_raises_deployerror(monkeypatch):
    """A non-JSON REST body (e.g. an HTML error page) surfaces as DeployError, not
    a raw JSONDecodeError the callers don't expect."""
    from unsloth_cli.deploy import DeployError
    from unsloth_cli.deploy import runpod_storage
    from unsloth_cli.deploy.runpod_client import RunPod

    class _Resp:
        def read(self): return b"<html>bad gateway</html>"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    _stub_runpod(monkeypatch)
    monkeypatch.setattr(
        runpod_storage.urllib.request, "urlopen", lambda req, timeout = 60: _Resp(),
    )
    p = RunPod(); p.auth({"api_key": "rpa_x"})
    with pytest.raises(DeployError, match = "non-JSON"):
        runpod_storage._rest(p, "GET", "/networkvolumes/x", None)


def test_empty_provider_opt_does_not_fall_through_to_env(monkeypatch):
    """An explicit empty --provider-opt value clears the option instead of silently
    falling back to the env var (--provider-opt has top precedence)."""
    import typer
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy.runpod_client import RunPod
    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_from_env")
    with pytest.raises(typer.Exit):
        deploy._authenticate(
            RunPod, {"api_key": ""}, need_local = False, interactive = False,
        )


def test_bootstrap_surfaces_password_when_change_password_drops_connection(monkeypatch):
    """If change_password's response is lost after the server applied it, the user
    must still see the (already rotated) password instead of being locked out."""
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy import DeployError

    class DropOnChange(_FakeStudioClient):
        def change_password(self, current, new):
            self.calls.append(("change_password", current, new))
            raise DeployError("POST /api/auth/change-password failed: connection reset")

    store = {}
    _stub_with_catalog(monkeypatch, create_pod = lambda **kw: {"id": "pod-drop"}, get_pod = _running_pod)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "bootstrap-pw")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(
        deploy, "StudioClient",
        lambda base_url: store.setdefault("client", DropOnChange(base_url)),
    )

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--model", "unsloth/Llama-3.2-1B-Instruct",
    ])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    rotated = [c for c in store["client"].calls if c[0] == "change_password"][0][2]
    assert rotated and rotated in combined
    assert "unsloth deploy stop pod-drop" in combined


def test_bootstrap_unexpected_error_still_prints_stop_hint(monkeypatch):
    """A non-DeployError during bootstrap (e.g. a KeyError) must still surface the
    stop hint so the billing instance isn't lost silently."""
    from unsloth_cli.commands import deploy

    class BoomOnKey(_FakeStudioClient):
        def create_api_key(self, name):
            raise KeyError("key")

    store = {}
    _stub_with_catalog(monkeypatch, create_pod = lambda **kw: {"id": "pod-boom"}, get_pod = _running_pod)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "bootstrap-pw")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(
        deploy, "StudioClient",
        lambda base_url: store.setdefault("client", BoomOnKey(base_url)),
    )

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--model", "unsloth/Llama-3.2-1B-Instruct",
    ])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "unsloth deploy stop pod-boom" in combined


def test_stage_warns_when_volume_delete_also_fails(monkeypatch, tmp_path):
    """If cleanup can't delete the volume after a failed upload, the user is told
    how to remove it, and the original upload error -- not the delete error --
    propagates."""
    from unsloth_cli.deploy import DeployError, runpod_storage
    from unsloth_cli.deploy.runpod_client import RunPod

    model = tmp_path / "m"; model.mkdir()
    (model / "adapter_config.json").write_text("{}")

    logs = []
    _stub_runpod(monkeypatch)
    monkeypatch.setattr(runpod_storage, "create_network_volume", lambda client, **kw: "vol-7")

    def del_fail(client, vid):
        raise DeployError("delete boom")

    def up_fail(local_path, **kw):
        raise DeployError("upload boom")

    monkeypatch.setattr(runpod_storage, "delete_network_volume", del_fail)
    monkeypatch.setattr(runpod_storage, "upload_path", up_fail)

    p = RunPod()
    p.auth({"api_key": "rpa_x", "s3_access_key": "u", "s3_secret_key": "s", "datacenter": "DC"})
    with pytest.raises(DeployError, match = "upload boom"):
        p.stage_local_model(model, gpu = _gpu(), log = logs.append)
    assert any("delete-storage vol-7" in m for m in logs)


# A concrete minimal Provider implementation for capability tests. Defined at
# module scope so it can be registered into PROVIDERS by name.
from unsloth_cli.deploy.base import Provider as _Provider  # noqa: E402


class _RealNoStorage(_Provider):
    name = "nostorage"

    @classmethod
    def option_schema(cls):
        return []

    def auth(self, options):
        pass

    def list_gpus(self, min_vram_gb = 0):
        return [_gpu()]

    def create_instance(self, **kwargs):
        return "i-1"

    def wait_ready(self, instance_id, timeout_s):
        pass

    def endpoint_url(self, instance_id, http_port):
        return "http://example"

    def terminate(self, instance_id):
        pass


# ---------------------------------------------------------------------------
# Modal provider
# ---------------------------------------------------------------------------


def _stub_modal(monkeypatch, *, poll_code = None, tunnels = None,
                lookup_error = None, upload_error = None):
    """Inject a fake `modal` module so Modal tests run offline, mirroring
    _stub_runpod. Returns (stub, state) where state records the SDK calls."""
    state = {
        "create_args": None, "create_kwargs": None, "secret_env": None,
        "image_ref": None, "app": None, "terminated": None,
        "volumes_created": [], "volumes_deleted": [], "uploaded": [],
    }
    tunnel_urls = tunnels if tunnels is not None else {8000: "https://sb-1.modal.host"}

    class _Tunnel:
        def __init__(self, url):
            self.url = url

    class _Sandbox:
        def __init__(self, object_id = "sb-1"):
            self.object_id = object_id

        def poll(self):
            return poll_code

        def tunnels(self, timeout = None):
            return {p: _Tunnel(u) for p, u in tunnel_urls.items()}

        def terminate(self, wait = False):
            state["terminated"] = self.object_id

    class _Batch:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def put_directory(self, local, remote):
            if upload_error:
                raise upload_error
            state["uploaded"].append(("dir", local, remote))

        def put_file(self, local, remote):
            if upload_error:
                raise upload_error
            state["uploaded"].append(("file", local, remote))

    class _Volume:
        def batch_upload(self):
            return _Batch()

    class _Image:
        def from_registry(self, ref, **kw):
            state["image_ref"] = ref
            return self

        def apt_install(self, *pkgs):
            return self

        def run_commands(self, *cmds, **kw):
            state["build_cmds"] = cmds
            return self

        def entrypoint(self, args):
            return self

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _SandboxNS:
        @staticmethod
        def create(*args, **kwargs):
            state["create_args"] = args
            state["create_kwargs"] = kwargs
            return _Sandbox()

        @staticmethod
        def from_id(object_id, client = None):
            return _Sandbox(object_id)

    class _VolumeNS:
        @staticmethod
        def from_name(name, create_if_missing = False, **kw):
            state["volumes_created"].append(name)
            return _Volume()

        @staticmethod
        def delete(name, **kw):
            state["volumes_deleted"].append(name)

    class _SecretNS:
        @staticmethod
        def from_dict(env):
            state["secret_env"] = env
            return ("secret", env)

    class _AppNS:
        @staticmethod
        def lookup(name, create_if_missing = False, **kw):
            if lookup_error:
                raise lookup_error
            state["app"] = name
            return ("app", name)

    stub = types.ModuleType("modal")
    stub.Sandbox = _SandboxNS
    stub.Volume = _VolumeNS
    stub.Secret = _SecretNS
    stub.App = _AppNS
    stub.Image = _Image()
    stub.enable_output = lambda: _NullCtx()
    monkeypatch.setitem(sys.modules, "modal", stub)
    return stub, state


def _authed_modal(monkeypatch, **stub_kw):
    stub, state = _stub_modal(monkeypatch, **stub_kw)
    from unsloth_cli.deploy.modal_client import Modal
    p = Modal()
    p.auth({})
    return p, state


def test_modal_registered():
    from unsloth_cli.deploy.base import Provider
    from unsloth_cli.deploy.provider import PROVIDERS, get_provider
    assert "modal" in PROVIDERS
    assert isinstance(get_provider("modal"), Provider)


def test_modal_declares_its_capabilities():
    from unsloth_cli.deploy.modal_client import Modal
    assert Modal.supports_local_model is True
    assert Modal.supports_ssh is False
    assert Modal.supports_pause is False
    assert Modal.reports_stock is False     # fixed-capacity: no stock band
    assert Modal.deploy_note                # non-empty 24h auto-stop note


def test_modal_list_gpus_filters_and_sorts_with_no_stock():
    from unsloth_cli.deploy.modal_client import Modal
    gpus = Modal().list_gpus(min_vram_gb = 40)
    assert gpus and all(g.vram_gb >= 40 for g in gpus)
    assert all(g.stock is None for g in gpus)
    costs = [g.cost_per_hour_usd for g in gpus]
    assert costs == sorted(costs)


def test_modal_auth_errors_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)  # `import modal` -> ImportError
    from unsloth_cli.deploy import DeployError
    from unsloth_cli.deploy.modal_client import Modal
    with pytest.raises(DeployError, match = r"unsloth\[deploy\]"):
        Modal().auth({})


def test_modal_auth_errors_on_bad_credentials(monkeypatch):
    _stub_modal(monkeypatch, lookup_error = RuntimeError("unauthenticated"))
    from unsloth_cli.deploy import DeployError
    from unsloth_cli.deploy.modal_client import Modal
    with pytest.raises(DeployError, match = "authentication failed"):
        Modal().auth({})


def test_modal_create_instance_passes_studio_command_ports_and_secret(monkeypatch):
    p, state = _authed_modal(monkeypatch)
    sid = p.create_instance(
        name = "unsloth-studio-1",
        gpu = _gpu(gid = "H100", vram = 80, price = 3.95),
        image = "ignored-by-modal",     # Modal builds from source, not a registry tag
        http_port = 8000,
        env = {"UNSLOTH_ADMIN_PASSWORD": "secret-pw"},
        disk_gb = 100,
    )
    assert sid == "sb-1"
    args = state["create_args"]
    assert args[0] == "/bin/sh" and args[1] == "-lc"
    # --api-only: the image whites out the built frontend; the deploy only needs the API.
    assert "unsloth studio --api-only -p 8000 -H 0.0.0.0" in args[2]
    # Seed the admin password into Studio's .bootstrap_password before launch
    # (Studio creates the admin user from it; it ignores UNSLOTH_ADMIN_PASSWORD).
    assert "UNSLOTH_STUDIO_HOME=" in args[2]
    assert ".bootstrap_password" in args[2]
    assert '"$UNSLOTH_ADMIN_PASSWORD"' in args[2]   # injected via secret, not inlined
    kwargs = state["create_kwargs"]
    assert kwargs["gpu"] == "H100"
    assert kwargs["encrypted_ports"] == [8000]      # TLS, never plaintext
    assert kwargs["timeout"] == 86400               # 24h hard cap
    assert kwargs["name"] == "unsloth-studio-1"
    assert state["secret_env"] == {"UNSLOTH_ADMIN_PASSWORD": "secret-pw"}


def test_modal_uses_prebaked_image(monkeypatch):
    from unsloth_cli.deploy.modal_client import STUDIO_IMAGE
    p, state = _authed_modal(monkeypatch)
    p.create_instance(
        name = "n", gpu = _gpu(), image = "x", http_port = 8000,
        env = {}, disk_gb = 100,
    )
    assert state["image_ref"] == STUDIO_IMAGE       # pulls the published image
    assert state.get("build_cmds") is None          # no from-source build (run_commands unused)


def test_modal_create_instance_mounts_staged_volume(monkeypatch):
    from unsloth_cli.deploy import StagedModel
    from unsloth_cli.deploy.modal_client import MODEL_MOUNT_DIR
    p, state = _authed_modal(monkeypatch)
    staged = StagedModel(
        model_path = f"{MODEL_MOUNT_DIR}/m", storage_id = "vol-1",
        summary = "", placement = None,
    )
    p.create_instance(
        name = "n", gpu = _gpu(), image = "img", http_port = 8000,
        env = {}, disk_gb = 100, staged = staged,
    )
    assert "vol-1" in state["volumes_created"]      # re-acquired by name
    assert MODEL_MOUNT_DIR in state["create_kwargs"]["volumes"]


def test_modal_wait_ready_raises_on_early_exit(monkeypatch):
    from unsloth_cli.deploy import DeployError
    p, _ = _authed_modal(monkeypatch, poll_code = 1)
    with pytest.raises(DeployError, match = "exited before serving"):
        p.wait_ready("sb-1", timeout_s = 5)


def test_modal_wait_ready_returns_when_tunnel_resolves(monkeypatch):
    p, _ = _authed_modal(monkeypatch, poll_code = None,
                         tunnels = {8000: "https://x.modal.host"})
    p.wait_ready("sb-1", timeout_s = 5)             # returns immediately


def test_modal_endpoint_url_uses_tunnel(monkeypatch):
    p, _ = _authed_modal(monkeypatch, tunnels = {8000: "https://abc.modal.host"})
    assert p.endpoint_url("sb-1", http_port = 8000) == "https://abc.modal.host"


def test_modal_terminate(monkeypatch):
    p, state = _authed_modal(monkeypatch)
    p.terminate("sb-1")
    assert state["terminated"] == "sb-1"


def test_modal_stage_local_model_uploads_to_volume(monkeypatch, tmp_path):
    from unsloth_cli.deploy.modal_client import MODEL_MOUNT_DIR
    p, state = _authed_modal(monkeypatch)
    model = tmp_path / "lora_model"
    model.mkdir()
    (model / "adapter_config.json").write_text("{}")

    staged = p.stage_local_model(model, gpu = _gpu())
    assert staged.model_path == f"{MODEL_MOUNT_DIR}/lora_model"
    assert staged.storage_id in state["volumes_created"]
    assert staged.placement is None                 # Modal needs no datacenter pin
    assert ("dir", str(model), "/lora_model") in state["uploaded"]


def test_modal_stage_deletes_volume_when_upload_fails(monkeypatch, tmp_path):
    from unsloth_cli.deploy import DeployError
    p, state = _authed_modal(monkeypatch, upload_error = RuntimeError("boom"))
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")

    with pytest.raises(DeployError, match = "upload failed"):
        p.stage_local_model(model, gpu = _gpu())
    assert state["volumes_deleted"]                 # billing volume cleaned up


def test_modal_yes_picks_cheapest_without_stock_warning(monkeypatch, capsys):
    from unsloth_cli.commands import deploy
    from unsloth_cli.deploy.modal_client import Modal
    chosen = deploy._pick_gpu(Modal(), override = None, min_vram_gb = 24, yes = True)
    cheapest = min(Modal().list_gpus(min_vram_gb = 24), key = lambda g: g.cost_per_hour_usd)
    assert chosen.id == cheapest.id
    # A fixed-capacity provider must not emit the "couldn't confirm stock" warning.
    assert "couldn't confirm" not in capsys.readouterr().err


def test_modal_stop_rejects_pause(monkeypatch):
    _stub_modal(monkeypatch)
    from unsloth_cli import app
    result = CliRunner().invoke(
        app, ["deploy", "stop", "sb-1", "--provider", "modal", "--pause"],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "can't pause" in combined
