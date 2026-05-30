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
    monkeypatch.delenv("RUNPOD_API_KEY", raising = False)
    with pytest.raises(DeployError, match = "RUNPOD_API_KEY"):
        RunPod().auth()


def test_list_gpus_drops_spot_and_priceless_and_sorts_by_cost(monkeypatch):
    from unsloth_cli.deploy.runpod_client import RunPod
    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    p = RunPod(); p.auth()

    gpus = p.list_gpus(min_vram_gb = 24)
    ids = [g.id for g in gpus]
    assert "spotonly" not in ids and "noprice" not in ids  # no on-demand price
    assert "A4000" not in ids                               # below min VRAM
    assert ids[0] == "A5000"                                # cheapest first
    assert [g.cost_per_hour_usd for g in gpus] == sorted(g.cost_per_hour_usd for g in gpus)


def test_create_pod_mounts_volume_at_workspace_and_opens_ports(monkeypatch):
    """unsloth-base writes under /workspace, so the volume must mount there;
    both the Studio HTTP port and SSH must be exposed."""
    from unsloth_cli.deploy.runpod_client import RunPod
    captured = {}
    _stub_runpod(monkeypatch, create_pod = lambda **kw: captured.update(kw) or {"id": "pod-abc"})
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")

    p = RunPod(); p.auth()
    pod_id = p.create_pod(
        name = "test", gpu_id = "A5000", image = "img:tag",
        ports = ["8000/http"], ssh_port = 22, disk_gb = 100,
        env = {"UNSLOTH_ADMIN_PASSWORD": "secret"},
    )
    assert pod_id == "pod-abc"
    assert captured["volume_mount_path"] == "/workspace"
    assert "8000/http" in captured["ports"]
    assert "22/tcp" in captured["ports"]


def test_wait_running_waits_for_runtime(monkeypatch):
    """desiredStatus flips to RUNNING before the container is live; we must
    wait until `runtime` is populated."""
    from unsloth_cli.deploy.runpod_client import RunPod
    states = iter([
        {"desiredStatus": "RUNNING", "runtime": None},
        {"desiredStatus": "RUNNING", "runtime": {"ports": []}},
    ])
    _stub_runpod(monkeypatch, get_pod = lambda pid: next(states))
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = RunPod(); p.auth()
    p.wait_running("pod-x", timeout_s = 30)  # returns without raising


def test_endpoint_url_prefers_direct_ip_else_proxy(monkeypatch):
    from unsloth_cli.deploy.runpod_client import RunPod
    # No SDK / no public port -> proxy hostname.
    assert RunPod().endpoint_url("podxyz", http_port = 8000) == "https://podxyz-8000.proxy.runpod.net"

    fake_pod = {"runtime": {"ports": [
        {"privatePort": 8000, "publicPort": 18000, "ip": "1.2.3.4", "isIpPublic": True},
    ]}}
    _stub_runpod(monkeypatch, get_pod = lambda pid: fake_pod)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    p = RunPod(); p.auth()
    assert p.endpoint_url("podxyz", http_port = 8000) == "http://1.2.3.4:18000"


def test_stop_terminates_by_default_keep_volume_pauses(monkeypatch):
    """`stop` fully terminates so users aren't surprised by storage billing;
    `--keep-volume` only pauses."""
    calls = []
    _stub_runpod(
        monkeypatch,
        terminate_pod = lambda pid: calls.append(("terminate", pid)),
        stop_pod = lambda pid: calls.append(("stop", pid)),
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    from unsloth_cli import app

    assert CliRunner().invoke(app, ["deploy", "stop", "pod-a"]).exit_code == 0
    assert CliRunner().invoke(app, ["deploy", "stop", "--keep-volume", "pod-b"]).exit_code == 0
    assert calls == [("terminate", "pod-a"), ("stop", "pod-b")]


# ---------------------------------------------------------------------------
# `deploy run` + pickers
# ---------------------------------------------------------------------------


def test_run_rejects_short_admin_password(monkeypatch):
    """Studio's change-password requires >= 8 chars; a shorter bootstrap
    password would brick the pod, so reject it up front."""
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

    def fake_deploy(runpod, gpu, **kw):
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

    (tmp_path / "lora_model").mkdir()
    (tmp_path / "lora_model" / "adapter_config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    captured = {}
    _stub_with_catalog(monkeypatch)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_x")
    monkeypatch.setenv("UNSLOTH_ADMIN_PASSWORD", "secret-pw")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY_ID", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_ACCESS_KEY", "rps_y")
    monkeypatch.setattr(deploy, "_deploy", lambda runpod, gpu, **kw: captured.update(kw))

    from unsloth_cli import app
    # pick model 1 (the local LoRA), GPU 1, then confirm
    result = CliRunner().invoke(app, ["deploy", "run"], input = "1\n1\ny\n")
    assert result.exit_code == 0, result.output
    assert captured["model"].endswith("/lora_model")
    assert "LoRA adapter" in result.output


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
        deploy.runpod_storage, "create_network_volume",
        lambda client, **kw: vol_kwargs.update(kw) or "vol-123",
    )
    monkeypatch.setattr(
        deploy.runpod_storage, "upload_path",
        lambda local_path, **kw: uploads.append((local_path, kw)),
    )
    monkeypatch.setattr(deploy, "StudioClient", _fake_studio(store))

    from unsloth_cli import app
    result = CliRunner().invoke(app, [
        "deploy", "run", "--yes", "--model", str(local), "--datacenter", "US-KS-2",
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

    # The user is told how to delete the volume so storage stops billing.
    assert "delete-volume vol-123" in result.output


def test_run_requires_s3_creds_for_local_model(monkeypatch, tmp_path):
    """Deploying a local model needs RunPod S3 credentials; without them we fail
    up front -- before creating any billing pod."""
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
    assert "S3 credentials" in combined
    assert not reached  # bailed before creating a pod


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
    from unsloth_cli.commands.deploy import _volume_size_gb, VOLUME_MIN_GB

    small = tmp_path / "m"
    small.mkdir()
    (small / "f").write_bytes(b"0" * 1000)
    assert _volume_size_gb(small) == VOLUME_MIN_GB


def test_bootstrap_failure_shows_stop_hint(monkeypatch):
    """Any bootstrap failure must tell the user how to stop the billing pod."""
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
    pod (the bootstrap password they passed in no longer works)."""
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
# Provider selection
# ---------------------------------------------------------------------------


def test_get_provider_rejects_unknown_and_lists_available():
    """An unknown --provider fails with the set of registered names."""
    from unsloth_cli.deploy import DeployError
    from unsloth_cli.deploy.provider import PROVIDERS, get_provider

    assert "runpod" in PROVIDERS
    with pytest.raises(DeployError, match = "Unknown provider"):
        get_provider("nope")
