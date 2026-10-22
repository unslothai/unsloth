# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib
import base64
import json
import threading
import time
import uuid
from pathlib import Path

import pytest


_CERTIFICATE = (
    "-----BEGIN ARGO TUNNEL TOKEN-----\n"
    + base64.b64encode(
        json.dumps(
            {"accountID": "account", "zoneID": "zone", "apiToken": "private-account-certificate"}
        ).encode()
    ).decode()
    + "\n-----END ARGO TUNNEL TOKEN-----\n"
)


@pytest.fixture
def named(tmp_path, monkeypatch):
    from auth import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
    return importlib.import_module("utils.cloudflare_named_tunnel")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Studio.Example.COM.", "studio.example.com"),
        ("studio.bücher.de", "studio.xn--bcher-kva.de"),
    ],
)
def test_hostname_is_canonicalized(named, raw, expected):
    assert named.normalize_hostname(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "https://studio.example.com",
        "example.com:443",
        "example.com/path",
        "user@example.com",
        "127.0.0.1",
        "[::1]",
        "localhost",
        "*.example.com",
        "-x.example.com",
        "a..example.com",
        "a\n.example.com",
    ],
)
def test_only_public_dns_hostnames_are_accepted(named, raw):
    with pytest.raises(ValueError):
        named.normalize_hostname(raw)


def test_mode_defaults_to_quick_and_does_not_hide_invalid_saved_data(named, monkeypatch):
    from storage import studio_db

    stored = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, default = None: stored.get(key, default)
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", stored.update)
    assert named.get_mode() == "quick"
    named.set_mode("named")
    assert named.get_mode() == "named"
    stored[named.MODE_KEY] = "unknown"
    with pytest.raises(named.NamedTunnelError, match = "configuration_invalid"):
        named.get_mode()


def test_bundle_is_encrypted_and_safe_status_never_returns_the_token(named):
    from storage import credential_secrets

    saved = named.save_token("Studio.Example.com", "private-tunnel-token")
    assert saved["hostname"] == "studio.example.com"
    assert named.load_bundle()["token"] == "private-tunnel-token"
    assert "private-tunnel-token" not in json.dumps(named.named_status())
    database = Path(credential_secrets.studio_db_path())
    assert b"private-tunnel-token" not in database.read_bytes()
    assert (
        named.save_token("studio.example.com", None)["configuration_id"]
        == saved["configuration_id"]
    )
    replaced = named.save_token("studio.example.com", "rotated-token")
    assert replaced["configuration_id"] != saved["configuration_id"]
    with pytest.raises(named.NamedTunnelError, match = "remove_configuration_first"):
        named.save_token("other.example.com", "different-token")


def test_unreadable_bundle_fails_closed(named, monkeypatch):
    from storage import credential_secrets
    monkeypatch.setattr(credential_secrets, "get_secret_with_presence", lambda *_: (None, True))
    with pytest.raises(named.NamedTunnelError, match = "credentials_unreadable"):
        named.load_bundle()


@pytest.fixture
def provisioner(named, monkeypatch):
    import cloudflare_tunnel

    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    named.set_mode("named")
    monkeypatch.setattr(named, "_operation", None)
    monkeypatch.setattr(cloudflare_tunnel, "ensure_cloudflared", lambda: "cloudflared")
    commands = []
    tunnel_id = str(uuid.uuid4())

    def command(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        commands.append(list(args))
        if "login" in args:
            cert = home / ".cloudflared" / "cert.pem"
            cert.parent.mkdir(exist_ok = True)
            cert.write_text(_CERTIFICATE)
        if "create" in args:
            credentials = {"TunnelID": tunnel_id, "AccountTag": "account", "TunnelSecret": "secret"}
            Path(args[args.index("--credentials-file") + 1]).write_text(json.dumps(credentials))
        if "list" in args:
            if "--id" in args:
                return 0, json.dumps([{"id": tunnel_id}])
            return 0, "[]"
        if "dns" in args:
            return 0, f"Added CNAME {operation.hostname} which will route to this tunnel"
        return 0, ""

    monkeypatch.setattr(named, "_run_command", command)
    monkeypatch.setattr(named, "_zone_name", lambda operation, bundle: "example.com", raising = False)
    monkeypatch.setattr(named, "_tunnel_status", lambda operation, bundle: "down", raising = False)
    yield commands, command
    operation = named._operation
    if operation and operation.thread.is_alive():
        named.cancel_setup(operation.operation_id)
        operation.thread.join(3)


def wait_for_state(named, state):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = named.named_status()["setup"]
        if status and status["state"] == state:
            return status
        time.sleep(0.01)
    pytest.fail(f"Expected {state}, received {named.named_status()}")


def test_automatic_setup_persists_identity_and_reuses_it(named, provisioner):
    commands, _ = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    assert bundle["certificate"] == _CERTIFICATE
    assert bundle["credentials"]["TunnelID"] == bundle["tunnel_id"]
    assert "private-account-certificate" not in json.dumps(named.named_status())
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    assert sum("create" in cmd for cmd in commands) == 1
    assert not any("--overwrite-dns" in cmd for cmd in commands)


@pytest.mark.parametrize(
    "reason",
    [
        "An A, AAAA or CNAME record already exists with that host",
        "Failed to create record studio.example.com with err An A, AAAA, or CNAME record with that host already exists. For more details, refer to <https://developers.cloudflare.com/dns/manage-dns-records/troubleshooting/records-with-same-name/>.",
    ],
)
def test_only_a_dns_conflict_allows_bound_single_use_confirmation(
    named, provisioner, monkeypatch, reason
):
    commands, command = provisioner

    def conflict(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "dns" in args and "--overwrite-dns" not in args:
            commands.append(list(args))
            return (
                1,
                "Failed to add route: code: 1003, reason: " + reason,
            )
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", conflict)
    named.start_setup("studio.example.com")
    status = wait_for_state(named, "dns_conflict")
    with pytest.raises(named.NamedTunnelError, match = "stale_dns_confirmation"):
        named.confirm_dns(status["operation_id"], "other.example.com", status["tunnel_id"])
    assert not any("--overwrite-dns" in cmd for cmd in commands)
    named.confirm_dns(status["operation_id"], status["hostname"], status["tunnel_id"])
    with pytest.raises(named.NamedTunnelError, match = "stale_dns_confirmation"):
        named.confirm_dns(status["operation_id"], status["hostname"], status["tunnel_id"])
    wait_for_state(named, "ready")
    assert sum("--overwrite-dns" in cmd for cmd in commands) == 1


@pytest.mark.parametrize(
    "output",
    [
        "authentication failed",
        "zone not found",
        "network timeout",
        "code: 1003, reason: Invalid DNS record",
        "An A, AAAA, or CNAME record with that host already exists.",
    ],
)
def test_other_dns_errors_never_offer_overwrite(named, provisioner, monkeypatch, output):
    _, command = provisioner

    def failed(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "dns" in args:
            return 1, output
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", failed)
    named.start_setup("studio.example.com")
    status = wait_for_state(named, "error")
    assert status["error"] == "dns_route_failed"
    assert named.load_bundle()["tunnel_id"]


def test_wrong_zone_result_requires_cleanup_and_is_not_ready(named, provisioner, monkeypatch):
    _, command = provisioner

    def wrong_zone(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "dns" in args:
            return 0, "Added CNAME studio.example.com.other.com which will route to this tunnel"
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", wrong_zone)
    named.start_setup("studio.example.com")
    status = wait_for_state(named, "error")
    assert status["error"] == "dns_hostname_mismatch"
    assert named.load_bundle()["dns_cleanup_hostname"] == "studio.example.com.other.com"
    assert not named.named_status()["configuration"]["ready"]


def test_cancel_keeps_partial_identity_and_rejects_old_confirmation(
    named, provisioner, monkeypatch
):
    _, command = provisioner
    entered = threading.Event()

    def waiting(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "dns" in args:
            entered.set()
            operation.cancel.wait(3)
            raise named.NamedTunnelError("setup_cancelled")
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", waiting)
    response = named.start_setup("studio.example.com")
    assert entered.wait(3)
    named.cancel_setup(response["operation_id"])
    wait_for_state(named, "error")
    assert named.load_bundle()["tunnel_id"]
    with pytest.raises(named.NamedTunnelError, match = "stale_dns_confirmation"):
        named.confirm_dns(
            response["operation_id"], "studio.example.com", named.load_bundle()["tunnel_id"]
        )


def test_login_link_and_environment_are_restricted(named, monkeypatch, tmp_path):
    monkeypatch.setenv("TUNNEL_TOKEN", "inherited-token")
    monkeypatch.setenv("TUNNEL_FORCE_PROVISIONING_DNS", "true")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "browser-profile"))
    environment = named.child_environment(tmp_path)
    assert "TUNNEL_TOKEN" not in environment
    assert "TUNNEL_FORCE_PROVISIONING_DNS" not in environment
    assert environment["HOME"] == str(tmp_path)
    assert environment["USERPROFILE"] == str(tmp_path / "browser-profile")
    assert named.login_url(
        "https://dash.cloudflare.com/argotunnel?callback=https%3A%2F%2Flogin.cloudflareaccess.org%2Fx"
    )
    for url in [
        "http://dash.cloudflare.com/argotunnel",
        "https://dash.cloudflare.com.evil/argotunnel",
        "https://evil@dash.cloudflare.com/argotunnel",
        "https://dash.cloudflare.com/evil",
    ]:
        assert named.login_url(url) is None


@pytest.mark.parametrize(
    "hostname,origin",
    [("studio.example.com", "http://127.0.0.1:9123"), ("studio.example.com", "http://[::1]:9124")],
)
def test_runtime_uses_bound_origin_and_keeps_lease_until_cleanup(
    named, provisioner, hostname, origin
):
    named.start_setup(hostname)
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    cmd, env, cleanup = named.prepare_connector("cloudflared", bundle, origin, "http2")
    config_path = Path(cmd[cmd.index("--config") + 1])
    config = json.loads(config_path.read_text())
    assert config["ingress"] == [
        {"hostname": hostname, "service": origin},
        {"service": "http_status:404"},
    ]
    assert json.loads(env["TUNNEL_CRED_CONTENTS"])["TunnelID"] == bundle["tunnel_id"]
    assert "--protocol" in cmd and "http2" in cmd
    assert bundle["credentials"]["TunnelSecret"] not in " ".join(cmd)
    with pytest.raises(named.NamedTunnelError, match = "named_tunnel_busy"):
        named._acquire_lock()
    cleanup()
    assert not config_path.exists()
    lock = named._acquire_lock()
    lock.release()


def test_token_runtime_uses_environment_and_rejects_replaced_snapshot(named):
    named.set_mode("named")
    named.save_token("studio.example.com", "first-token")
    bundle = named.load_bundle()
    named.save_token("studio.example.com", "second-token")
    with pytest.raises(named.NamedTunnelError, match = "configuration_changed"):
        named.prepare_connector("cloudflared", bundle, "http://localhost:8888", None)
    cmd, env, cleanup = named.prepare_connector(
        "cloudflared", named.load_bundle(), "http://localhost:8888", None
    )
    try:
        assert env["TUNNEL_TOKEN"] == "second-token"
        assert "second-token" not in " ".join(cmd)
        assert "TUNNEL_CRED_CONTENTS" not in env
    finally:
        cleanup()


@pytest.mark.parametrize("remote_status", ["healthy", "degraded"])
def test_active_cloudflare_status_blocks_removal_without_deleting(
    named, provisioner, monkeypatch, remote_status
):
    commands, _ = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    monkeypatch.setattr(named, "_tunnel_status", lambda operation, bundle: remote_status)
    named.remove_named(bundle["configuration_id"])
    named._operation.thread.join(3)
    assert named.named_status()["setup"]["error"] == "tunnel_still_active"
    assert named.load_bundle() == bundle
    assert not any("delete" in command for command in commands)


@pytest.mark.parametrize("remote_status", ["down", "inactive"])
def test_removal_checks_cloudflare_status_before_nonforced_delete(
    named, provisioner, monkeypatch, remote_status
):
    commands, _ = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    configuration = named.named_status()["configuration"]

    def status(operation, bundle):
        commands.append(["check-status", bundle["tunnel_id"]])
        return remote_status

    monkeypatch.setattr(named, "_tunnel_status", status)
    named.remove_named(configuration["configuration_id"])
    named._operation.thread.join(3)
    setup = named.named_status()["setup"]
    assert setup["state"] == "removed"
    assert commands[-2:] == [
        ["check-status", configuration["tunnel_id"]],
        ["delete", configuration["tunnel_id"]],
    ]
    assert [cmd for cmd in commands if "delete" in cmd] == [["delete", configuration["tunnel_id"]]]
    assert setup["dns_cleanup_hostname"] == "studio.example.com"
    assert named.load_bundle() is None


@pytest.mark.parametrize("error", ["tunnel_status_unavailable", "cloudflare_login_required"])
def test_failed_status_check_retains_configuration_without_delete(
    named, provisioner, monkeypatch, error
):
    commands, _ = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()

    def unavailable(operation, bundle):
        raise named.NamedTunnelError(error)

    monkeypatch.setattr(named, "_tunnel_status", unavailable)
    named.remove_named(bundle["configuration_id"])
    named._operation.thread.join(3)
    assert named.named_status()["setup"]["error"] == error
    assert named.load_bundle() == bundle
    assert not any("delete" in command for command in commands)


@pytest.mark.parametrize("listed,code", [([], 0), ([{"id": "still-exists"}], 0), ([], 1)])
def test_missing_status_requires_successful_uuid_lookup_before_local_removal(
    named, provisioner, monkeypatch, listed, code
):
    commands, command = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    monkeypatch.setattr(named, "_tunnel_status", lambda operation, bundle: "not_found")

    def lookup(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "list" in args and "--id" in args:
            commands.append(list(args))
            return code, json.dumps(listed)
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", lookup)
    named.remove_named(bundle["configuration_id"])
    named._operation.thread.join(3)
    assert ["list", "--id", bundle["tunnel_id"], "--output", "json"] in commands
    assert not any("delete" in command for command in commands)
    if not code and not listed:
        assert named.named_status()["setup"]["state"] == "removed"
        assert named.load_bundle() is None
    else:
        assert named.named_status()["setup"]["error"] == "tunnel_status_unavailable"
        assert named.load_bundle() == bundle


def test_deleted_cloudflare_status_only_removes_local_configuration(
    named, provisioner, monkeypatch
):
    commands, _ = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    monkeypatch.setattr(named, "_tunnel_status", lambda operation, bundle: "deleted")
    named.remove_named(bundle["configuration_id"])
    named._operation.thread.join(3)
    assert named.named_status()["setup"]["state"] == "removed"
    assert named.named_status()["setup"]["dns_cleanup_hostname"] == "studio.example.com"
    assert not any("delete" in command for command in commands)


@pytest.mark.parametrize(
    "output",
    [
        "Cannot delete tunnel with active connections",
        json.dumps(
            {
                "level": "error",
                "error": "Error deleting tunnel: code: 1022, reason: Cannot delete tunnel because it has active connections. Close any cloudflareds running the tunnel.",
            }
        ),
    ],
)
def test_delete_failure_retains_credentials_and_can_be_retried(
    named, provisioner, monkeypatch, output
):
    _, command = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    configuration = named.named_status()["configuration"]

    def in_use(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "delete" in args:
            return 1, output
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", in_use)
    named.remove_named(configuration["configuration_id"])
    assert wait_for_state(named, "error")["error"] == "tunnel_still_active"
    named._operation.thread.join(3)
    assert named.load_bundle()["credentials"]
    monkeypatch.setattr(named, "_run_command", command)
    named.remove_named(configuration["configuration_id"])
    wait_for_state(named, "removed")


@pytest.mark.parametrize(
    "output", ["network connection failed", "permission denied", "invalid tunnel"]
)
def test_other_delete_errors_are_not_reported_as_active_connections(
    named, provisioner, monkeypatch, output
):
    _, command = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()

    def failed(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        return (1, output) if "delete" in args else command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", failed)
    named.remove_named(bundle["configuration_id"])
    named._operation.thread.join(3)
    assert named.named_status()["setup"]["error"] == "tunnel_delete_failed"
    assert named.load_bundle() == bundle


def test_unreadable_absence_confirmation_retains_configuration(named, provisioner, monkeypatch):
    commands, command = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    monkeypatch.setattr(named, "_tunnel_status", lambda operation, bundle: "not_found")

    def invalid_lookup(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "list" in args and "--id" in args:
            return 0, "invalid response"
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", invalid_lookup)
    named.remove_named(bundle["configuration_id"])
    named._operation.thread.join(3)
    assert named.named_status()["setup"]["error"] == "tunnel_status_unavailable"
    assert named.load_bundle() == bundle
    assert not any("delete" in command for command in commands)


def test_token_removal_is_local_and_requires_current_configuration(named):
    configuration = named.save_token("studio.example.com", "token")
    with pytest.raises(named.NamedTunnelError, match = "configuration_changed"):
        named.remove_named(str(uuid.uuid4()))
    named.remove_named(configuration["configuration_id"])
    assert named.load_bundle() is None


def test_new_token_configuration_hides_previous_automatic_removal(named, provisioner):
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    named.remove_named(named.load_bundle()["configuration_id"])
    wait_for_state(named, "removed")
    named._operation.thread.join(3)
    assert named.named_status()["setup"]["dns_cleanup_hostname"] == "studio.example.com"
    named.save_token("studio.example.com", "token")
    assert named.named_status()["setup"] is None


def test_cloudflare_null_list_allows_first_tunnel_creation(named, provisioner, monkeypatch):
    commands, command = provisioner

    def empty_list(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "list" in args:
            return 0, 'null\n{"level":"warn","message":"A newer version is available"}\n'
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", empty_list)
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    assert sum("create" in cmd for cmd in commands) == 1


def test_resume_finds_existing_tunnel_with_cloudflare_zero_deleted_timestamp(
    named, provisioner, monkeypatch
):
    commands, command = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    bundle.pop("credentials")
    bundle.pop("tunnel_id")
    bundle["ready"] = False
    named._save_bundle(bundle)
    recovered_id = str(uuid.uuid4())

    def recover(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "list" in args:
            return 0, json.dumps(
                [
                    {
                        "name": bundle["tunnel_name"],
                        "id": recovered_id,
                        "deleted_at": "0001-01-01T00:00:00Z",
                    }
                ]
            )
        if "token" in args:
            Path(args[args.index("--cred-file") + 1]).write_text(
                json.dumps(
                    {"TunnelID": recovered_id, "AccountTag": "account", "TunnelSecret": "secret"}
                )
            )
            return 0, ""
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", recover)
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    assert named.load_bundle()["tunnel_id"] == recovered_id
    assert sum("create" in cmd for cmd in commands) == 1


def test_already_deleted_tunnel_is_removed_locally(named, provisioner, monkeypatch):
    _, command = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)

    def deleted(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "delete" in args:
            return 1, f"Tunnel {named.load_bundle()['tunnel_id']} has already been deleted"
        if "list" in args and "--id" in args:
            return 0, "[]"
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", deleted)
    named.remove_named(named.load_bundle()["configuration_id"])
    wait_for_state(named, "removed")
    assert named.load_bundle() is None


def test_management_command_isolated_home_output_and_login_link(named, monkeypatch, tmp_path):
    import cloudflare_tunnel
    import io
    import tempfile

    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    operation = named.NamedTunnelOperation(
        "studio.example.com",
        str(uuid.uuid4()),
        cloudflare_tunnel.capture_studio_tunnel_start_admission(),
        str(tmp_path),
    )
    monkeypatch.setattr(named, "_operation", operation)
    calls = []

    class Process:
        pid = 424243
        returncode = 0
        stdout = io.StringIO("https://dash.cloudflare.com/argotunnel?callback=test\n")

        def poll(self):
            return 0

    monkeypatch.setattr(cloudflare_tunnel, "_spawn_child", lambda spawn: spawn())
    monkeypatch.setattr(cloudflare_tunnel, "_adopt_pid", lambda pid: None)
    monkeypatch.setattr(cloudflare_tunnel, "_forget_pid", lambda pid: None)
    monkeypatch.setattr(
        named.subprocess, "Popen", lambda cmd, **kwargs: (calls.append((cmd, kwargs)), Process())[1]
    )
    with tempfile.TemporaryDirectory(dir = tmp_path) as directory:
        home = Path(directory)
        assert named._run_command(operation, "cloudflared", ["login"], home) == (
            0,
            "https://dash.cloudflare.com/argotunnel?callback=test\n",
        )
    cmd, kwargs = calls[0]
    assert "--logformat" not in cmd
    assert cmd[cmd.index("--output") + 1] == "json"
    assert kwargs["env"]["HOME"] == str(home)
    assert not home.exists()
    assert Path(kwargs["cwd"]).is_dir()
    assert operation.auth_url == "https://dash.cloudflare.com/argotunnel?callback=test"


def test_setup_cleans_private_home_while_login_browser_is_running(
    named, provisioner, monkeypatch, tmp_path
):
    import subprocess
    import sys
    from utils.paths import private_tmp

    if sys.platform != "win32":
        pytest.skip("Windows prevents removal of an open browser profile file")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "browser-profile"))
    monkeypatch.setattr(private_tmp, "private_tmp_dir", lambda name: tmp_path)
    _, command = provisioner
    browsers = []
    homes = []

    def login(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        result = command(operation, binary, args, home, timeout)
        if "login" in args:
            browser = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import os, pathlib, sys; "
                    "profile = pathlib.Path(os.environ['USERPROFILE']) / 'AppData' / 'Roaming' / 'Mozilla' / 'Firefox'; "
                    "assert profile.is_relative_to(pathlib.Path(sys.argv[1])); "
                    "profile.mkdir(parents=True, exist_ok=True); "
                    "handle = (profile / 'browser.lock').open('wb'); "
                    "print('ready', flush=True); sys.stdin.buffer.read(1)",
                    str(tmp_path),
                ],
                env = named.child_environment(home),
                cwd = home.parent,
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                creationflags = subprocess.CREATE_NO_WINDOW,
            )
            browsers.append(browser)
            homes.append(home)
            assert browser.stdout.readline().strip() == b"ready"
        return result

    monkeypatch.setattr(named, "_run_command", login)
    try:
        named.start_setup("studio.example.com")
        named._operation.thread.join(5)
        assert not named._operation.thread.is_alive()
        assert named.named_status()["setup"]["state"] == "ready"
        assert named.named_status()["configuration"]["ready"]
        assert not homes[0].exists()
        assert browsers[0].poll() is None
    finally:
        for browser in browsers:
            browser.communicate(b"\n", timeout = 5)


@pytest.mark.parametrize(
    "interruption,error",
    [
        ("command_timeout", "command_timed_out"),
        ("setup_timeout", "setup_timed_out"),
        ("cancel", "setup_cancelled"),
        ("shutdown", "setup_cancelled"),
    ],
)
def test_interrupted_management_command_stops_and_forgets_child(
    named, monkeypatch, tmp_path, interruption, error
):
    import cloudflare_tunnel
    import io

    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    operation = named.NamedTunnelOperation(
        "studio.example.com",
        str(uuid.uuid4()),
        cloudflare_tunnel.capture_studio_tunnel_start_admission(),
        str(tmp_path),
    )
    monkeypatch.setattr(named, "_operation", operation)
    actions = []

    class Process:
        pid = 424243
        returncode = None
        stdout = io.StringIO("")

        def poll(self):
            return self.returncode

        def terminate(self):
            actions.append("terminate")
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    def spawn(*args, **kwargs):
        if interruption == "cancel":
            operation.cancel.set()
        elif interruption == "shutdown":
            cloudflare_tunnel.close_studio_tunnel_lifecycle()
        elif interruption == "setup_timeout":
            operation.deadline = time.monotonic() - 1
        return Process()

    monkeypatch.setattr(cloudflare_tunnel, "_spawn_child", lambda spawn: spawn())
    monkeypatch.setattr(cloudflare_tunnel, "_adopt_pid", lambda pid: actions.append("adopt"))
    monkeypatch.setattr(cloudflare_tunnel, "_forget_pid", lambda pid: actions.append("forget"))
    monkeypatch.setattr(named.subprocess, "Popen", spawn)
    with pytest.raises(named.NamedTunnelError, match = error):
        named._run_command(
            operation,
            "cloudflared",
            ["login"],
            tmp_path,
            timeout = 0 if interruption == "command_timeout" else 60,
        )
    assert actions == ["adopt", "terminate", "forget"]


def test_unconfirmed_command_exit_keeps_lease_and_private_files(named, monkeypatch, tmp_path):
    import cloudflare_tunnel
    import io

    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    operation = named.NamedTunnelOperation(
        "studio.example.com",
        str(uuid.uuid4()),
        cloudflare_tunnel.capture_studio_tunnel_start_admission(),
        str(tmp_path),
    )
    monkeypatch.setattr(named, "_operation", operation)
    stuck = threading.Event()
    paths = []

    class Process:
        pid = 424243
        returncode = None
        stdout = io.StringIO("")

        def poll(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(cloudflare_tunnel, "ensure_cloudflared", lambda: "cloudflared")
    monkeypatch.setattr(cloudflare_tunnel, "_spawn_child", lambda spawn: spawn())
    monkeypatch.setattr(cloudflare_tunnel, "_adopt_pid", lambda pid: None)
    monkeypatch.setattr(cloudflare_tunnel, "_forget_pid", lambda pid: None)
    monkeypatch.setattr(named.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(named, "_stop_command", lambda proc: (stuck.set(), False)[1])

    def configure(operation, binary, home, bundle):
        paths.append(home / "config.json")
        named._run_command(operation, binary, ["login"], home, timeout = 0)

    monkeypatch.setattr(named, "_automatic_setup", configure)
    operation.thread = threading.Thread(
        target = named._setup_worker, args = (operation, named._acquire_lock(), {}, False)
    )
    operation.thread.start()
    try:
        assert stuck.wait(3)
        assert paths[0].is_file()
        with pytest.raises(named.NamedTunnelError, match = "named_tunnel_busy"):
            named._acquire_lock()
    finally:
        process.returncode = 0
        operation.thread.join(3)
    assert not operation.thread.is_alive()
    assert not paths[0].exists()
    lock = named._acquire_lock()
    lock.release()


def test_reauthorization_in_another_account_preserves_original_identity(
    named, provisioner, monkeypatch
):
    commands, command = provisioner
    named.start_setup("studio.example.com")
    wait_for_state(named, "ready")
    named._operation.thread.join(3)
    original = named.load_bundle()

    def other_account(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        result = command(operation, binary, args, home, timeout)
        if "login" in args:
            (home / ".cloudflared" / "cert.pem").write_text(
                _CERTIFICATE.replace(
                    base64.b64encode(
                        json.dumps(
                            {
                                "accountID": "account",
                                "zoneID": "zone",
                                "apiToken": "private-account-certificate",
                            }
                        ).encode()
                    ).decode(),
                    base64.b64encode(
                        json.dumps(
                            {
                                "accountID": "other-account",
                                "zoneID": "other-zone",
                                "apiToken": "private",
                            }
                        ).encode()
                    ).decode(),
                )
            )
        return result

    monkeypatch.setattr(named, "_run_command", other_account)
    named.start_setup("studio.example.com", reauthenticate = True)
    status = wait_for_state(named, "error")
    assert status["error"] == "cloudflare_login_required"
    assert named.load_bundle()["certificate"] == original["certificate"]
    assert named.load_bundle()["tunnel_id"] == original["tunnel_id"]
    assert sum("create" in cmd for cmd in commands) == 1


def test_wrong_certificate_zone_never_routes_or_offers_overwrite(named, provisioner, monkeypatch):
    commands, _ = provisioner
    monkeypatch.setattr(named, "_zone_name", lambda operation, bundle: "other.example.net")
    named.start_setup("studio.example.com")
    status = wait_for_state(named, "error")
    assert status["error"] == "dns_hostname_mismatch"
    assert not any("dns" in command for command in commands)


def test_runtime_snapshot_cannot_start_after_mode_changed(named):
    named.set_mode("named")
    named.save_token("studio.example.com", "token")
    bundle = named.load_bundle()
    named.set_mode("quick")
    with pytest.raises(named.NamedTunnelError, match = "configuration_changed"):
        named.prepare_connector("cloudflared", bundle, "http://localhost:8888", None)


def test_cancelled_route_keeps_cleanup_hostname(named, provisioner, monkeypatch):
    _, command = provisioner

    def cancel_after_route(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        result = command(operation, binary, args, home, timeout)
        if "dns" in args:
            operation.cancel.set()
            raise named.NamedTunnelError("setup_cancelled")
        return result

    monkeypatch.setattr(named, "_run_command", cancel_after_route)
    named.start_setup("studio.example.com")
    wait_for_state(named, "error")
    assert named.load_bundle()["dns_cleanup_hostname"] == "studio.example.com"


def test_zone_preflight_uses_only_cloudflare_certificate_scope(named, monkeypatch, tmp_path):
    import cloudflare_tunnel
    import urllib.request
    import io

    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    operation = named.NamedTunnelOperation(
        "studio.example.com",
        str(uuid.uuid4()),
        cloudflare_tunnel.capture_studio_tunnel_start_admission(),
        str(tmp_path),
    )
    monkeypatch.setattr(named, "_operation", operation)
    calls = []

    def response(request, timeout):
        calls.append(request)
        assert 0 < timeout <= 60
        return io.BytesIO(json.dumps({"success": True, "result": {"name": "Example.COM"}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", response)
    assert named._zone_name(operation, {"certificate": _CERTIFICATE}) == "example.com"
    assert calls[0].full_url == "https://api.cloudflare.com/client/v4/zones/zone"
    assert calls[0].headers["Authorization"] == "Bearer private-account-certificate"


@pytest.mark.parametrize(
    "output,error",
    [
        ("Failed to fetch page. Server returned: 401", "cloudflare_login_required"),
        ("Failed to fetch page. Server returned: 403", "cloudflare_login_required"),
        ("Failed to fetch page. Server returned: 429", "tunnel_lookup_failed"),
        ("Failed to fetch page. Server returned: 500", "tunnel_lookup_failed"),
        ("REST request failed: dial tcp: i/o timeout", "tunnel_lookup_failed"),
        (
            '{"level":"error","error":"Failed to fetch page. Server returned: 500"}',
            "tunnel_lookup_failed",
        ),
    ],
)
@pytest.mark.parametrize("remove", [False, True])
def test_failed_tunnel_lookup_distinguishes_authorization_from_availability(
    named, provisioner, monkeypatch, output, error, remove
):
    commands, command = provisioner

    def failed_lookup(
        operation,
        binary,
        args,
        home,
        timeout = 60,
    ):
        if "list" in args:
            commands.append(list(args))
            return 1, output
        return command(operation, binary, args, home, timeout)

    monkeypatch.setattr(named, "_run_command", failed_lookup)
    named.start_setup("studio.example.com")
    wait_for_state(named, "error")
    named._operation.thread.join(3)
    bundle = named.load_bundle()
    if remove:
        named.remove_named(bundle["configuration_id"])
        named._operation.thread.join(3)
    assert named.named_status()["setup"]["error"] == error
    assert named.load_bundle() == bundle
    assert not any("create" in args or "delete" in args for args in commands)


@pytest.fixture
def removal_probe(named, monkeypatch, tmp_path):
    import cloudflare_tunnel

    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    operation = named.NamedTunnelOperation(
        "studio.example.com",
        str(uuid.uuid4()),
        cloudflare_tunnel.capture_studio_tunnel_start_admission(),
        str(tmp_path),
    )
    monkeypatch.setattr(named, "_operation", operation)
    return operation, {"certificate": _CERTIFICATE, "tunnel_id": str(uuid.uuid4())}


@pytest.mark.parametrize(
    "http_status,error",
    [
        (401, "cloudflare_login_required"),
        (403, "cloudflare_login_required"),
        (429, "dns_route_failed"),
        (500, "dns_route_failed"),
    ],
)
def test_zone_preflight_distinguishes_authorization_from_availability(
    named, removal_probe, monkeypatch, http_status, error
):
    import urllib.error
    import urllib.request

    operation, bundle = removal_probe

    def unavailable(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.cloudflare.com/", http_status, "test", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(named.NamedTunnelError, match = "^" + error + "$"):
        named._zone_name(operation, bundle)


@pytest.mark.parametrize(
    "payload", [b"not-json", b"null", b'{"success":false}', b'{"success":true,"result":{}}']
)
def test_malformed_zone_response_does_not_request_reauthorization(
    named, removal_probe, monkeypatch, payload
):
    import io
    import urllib.request

    operation, bundle = removal_probe
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    with pytest.raises(named.NamedTunnelError, match = "^dns_route_failed$"):
        named._zone_name(operation, bundle)


@pytest.mark.parametrize("cancel", [False, True])
def test_zone_network_failure_preserves_cancellation_and_does_not_request_reauthorization(
    named, removal_probe, monkeypatch, cancel
):
    import urllib.error
    import urllib.request

    operation, bundle = removal_probe

    def unavailable(*args, **kwargs):
        if cancel:
            operation.cancel.set()
        raise urllib.error.URLError("network unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    error = "setup_cancelled" if cancel else "dns_route_failed"
    with pytest.raises(named.NamedTunnelError, match = "^" + error + "$"):
        named._zone_name(operation, bundle)


@pytest.mark.parametrize("status", ["down", "inactive", "healthy", "degraded"])
def test_cloudflare_removal_status_uses_saved_tunnel_and_private_certificate(
    named, removal_probe, monkeypatch, status
):
    import io
    import urllib.request

    operation, bundle = removal_probe
    calls = []

    def response(request, timeout):
        calls.append(request)
        assert request.get_method() == "GET"
        assert 0 < timeout <= 60
        return io.BytesIO(
            json.dumps(
                {
                    "success": True,
                    "result": {
                        "id": bundle["tunnel_id"],
                        "status": status,
                        "deleted_at": None,
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", response)
    assert named._tunnel_status(operation, bundle) == status
    assert (
        calls[0].full_url
        == "https://api.cloudflare.com/client/v4/accounts/account/cfd_tunnel/" + bundle["tunnel_id"]
    )
    assert calls[0].headers["Authorization"] == "Bearer private-account-certificate"


@pytest.mark.parametrize(
    "result",
    [None, [], {}, {"status": "down"}, {"status": "unknown"}, {"id": "other", "status": "down"}],
)
def test_unreliable_cloudflare_removal_status_is_not_assumed_down(
    named, removal_probe, monkeypatch, result
):
    import io
    import urllib.request

    operation, bundle = removal_probe
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            json.dumps({"success": True, "result": result}).encode()
        ),
    )
    with pytest.raises(named.NamedTunnelError, match = "^tunnel_status_unavailable$"):
        named._tunnel_status(operation, bundle)


@pytest.mark.parametrize(
    "http_status,error",
    [
        (401, "cloudflare_login_required"),
        (403, "cloudflare_login_required"),
        (429, "tunnel_status_unavailable"),
        (500, "tunnel_status_unavailable"),
    ],
)
def test_cloudflare_removal_status_distinguishes_authorization_from_availability(
    named, removal_probe, monkeypatch, http_status, error
):
    import urllib.error
    import urllib.request

    operation, bundle = removal_probe

    def unavailable(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.cloudflare.com/", http_status, "test", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(named.NamedTunnelError, match = "^" + error + "$"):
        named._tunnel_status(operation, bundle)


def test_cloudflare_removal_status_404_is_only_an_unconfirmed_absence(
    named, removal_probe, monkeypatch
):
    import urllib.error
    import urllib.request

    operation, bundle = removal_probe

    def absent(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.cloudflare.com/", 404, "test", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", absent)
    assert named._tunnel_status(operation, bundle) == "not_found"


@pytest.mark.parametrize("status", ["unknown", None, [], {}])
def test_unrecognized_cloudflare_state_preserves_status_check_failure(
    named, removal_probe, monkeypatch, status
):
    import io
    import urllib.request

    operation, bundle = removal_probe
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            json.dumps(
                {
                    "success": True,
                    "result": {"id": bundle["tunnel_id"], "status": status},
                }
            ).encode()
        ),
    )
    with pytest.raises(named.NamedTunnelError, match = "^tunnel_status_unavailable$"):
        named._tunnel_status(operation, bundle)


@pytest.mark.parametrize(
    "failure", ["network", "invalid_json", "unsuccessful", "unexpected_payload"]
)
def test_unavailable_cloudflare_status_does_not_expose_raw_errors(
    named, removal_probe, monkeypatch, failure
):
    import io
    import urllib.error
    import urllib.request

    operation, bundle = removal_probe

    def unavailable(*args, **kwargs):
        if failure == "network":
            raise urllib.error.URLError("private-account-certificate")
        payload = {
            "invalid_json": b"not json",
            "unsuccessful": b'{"success":false,"errors":[{"message":"private-account-certificate"}]}',
            "unexpected_payload": b"[]",
        }[failure]
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(named.NamedTunnelError, match = "^tunnel_status_unavailable$"):
        named._tunnel_status(operation, bundle)


def test_cloudflare_removal_status_recognizes_an_already_deleted_tunnel(
    named, removal_probe, monkeypatch
):
    import io
    import urllib.request

    operation, bundle = removal_probe
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            json.dumps(
                {
                    "success": True,
                    "result": {"id": bundle["tunnel_id"], "deleted_at": "2026-10-22T00:00:00Z"},
                }
            ).encode()
        ),
    )
    assert named._tunnel_status(operation, bundle) == "deleted"


@pytest.mark.parametrize("shutdown", [False, True])
def test_cancel_or_shutdown_during_removal_status_read_invalidates_result(
    named, removal_probe, monkeypatch, shutdown
):
    import cloudflare_tunnel
    import io
    import urllib.request

    operation, bundle = removal_probe

    def cancelled(*args, **kwargs):
        if shutdown:
            cloudflare_tunnel.close_studio_tunnel_lifecycle()
        else:
            operation.cancel.set()
        return io.BytesIO(
            json.dumps(
                {
                    "success": True,
                    "result": {
                        "id": bundle["tunnel_id"],
                        "status": "down",
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", cancelled)
    with pytest.raises(named.NamedTunnelError, match = "^setup_cancelled$"):
        named._tunnel_status(operation, bundle)
