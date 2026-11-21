# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ipaddress
import base64
import json
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit

from filelock import FileLock, Timeout

MODE_KEY = "remote_access_tunnel_mode"
_SECRET_KIND = "cloudflare_named_tunnel"
_SECRET_SCOPE = "default"
_SETUP_TIMEOUT = 15 * 60
_COMMAND_TIMEOUT = 60
_operation_lock = threading.RLock()
_operation: NamedTunnelOperation | None = None


class NamedTunnelError(RuntimeError):
    pass


def normalize_hostname(value: str) -> str:
    if not isinstance(value, str) or any(c.isspace() for c in value):
        raise ValueError("invalid_hostname")
    try:
        hostname = value.removesuffix(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise ValueError("invalid_hostname") from None
    labels = hostname.split(".")
    if (
        len(hostname) > 253
        or len(labels) < 2
        or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)
    ):
        raise ValueError("invalid_hostname")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return hostname
    raise ValueError("invalid_hostname")


def get_mode() -> str:
    from storage.studio_db import get_app_setting

    mode = get_app_setting(MODE_KEY, "quick")
    if not isinstance(mode, str) or mode not in {"quick", "named"}:
        raise NamedTunnelError("configuration_invalid")
    return mode


def set_mode(mode: str) -> None:
    if mode not in {"quick", "named"}:
        raise ValueError("invalid_tunnel_mode")
    from storage.studio_db import upsert_app_settings

    lock = _acquire_lock()
    try:
        upsert_app_settings({MODE_KEY: mode})
    finally:
        lock.release()


def _acquire_lock() -> FileLock:
    from utils.paths import studio_root

    path = studio_root() / ".cloudflare-named-tunnel.lock"
    path.parent.mkdir(parents = True, exist_ok = True)
    lock = FileLock(str(path), thread_local = False)
    try:
        lock.acquire(timeout = 0)
    except Timeout:
        raise NamedTunnelError("named_tunnel_busy") from None
    return lock


def load_bundle() -> dict | None:
    from storage.credential_secrets import get_secret_with_presence

    raw, present = get_secret_with_presence(_SECRET_KIND, _SECRET_SCOPE)
    if not present:
        return None
    if raw is None:
        raise NamedTunnelError("credentials_unreadable")
    try:
        bundle = json.loads(raw)
        uuid.UUID(bundle["configuration_id"])
        if normalize_hostname(bundle["hostname"]) != bundle["hostname"]:
            raise ValueError()
        if bundle["method"] == "token":
            if not isinstance(bundle["token"], str) or not bundle["token"]:
                raise ValueError()
        elif bundle["method"] == "automatic":
            if not re.fullmatch(r"unsloth-studio-[0-9a-f]{32}", bundle["tunnel_name"]):
                raise ValueError()
            if bundle.get("tunnel_id"):
                uuid.UUID(bundle["tunnel_id"])
            if not isinstance(bundle.get("ready", False), bool) or (
                "certificate" in bundle and not isinstance(bundle["certificate"], str)
            ):
                raise ValueError()
            if bundle.get("credentials"):
                credentials = bundle["credentials"]
                if credentials["TunnelID"] != bundle["tunnel_id"] or not all(
                    isinstance(credentials[key], str) and credentials[key]
                    for key in ("AccountTag", "TunnelSecret")
                ):
                    raise ValueError()
            if bundle.get("ready") and not (
                bundle.get("tunnel_id") and bundle.get("credentials") and bundle.get("certificate")
            ):
                raise ValueError()
        else:
            raise ValueError()
        return bundle
    except (ValueError, TypeError, KeyError, AttributeError):
        raise NamedTunnelError("configuration_invalid") from None


def _save_bundle(bundle: dict) -> None:
    from storage.credential_secrets import upsert_secret
    upsert_secret(_SECRET_KIND, _SECRET_SCOPE, json.dumps(bundle))


def _public_configuration(bundle: dict | None) -> dict | None:
    if bundle is None:
        return None
    return {
        "configuration_id": bundle["configuration_id"],
        "method": bundle["method"],
        "hostname": bundle["hostname"],
        "ready": bundle["method"] == "token" or bool(bundle.get("ready")),
        "tunnel_id": bundle.get("tunnel_id"),
        "dns_cleanup_hostname": bundle.get("dns_cleanup_hostname"),
    }


def save_token(hostname: str, token: str | None) -> dict:
    hostname = normalize_hostname(hostname)
    lock = _acquire_lock()
    try:
        bundle = load_bundle()
        if bundle and (bundle["method"] != "token" or bundle["hostname"] != hostname):
            raise NamedTunnelError("remove_configuration_first")
        if token is None:
            if bundle is None:
                raise NamedTunnelError("token_required")
            return _public_configuration(bundle)
        if not token or len(token) > 16384 or any(c.isspace() for c in token):
            raise ValueError("invalid_tunnel_token")
        bundle = {
            "configuration_id": str(uuid.uuid4()),
            "method": "token",
            "hostname": hostname,
            "token": token,
        }
        _save_bundle(bundle)
        return _public_configuration(bundle)
    finally:
        lock.release()


def named_status() -> dict:
    from utils.paths import studio_root
    with _operation_lock:
        bundle = load_bundle()
        return {
            "configuration": _public_configuration(bundle),
            "setup": _operation.status()
            if _operation
            and _operation.root == str(studio_root())
            and (bundle is None or bundle["configuration_id"] == _operation.configuration_id)
            else None,
        }


@dataclass
class NamedTunnelOperation:
    hostname: str
    configuration_id: str
    admission: tuple[int, int]
    root: str
    reauthenticate: bool = False
    operation_id: str = field(default_factory = lambda: str(uuid.uuid4()))
    state: str = "preparing"
    error: str | None = None
    auth_url: str | None = None
    tunnel_id: str | None = None
    dns_cleanup_hostname: str | None = None
    deadline: float = field(default_factory = lambda: time.monotonic() + _SETUP_TIMEOUT)
    cancel: threading.Event = field(default_factory = threading.Event)
    confirmed: threading.Event = field(default_factory = threading.Event)
    thread: threading.Thread | None = None

    def status(self) -> dict:
        return {
            "operation_id": self.operation_id,
            "state": self.state,
            "hostname": self.hostname,
            "tunnel_id": self.tunnel_id,
            "auth_url": self.auth_url,
            "error": self.error,
            "dns_cleanup_hostname": self.dns_cleanup_hostname,
            "active": self.thread is not None and self.thread.is_alive(),
        }


def child_environment(home: Path) -> dict:
    from cloudflare_tunnel import _cloudflared_environment

    environment = _cloudflared_environment()
    # cloudflared honors HOME on Windows; USERPROFILE belongs to the login browser.
    environment["HOME"] = str(home)
    return environment


def login_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme == "https"
            and parsed.netloc == "dash.cloudflare.com"
            and parsed.path == "/argotunnel"
            and not parsed.fragment
            and not any(c.isspace() for c in value)
        ):
            return value
    except ValueError:
        pass
    return None


def _check_current(operation: NamedTunnelOperation) -> None:
    from cloudflare_tunnel import capture_studio_tunnel_start_admission
    if (
        operation is not _operation
        or operation.cancel.is_set()
        or capture_studio_tunnel_start_admission() != operation.admission
    ):
        raise NamedTunnelError("setup_cancelled")
    if time.monotonic() >= operation.deadline:
        raise NamedTunnelError("setup_timed_out")


def _save_current(operation: NamedTunnelOperation, bundle: dict) -> None:
    with _operation_lock:
        _check_current(operation)
        _save_bundle(bundle)


def _stop_command(proc: subprocess.Popen) -> bool:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout = 5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout = 5)
        return proc.poll() is not None
    except Exception:
        return False


def _run_command(
    operation: NamedTunnelOperation,
    binary: str,
    args: list[str],
    home: Path,
    timeout: float = _COMMAND_TIMEOUT,
) -> tuple[int, str]:
    from cloudflare_tunnel import (
        _adopt_pid,
        _forget_pid,
        _lifetime_kwargs,
        _spawn_child,
        _windows_hidden_kwargs,
    )

    config = home / "config.json"
    config.write_text("{}", encoding = "utf-8")
    cmd = [binary, "tunnel", "--config", str(config), "--no-autoupdate", "--output", "json"]
    if "login" not in args:
        cmd += ["--origincert", str(home / ".cloudflared" / "cert.pem")]
    cmd += args
    with _operation_lock:
        _check_current(operation)
        proc = _spawn_child(
            lambda: subprocess.Popen(
                cmd,
                env = child_environment(home),
                # The login browser can outlive this home and pin its directory on Windows.
                cwd = home.parent,
                stdin = subprocess.DEVNULL,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                **_windows_hidden_kwargs(),
                **_lifetime_kwargs(),
            )
        )
        _adopt_pid(proc.pid)
    output = deque(maxlen = 128)

    def _read() -> None:
        for line in proc.stdout:
            output.append(line[:16384])
            if "login" in args:
                for candidate in re.findall(r"https://[^\s\"<>]+", _cli_messages(line)):
                    url = login_url(candidate)
                    if url:
                        with _operation_lock:
                            if not operation.cancel.is_set():
                                operation.auth_url = url
                                operation.state = "awaiting_login"

    reader = threading.Thread(target = _read, daemon = True, name = "cloudflared-setup-output")
    reader.start()
    deadline = min(operation.deadline, time.monotonic() + timeout)
    try:
        while proc.poll() is None:
            _check_current(operation)
            if time.monotonic() >= deadline:
                raise NamedTunnelError("command_timed_out")
            operation.cancel.wait(0.05)
        reader.join(1)
        _check_current(operation)
        return proc.returncode, "".join(output)
    finally:
        if not _stop_command(proc):
            with _operation_lock:
                operation.state, operation.error = "error", "cloudflared_could_not_be_stopped"
            # Keep the installation lease and private files until the child has actually exited.
            while proc.poll() is None:
                time.sleep(0.1)
        _forget_pid(proc.pid)


def _cli_messages(output: str) -> str:
    messages = []
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            messages.append(line)
            continue
        if isinstance(record, dict):
            messages.append(str(record.get("message", "")))
            messages.append(str(record.get("error", "")))
    return "\n".join(messages)


def _listed_tunnels(output: str, code: int = 0) -> list[dict]:
    if code:
        if re.search(r"\bServer returned:\s*(?:401|403)\b", _cli_messages(output)):
            raise NamedTunnelError("cloudflare_login_required")
        raise NamedTunnelError("tunnel_lookup_failed")
    match = re.search(r"(?m)^\s*(?:\[|null\b)", output)
    try:
        tunnels, _ = json.JSONDecoder().raw_decode(output[match.start() :].lstrip())
        if tunnels is None:
            return []
        if not isinstance(tunnels, list) or not all(isinstance(item, dict) for item in tunnels):
            raise ValueError()
        return tunnels
    except (ValueError, AttributeError):
        raise NamedTunnelError("tunnel_lookup_failed") from None


def _read_credentials(path: Path, tunnel_id: str | None = None) -> dict:
    try:
        credentials = json.loads(path.read_text(encoding = "utf-8"))
        parsed_id = str(uuid.UUID(credentials["TunnelID"]))
        if tunnel_id and tunnel_id != parsed_id:
            raise ValueError()
        if not all(
            isinstance(credentials[key], str) and credentials[key]
            for key in ("AccountTag", "TunnelSecret")
        ):
            raise ValueError()
        return {key: credentials[key] for key in ("AccountTag", "TunnelSecret", "TunnelID")}
    except (OSError, ValueError, KeyError, TypeError):
        raise NamedTunnelError("tunnel_credentials_invalid") from None


def _certificate_data(certificate: str) -> dict:
    try:
        match = re.search(
            r"-----BEGIN ARGO TUNNEL TOKEN-----\s+([A-Za-z0-9+/=\s]+)-----END ARGO TUNNEL TOKEN-----",
            certificate,
        )
        data = json.loads(base64.b64decode("".join(match[1].split()), validate = True))
        if not all(
            isinstance(data[key], str) and data[key] for key in ("accountID", "zoneID", "apiToken")
        ):
            raise ValueError()
        return data
    except (ValueError, TypeError, KeyError):
        raise NamedTunnelError("cloudflare_login_failed") from None


def _zone_name(operation: NamedTunnelOperation, bundle: dict) -> str:
    import urllib.error
    import urllib.request

    _check_current(operation)
    certificate = _certificate_data(bundle["certificate"])
    try:
        request = urllib.request.Request(
            "https://api.cloudflare.com/client/v4/zones/" + quote(certificate["zoneID"], safe = ""),
            headers = {"Authorization": "Bearer " + certificate["apiToken"]},
        )
        with urllib.request.urlopen(
            request, timeout = min(_COMMAND_TIMEOUT, operation.deadline - time.monotonic())
        ) as response:
            payload = json.loads(response.read(65536))
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError()
        zone = normalize_hostname(payload["result"]["name"])
    except urllib.error.HTTPError as exc:
        _check_current(operation)
        if exc.code in {401, 403}:
            raise NamedTunnelError("cloudflare_login_required") from None
        raise NamedTunnelError("dns_route_failed") from None
    except (OSError, ValueError, TypeError, KeyError):
        _check_current(operation)
        raise NamedTunnelError("dns_route_failed") from None
    _check_current(operation)
    return zone


def _route_dns(operation: NamedTunnelOperation, binary: str, home: Path, bundle: dict) -> None:
    zone = _zone_name(operation, bundle)
    if operation.hostname != zone and not operation.hostname.endswith("." + zone):
        raise NamedTunnelError("dns_hostname_mismatch")
    # Cloudflare may accept a route just before cancellation or a lost CLI response.
    bundle["dns_cleanup_hostname"] = operation.hostname
    _save_current(operation, bundle)
    args = ["route", "dns", bundle["tunnel_id"], bundle["hostname"]]
    code, output = _run_command(operation, binary, args, home)
    output = _cli_messages(output)
    if (
        code
        and re.search(r"\b1003\b", output)
        and re.search(
            r"An A, AAAA,? or CNAME record (?:already exists with that host|with that host already exists)",
            output,
        )
    ):
        with _operation_lock:
            _check_current(operation)
            operation.state = "dns_conflict"
        while not operation.confirmed.wait(0.05):
            _check_current(operation)
        _check_current(operation)
        code, output = _run_command(
            operation,
            binary,
            ["route", "dns", "--overwrite-dns", bundle["tunnel_id"], bundle["hostname"]],
            home,
        )
        output = _cli_messages(output)
    if code:
        raise NamedTunnelError("dns_route_failed")
    match = re.search(
        r"(?:Added CNAME (\S+) which will route to this tunnel|(\S+) (?:updated to route to your tunnel|is already configured to route to your tunnel))",
        output,
    )
    if not match:
        raise NamedTunnelError("dns_result_unrecognized")
    actual = normalize_hostname(next(value for value in match.groups() if value))
    bundle["dns_cleanup_hostname"] = actual
    _save_current(operation, bundle)
    if actual != operation.hostname:
        operation.dns_cleanup_hostname = actual
        raise NamedTunnelError("dns_hostname_mismatch")


def _automatic_setup(
    operation: NamedTunnelOperation, binary: str, home: Path, bundle: dict
) -> None:
    cert = home / ".cloudflared" / "cert.pem"
    cert.parent.mkdir(mode = 0o700)
    if bundle.get("certificate") and not operation.reauthenticate:
        cert.write_text(bundle["certificate"], encoding = "utf-8")
    else:
        operation.state = "awaiting_login"
        code, _ = _run_command(operation, binary, ["login"], home, timeout = _SETUP_TIMEOUT)
        if code or not cert.is_file():
            raise NamedTunnelError("cloudflare_login_failed")
        certificate = cert.read_text(encoding = "utf-8")
        account = _certificate_data(certificate)["accountID"]
        expected = bundle.get("account_id") or (bundle.get("credentials") or {}).get("AccountTag")
        if expected and account != expected:
            raise NamedTunnelError("cloudflare_login_required")
        bundle.update(certificate = certificate, account_id = account)
        _save_current(operation, bundle)
    operation.auth_url = None
    operation.state = "configuring"
    credentials_path = home / "credentials.json"
    if not bundle.get("credentials"):
        if not bundle.get("tunnel_id"):
            code, output = _run_command(
                operation,
                binary,
                ["list", "--name", bundle["tunnel_name"], "--output", "json"],
                home,
            )
            try:
                matches = [
                    item
                    for item in _listed_tunnels(output, code)
                    if item["name"] == bundle["tunnel_name"]
                ]
            except (ValueError, TypeError, KeyError):
                raise NamedTunnelError("tunnel_lookup_failed") from None
            if matches:
                bundle["tunnel_id"] = str(uuid.UUID(matches[0]["id"]))
                _save_current(operation, bundle)
        if bundle.get("tunnel_id"):
            code, _ = _run_command(
                operation,
                binary,
                ["token", "--cred-file", str(credentials_path), bundle["tunnel_id"]],
                home,
            )
        else:
            code, _ = _run_command(
                operation,
                binary,
                ["create", "--credentials-file", str(credentials_path), bundle["tunnel_name"]],
                home,
            )
        if credentials_path.is_file():
            credentials = _read_credentials(credentials_path, bundle.get("tunnel_id"))
            if credentials["AccountTag"] != bundle["account_id"]:
                raise NamedTunnelError("cloudflare_login_required")
            bundle["credentials"] = credentials
            bundle["tunnel_id"] = bundle["credentials"]["TunnelID"]
            _save_current(operation, bundle)
        if code or not bundle.get("credentials"):
            raise NamedTunnelError("tunnel_create_failed")
    operation.tunnel_id = bundle["tunnel_id"]
    _route_dns(operation, binary, home, bundle)
    bundle["ready"] = True
    _save_current(operation, bundle)
    operation.state = "ready"


def _setup_worker(
    operation: NamedTunnelOperation, lock: FileLock, bundle: dict, remove: bool
) -> None:
    from cloudflare_tunnel import ensure_cloudflared
    from utils.paths.private_tmp import private_tmp_dir
    try:
        binary = ensure_cloudflared()
        if not binary:
            raise NamedTunnelError("cloudflared_unavailable")
        _check_current(operation)
        with tempfile.TemporaryDirectory(dir = private_tmp_dir("unsloth-cloudflared")) as directory:
            home = Path(directory)
            if remove:
                _remove_automatic(operation, binary, home, bundle)
            else:
                _automatic_setup(operation, binary, home, bundle)
    except NamedTunnelError as exc:
        operation.state, operation.error = "error", str(exc)
    except Exception:
        operation.state, operation.error = "error", "named_setup_failed"
    finally:
        operation.auth_url = None
        lock.release()


def start_setup(hostname: str, *, reauthenticate: bool = False) -> dict:
    global _operation
    from cloudflare_tunnel import capture_studio_tunnel_start_admission
    from utils.paths import studio_root

    hostname = normalize_hostname(hostname)
    with _operation_lock:
        if _operation and _operation.thread and _operation.thread.is_alive():
            if _operation.hostname == hostname and _operation.state != "removing":
                return _operation.status()
            raise NamedTunnelError("named_tunnel_busy")
        admission = capture_studio_tunnel_start_admission()
        if admission is None:
            raise NamedTunnelError("server_shutting_down")
        lock = _acquire_lock()
        try:
            bundle = load_bundle()
            if bundle and (bundle["method"] != "automatic" or bundle["hostname"] != hostname):
                raise NamedTunnelError("remove_configuration_first")
            if bundle is None:
                bundle = {
                    "configuration_id": str(uuid.uuid4()),
                    "method": "automatic",
                    "hostname": hostname,
                    "tunnel_name": "unsloth-studio-" + uuid.uuid4().hex,
                    "ready": False,
                }
            bundle["ready"] = False
            _save_bundle(bundle)
            operation = NamedTunnelOperation(
                hostname,
                bundle["configuration_id"],
                admission,
                str(studio_root()),
                reauthenticate = reauthenticate,
                tunnel_id = bundle.get("tunnel_id"),
            )
            _operation = operation
            operation.thread = threading.Thread(
                target = _setup_worker,
                args = (operation, lock, bundle, False),
                daemon = True,
                name = "cloudflared-setup",
            )
            operation.thread.start()
            return operation.status()
        except Exception:
            lock.release()
            raise


def confirm_dns(operation_id: str, hostname: str, tunnel_id: str) -> None:
    with _operation_lock:
        operation = _operation
        if not operation or (
            operation.operation_id != operation_id
            or operation.hostname != normalize_hostname(hostname)
            or operation.tunnel_id != tunnel_id
            or operation.state != "dns_conflict"
            or operation.cancel.is_set()
            or operation.confirmed.is_set()
        ):
            raise NamedTunnelError("stale_dns_confirmation")
        _check_current(operation)
        operation.state = "configuring"
        operation.confirmed.set()


def cancel_setup(operation_id: str | None = None) -> None:
    with _operation_lock:
        if _operation and (operation_id is None or operation_id == _operation.operation_id):
            _operation.cancel.set()
        elif operation_id:
            raise NamedTunnelError("stale_setup_operation")


def _delete_bundle(operation: NamedTunnelOperation | None = None) -> None:
    from storage.credential_secrets import delete_secret
    with _operation_lock:
        if operation:
            _check_current(operation)
        delete_secret(_SECRET_KIND, _SECRET_SCOPE)


def _remove_automatic(
    operation: NamedTunnelOperation, binary: str, home: Path, bundle: dict
) -> None:
    if bundle.get("certificate"):
        if _certificate_data(bundle["certificate"])["accountID"] != bundle.get("account_id"):
            raise NamedTunnelError("cloudflare_login_required")
        cert = home / ".cloudflared" / "cert.pem"
        cert.parent.mkdir(mode = 0o700)
        cert.write_text(bundle["certificate"], encoding = "utf-8")
        if not bundle.get("tunnel_id"):
            code, output = _run_command(
                operation,
                binary,
                ["list", "--name", bundle["tunnel_name"], "--output", "json"],
                home,
            )
            matches = [
                item
                for item in _listed_tunnels(output, code)
                if item["name"] == bundle["tunnel_name"]
            ]
            if matches:
                bundle["tunnel_id"] = str(uuid.UUID(matches[0]["id"]))
                _save_current(operation, bundle)
    elif bundle.get("tunnel_id"):
        raise NamedTunnelError("cloudflare_login_required")
    if bundle.get("tunnel_id"):
        status = _tunnel_status(operation, bundle)
        if status in {"healthy", "degraded"}:
            raise NamedTunnelError("tunnel_still_active")
        if status == "not_found":
            code, output = _run_command(
                operation, binary, ["list", "--id", bundle["tunnel_id"], "--output", "json"], home
            )
            try:
                absent = not code and not _listed_tunnels(output)
            except NamedTunnelError:
                absent = False
            if not absent:
                raise NamedTunnelError("tunnel_status_unavailable")
        elif status != "deleted":
            code, output = _run_command(operation, binary, ["delete", bundle["tunnel_id"]], home)
            if code:
                if re.search(
                    r"\bCannot delete tunnel (?:because it has|with) active connections\b",
                    _cli_messages(output),
                    re.IGNORECASE,
                ):
                    raise NamedTunnelError("tunnel_still_active")
                code, output = _run_command(
                    operation,
                    binary,
                    ["list", "--id", bundle["tunnel_id"], "--output", "json"],
                    home,
                )
                if code or _listed_tunnels(output):
                    raise NamedTunnelError("tunnel_delete_failed")
    _delete_bundle(operation)
    operation.dns_cleanup_hostname = bundle.get("dns_cleanup_hostname")
    operation.state = "removed"


def _tunnel_status(operation: NamedTunnelOperation, bundle: dict) -> str:
    import urllib.error
    import urllib.request

    _check_current(operation)
    certificate = _certificate_data(bundle["certificate"])
    request = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/"
        + quote(certificate["accountID"], safe = "")
        + "/cfd_tunnel/"
        + quote(bundle["tunnel_id"], safe = ""),
        headers = {"Authorization": "Bearer " + certificate["apiToken"]},
    )
    try:
        with urllib.request.urlopen(
            request, timeout = min(_COMMAND_TIMEOUT, operation.deadline - time.monotonic())
        ) as response:
            payload = json.loads(response.read(65536))
    except urllib.error.HTTPError as exc:
        _check_current(operation)
        if exc.code == 404:
            return "not_found"
        if exc.code in {401, 403}:
            raise NamedTunnelError("cloudflare_login_required") from None
        raise NamedTunnelError("tunnel_status_unavailable") from None
    except (OSError, ValueError):
        _check_current(operation)
        raise NamedTunnelError("tunnel_status_unavailable") from None
    _check_current(operation)
    result = payload.get("result") if isinstance(payload, dict) else None
    if (
        not isinstance(result, dict)
        or payload.get("success") is not True
        or result.get("id") != bundle["tunnel_id"]
    ):
        raise NamedTunnelError("tunnel_status_unavailable")
    if result.get("deleted_at"):
        if not isinstance(result["deleted_at"], str):
            raise NamedTunnelError("tunnel_status_unavailable")
        return "deleted"
    status = result.get("status")
    if not isinstance(status, str) or status not in {"down", "inactive", "healthy", "degraded"}:
        raise NamedTunnelError("tunnel_status_unavailable")
    return status


def remove_named(configuration_id: str) -> None:
    global _operation
    from cloudflare_tunnel import capture_studio_tunnel_start_admission
    from utils.paths import studio_root

    with _operation_lock:
        lock = _acquire_lock()
        try:
            bundle = load_bundle()
            if not bundle or bundle["configuration_id"] != configuration_id:
                raise NamedTunnelError("configuration_changed")
            admission = capture_studio_tunnel_start_admission()
            if admission is None:
                raise NamedTunnelError("server_shutting_down")
            if bundle["method"] == "token":
                _delete_bundle()
                _operation = None
                lock.release()
                return
            operation = NamedTunnelOperation(
                bundle["hostname"],
                configuration_id,
                admission,
                str(studio_root()),
                state = "removing",
                tunnel_id = bundle.get("tunnel_id"),
            )
            _operation = operation
            operation.thread = threading.Thread(
                target = _setup_worker,
                args = (operation, lock, bundle, True),
                daemon = True,
                name = "cloudflared-remove",
            )
            operation.thread.start()
        except Exception:
            lock.release()
            raise


def prepare_connector(binary: str, bundle: dict, origin: str, protocol: str | None):
    from utils.paths.private_tmp import private_tmp_dir

    lock = _acquire_lock()
    directory = None
    try:
        current = load_bundle()
        if (
            get_mode() != "named"
            or not current
            or bundle["configuration_id"] != current["configuration_id"]
        ):
            raise NamedTunnelError("configuration_changed")
        if current["method"] == "automatic" and not current.get("ready"):
            raise NamedTunnelError("named_setup_required")
        directory = tempfile.TemporaryDirectory(dir = private_tmp_dir("unsloth-cloudflared"))
        home = Path(directory.name)
        environment = child_environment(home)
        config = {}
        if current["method"] == "token":
            environment["TUNNEL_TOKEN"] = current["token"]
        else:
            environment["TUNNEL_CRED_CONTENTS"] = json.dumps(current["credentials"])
            config = {
                "tunnel": current["tunnel_id"],
                "ingress": [
                    {"hostname": current["hostname"], "service": origin},
                    {"service": "http_status:404"},
                ],
            }
        config_path = home / "config.json"
        config_path.write_text(json.dumps(config), encoding = "utf-8")
        command = [binary, "tunnel", "--config", str(config_path), "--no-autoupdate"]
        if protocol:
            command += ["--protocol", protocol]
        command += ["run"]
        if current["method"] == "automatic":
            command += [current["tunnel_id"]]

        def cleanup() -> None:
            directory.cleanup()
            lock.release()

        return command, environment, cleanup
    except Exception:
        if directory:
            directory.cleanup()
        lock.release()
        raise


def expected_root_id() -> str | None:
    from utils.paths import studio_root
    try:
        value = (studio_root() / "share" / "studio_install_id").read_text(encoding = "utf-8").strip()
    except OSError:
        return None
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None
