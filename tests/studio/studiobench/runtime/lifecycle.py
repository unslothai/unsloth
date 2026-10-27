# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vendored from studio_test_kit; stdlib-only, so the shipped zipapp needs nothing beyond Playwright."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class StudioInstall:
    home: Path
    repo: Path
    branch: str
    bootstrap_password: Optional[str] = None
    port: Optional[int] = None
    pid: Optional[int] = None
    # The commit `branch` resolved to; a branch or movable tag is not a build.
    commit: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# Fallback token TTL (backend ACCESS_TOKEN_EXPIRE_MINUTES); the JWT exp claim wins when readable.
ACCESS_TOKEN_TTL_S = 60 * 60
# Refresh margin: seeding a 1M-token thread is one request with a 900s timeout.
TOKEN_REFRESH_MARGIN_S = 15 * 60


def jwt_expiry(token: str) -> Optional[float]:
    """Reads exp without verifying anything; it is a clock for refresh, not authentication."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
        exp = claims.get("exp")
        return float(exp) if exp is not None else None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class StudioAuth:
    """Re-logs in before exp, not refresh: refresh tokens are single use, and the page holds one copy."""

    access_token: str
    refresh_token: str
    base_url: str
    username: str
    password: str
    expires_at: Optional[float] = None
    # Called after rotation; the browser context seeds localStorage from a snapshot and must re-seed.
    on_rotate: Optional[Callable[["StudioAuth"], None]] = None
    rotations: int = field(default = 0, init = False)
    # Disabled when a fresh token still reads as expiring (clock skew with the server).
    proactive: bool = field(default = True, init = False)
    hook_error: Optional[str] = field(default = None, init = False)

    def __post_init__(self) -> None:
        if self.expires_at is None:
            self.expires_at = jwt_expiry(self.access_token) or (time.time() + ACCESS_TOKEN_TTL_S)

    def seconds_left(self) -> float:
        return float(self.expires_at or 0) - time.time()

    def needs_refresh(self, margin_s: Optional[float] = None) -> bool:
        margin = TOKEN_REFRESH_MARGIN_S if margin_s is None else margin_s
        return self.seconds_left() <= margin

    def token(self, margin_s: Optional[float] = None) -> str:
        """The access token to send, re-minted first if it is close to expiring."""
        if self.proactive and self.needs_refresh(margin_s):
            self.rotate()
        return self.access_token

    def rotate(self) -> str:
        """Refresh is skipped for a token already inside the margin, or clock skew would re-login
        per request."""
        fresh = login(self.base_url, self.username, self.password)
        self.access_token = fresh.access_token
        self.refresh_token = fresh.refresh_token or self.refresh_token
        self.expires_at = fresh.expires_at
        self.rotations += 1
        if self.needs_refresh():
            self.proactive = False
        if self.on_rotate is not None:
            try:
                self.on_rotate(self)
            except Exception as exc:  # noqa: BLE001
                self.hook_error = f"{type(exc).__name__}: {exc}"
        return self.access_token


def auth_request_json(
    auth: StudioAuth,
    url: str,
    *,
    method: str = "GET",
    body: Optional[dict] = None,
    timeout: float = 30.0,
) -> Any:
    """Retries a 401 once after a fresh login; the login stays outside the try, so its 401 is not
    retried."""
    bearer = auth.token()
    try:
        return request_json(url, method = method, body = body, token = bearer, timeout = timeout)
    except HttpError as exc:
        if exc.status != 401:
            raise
    auth.rotate()
    return request_json(url, method = method, body = body, token = auth.access_token, timeout = timeout)


@dataclass
class ProviderSeed:
    """Must be custom, not openai: openai routes to /v1/responses, which never reaches this pacer."""

    provider_type: str
    name: str
    base_url: str
    models: list[str]
    api_key: str
    id: str = field(default_factory = lambda: uuid.uuid4().hex[:16])

    def as_provider_entry(self) -> dict:
        return {
            "id": self.id,
            "providerType": self.provider_type,
            "name": self.name,
            "baseUrl": self.base_url,
            "models": list(self.models),
        }


def pacer_provider(
    base_url: str,
    models: list[str],
    api_key: str = "sb-local",
) -> ProviderSeed:
    return ProviderSeed(
        provider_type = "custom",
        name = "studiobench pacer",
        base_url = base_url,
        models = models,
        api_key = api_key,
    )


def register_provider(base_url: str, auth: StudioAuth, provider: ProviderSeed) -> str:
    """The SPA checks selections against GET /api/providers/, so the id must be the backend's own."""
    existing = auth_request_json(auth, f"{base_url.rstrip('/')}/api/providers/") or []
    for row in existing:
        # Idempotent: each run binds a new pacer port, so a stale entry would be a dead duplicate model.
        if row.get("display_name") == provider.name:
            try:
                auth_request_json(
                    auth,
                    f"{base_url.rstrip('/')}/api/providers/{row['id']}",
                    method = "DELETE",
                )
            except HttpError:
                pass
    created = auth_request_json(
        auth,
        f"{base_url.rstrip('/')}/api/providers/",
        method = "POST",
        body = {
            "provider_type": provider.provider_type,
            "display_name": provider.name,
            "base_url": provider.base_url,
            "models": list(provider.models),
            "available_models": list(provider.models),
        },
    )
    provider.id = created["id"]
    return created["id"]


def external_checkpoint_id(provider: ProviderSeed, model_id: str) -> str:
    """Must use buildExternalModelId's format, which chat-runtime-store reads back on restore."""
    from urllib.parse import quote
    return f"external::{provider.id}::{quote(model_id, safe = '')}"


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:400]}")
        self.status = status
        self.body = body
        self.url = url


def request_json(
    url: str,
    *,
    method: str = "GET",
    body: Optional[dict] = None,
    token: Optional[str] = None,
    timeout: float = 30.0,
) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data = data, headers = headers, method = method)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read().decode("utf-8", "replace"), url) from exc
    return json.loads(raw) if raw.strip() else None


def wait_for_healthz(base_url: str, timeout_s: float = 180.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/healthz", timeout = 3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def _run(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    check: bool = True,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd = cwd,
        env = {**os.environ, **(env or {})},
        check = check,
        timeout = timeout,
        text = True,
        capture_output = True,
    )


def checkout_ref(repo: Path, ref: str) -> str:
    """Not git clone --branch: it only accepts branch and tag names, so a commit sha or ref^1 fails."""
    fetched = _run(["git", "fetch", "--tags", "origin", ref], cwd = repo, check = False)
    if fetched.returncode != 0:
        # The remote may not serve this ref by name (e.g. `ref^1`), so fetch everything and resolve locally.
        _run(["git", "fetch", "--tags", "origin"], cwd = repo, check = False)
    candidates = [] if fetched.returncode != 0 else ["FETCH_HEAD"]
    candidates += [f"origin/{ref}", ref]
    for candidate in candidates:
        got = _run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            cwd = repo,
            check = False,
        )
        commit = got.stdout.strip()
        if got.returncode == 0 and commit:
            _run(["git", "checkout", "--force", "--detach", commit], cwd = repo)
            _run(["git", "reset", "--hard", commit], cwd = repo)
            return commit
    raise RuntimeError(
        f"{ref!r} could not be resolved in {repo}: it is not a branch, a tag or a commit this "
        "remote will serve"
    )


# install.sh budget, excluded from tier wall clock; the watchdog adds it to its deadline.
INSTALL_TIMEOUT_S = 60 * 45


def install_studio(
    branch: str,
    home: Path,
    repo: Optional[Path] = None,
    remote: str = "https://github.com/unslothai/unsloth",
    reuse_clone: bool = True,
) -> StudioInstall:
    home = Path(home).resolve()
    home.mkdir(parents = True, exist_ok = True)
    repo = (repo or (home.parent / f"{home.name}_repo")).resolve()
    if not (reuse_clone and (repo / ".git").exists()):
        if repo.exists():
            shutil.rmtree(repo)
        # Clone without --branch so branches, tags and shas share one checkout path.
        _run(["git", "clone", remote, str(repo)])
    # Kept so a resumed run can detect that a branch moved underneath it.
    commit = checkout_ref(repo, branch)
    install_sh = repo / "install.sh"
    if not install_sh.exists():
        raise FileNotFoundError(f"install.sh missing at {install_sh}")
    _run(
        ["bash", str(install_sh), "--local"],
        cwd = repo,
        env = {"UNSLOTH_STUDIO_HOME": str(home)},
        timeout = INSTALL_TIMEOUT_S,
    )
    return StudioInstall(home = home, repo = repo, branch = branch, commit = commit)


def _find_unsloth_bin(install: StudioInstall) -> str:
    for candidate in (
        install.home / "bin" / "unsloth",
        install.home / ".venv_t5_550" / "bin" / "unsloth",
        install.home / ".venv_t5_530" / "bin" / "unsloth",
    ):
        if candidate.exists():
            return str(candidate)
    for venv in sorted(install.home.glob(".venv*")):
        candidate = venv / "bin" / "unsloth"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"`unsloth` CLI not found under {install.home}")


_PW_RE = re.compile(r"(?i)(?:bootstrap|initial|generated)\s*password(?:\s+is)?\s*[:=]?\s+(\S+)")


def _read_bootstrap_password(home: Path, log_path: Path, deadline: float) -> Optional[str]:
    boot_file = home / "auth" / ".bootstrap_password"
    while time.time() < deadline:
        try:
            if boot_file.exists():
                secret = boot_file.read_text(errors = "ignore").strip()
                if secret:
                    return secret
        except OSError:
            pass
        if log_path.exists():
            m = _PW_RE.search(log_path.read_text(errors = "ignore"))
            if m:
                return m.group(1).strip().strip(".,")
        time.sleep(0.5)
    return None


# Polled because the pid appears only after `setsid -f` returns; tests set it to zero.
PID_DISCOVERY_TIMEOUT_S = 15.0


def _discover_pid(port: int, timeout_s: Optional[float] = None) -> Optional[int]:
    """The pid Popen returns is setsid's, not the server's, so the server is found by port with pgrep."""
    if timeout_s is None:
        timeout_s = PID_DISCOVERY_TIMEOUT_S
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        try:
            out = _run(["pgrep", "-f", f"unsloth studio.*-p {port}"], check = False).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
        if out:
            try:
                return int(out.splitlines()[0])
            except ValueError:
                return None
        if time.time() >= deadline:
            return None
        time.sleep(0.5)


def port_is_busy(
    port: int,
    host: str = "127.0.0.1",
    timeout_s: float = 1.0,
) -> bool:
    """Connects rather than binds, since the detached server binds later; a busy port is not ours."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            return sock.connect_ex((host, int(port))) == 0
        except OSError:
            return False


def launch_studio(
    install: StudioInstall,
    port: int,
    log_path: Path,
    extra_env: Optional[dict] = None,
    healthz_timeout_s: int = 240,
    password_timeout_s: int = 30,
) -> StudioInstall:
    # Refuse a busy port before launching, or we would measure a stale server left by --keep-studio.
    if port_is_busy(port):
        holder = _discover_pid(port, 0.0)
        raise RuntimeError(
            f"port {port} is already in use"
            + (f" by Unsloth pid {holder}" if holder else "")
            + ". An Unsloth launched here would abort or land on another port while this harness "
            "measured whatever is already answering. Stop it (`unsloth studio stop`, or the "
            "Unsloth a previous --keep-studio run left behind) or pass --port."
        )
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents = True, exist_ok = True)
    log_path.write_text("")
    bin_path = _find_unsloth_bin(install)
    env = {"UNSLOTH_STUDIO_HOME": str(install.home), **(extra_env or {})}
    # The opt-in SSRF guard would reject the pacer's 127.0.0.1 base URL.
    env.pop("UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS", None)
    cmd = [
        "setsid",
        "-f",
        "bash",
        "-c",
        f"{shlex.quote(bin_path)} studio -p {port} 2>&1 | tee -a {shlex.quote(str(log_path))}",
    ]
    subprocess.Popen(
        cmd,
        env = {**os.environ, **env},
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        start_new_session = True,
    )
    install.port = port
    install.bootstrap_password = _read_bootstrap_password(
        install.home, log_path, time.time() + password_timeout_s
    )
    # Discover the pid before the health check so a never-healthy server can still be stopped.
    install.pid = _discover_pid(port)
    healthy = wait_for_healthz(install.base_url, healthz_timeout_s)
    if install.pid is None:
        install.pid = _discover_pid(port, 0.0)
    if not healthy:
        stop_studio(install)
        raise TimeoutError(f"Unsloth on :{port} did not pass /healthz within {healthz_timeout_s}s")
    return install


def stop_studio(install: StudioInstall) -> None:
    if install.pid:
        try:
            os.killpg(os.getpgid(install.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass


BENCH_PASSWORD = "studiobench-Passw0rd!"


def login(base_url: str, username: str, password: str) -> StudioAuth:
    body = request_json(
        f"{base_url}/api/auth/login",
        method = "POST",
        body = {"username": username, "password": password},
    )
    return StudioAuth(
        access_token = body["access_token"],
        refresh_token = body.get("refresh_token", ""),
        base_url = base_url,
        username = username,
        password = password,
        expires_at = jwt_expiry(body["access_token"]),
    )


def authenticate(
    base_url: str,
    username: str,
    password: str,
    new_password: str = BENCH_PASSWORD,
) -> StudioAuth:
    """Clears must_change_password first: until then every route returns 403 while /healthz returns 200."""
    # Also try the password a previous run rotated to, so reruns and --resume work.
    attempts = [password, new_password] if password != new_password else [password]
    auth = None
    last: Optional[Exception] = None
    for candidate in attempts:
        if not candidate:
            continue
        try:
            auth = login(base_url, username, candidate)
            password = candidate
            break
        except HttpError as exc:
            if exc.status != 401:
                raise
            last = exc
    if auth is None:
        raise RuntimeError(
            f"could not log in as {username!r} with the supplied password or with the password a "
            f"previous studiobench run would have rotated to. Last error: {last}"
        )
    try:
        status = request_json(f"{base_url}/api/auth/status", token = auth.access_token) or {}
    except HttpError:
        status = {}
    if status.get("requires_password_change"):
        body = request_json(
            f"{base_url}/api/auth/change-password",
            method = "POST",
            token = auth.access_token,
            body = {"current_password": password, "new_password": new_password},
        )
        auth = StudioAuth(
            access_token = body["access_token"],
            refresh_token = body.get("refresh_token", ""),
            base_url = base_url,
            username = username,
            password = new_password,
            expires_at = jwt_expiry(body["access_token"]),
        )
    return auth


def seed_init_script(
    auth: StudioAuth,
    providers: list[ProviderSeed],
    extra_local_storage: Optional[dict] = None,
) -> str:
    """Writes unsloth_auth_refresh_token, the app's key; each script writes only a later-expiring token."""
    auth_payload = {
        "unsloth_auth_token": auth.access_token,
        "unsloth_auth_refresh_token": auth.refresh_token,
    }
    payload = {
        "unsloth_chat_external_providers": json.dumps([p.as_provider_entry() for p in providers]),
        "unsloth_chat_external_provider_keys": json.dumps(
            {p.id: p.api_key for p in providers if p.api_key}
        ),
        "unsloth_chat_connections_enabled": "true",
    }
    for k, v in (extra_local_storage or {}).items():
        payload[k] = v if isinstance(v, str) else json.dumps(v)
    # Read exp from the unverified JWT (unpadded base64url); unreadable tokens score 0.
    exp_of = (
        "const expOf = (t) => { try { let p = String(t).split('.')[1]"
        ".replace(/-/g, '+').replace(/_/g, '/'); while (p.length % 4) p += '='; "
        "return Number(JSON.parse(window.atob(p)).exp) || 0; } catch (e) { return 0; } };"
    )
    return (
        "(() => { const seed = "
        + json.dumps(payload)
        + "; for (const k of Object.keys(seed)) { try { window.localStorage.setItem(k, seed[k]);"
        " } catch (e) {} } const auth = "
        + json.dumps(auth_payload)
        + "; "
        + exp_of
        + " try { const held = window.localStorage.getItem('unsloth_auth_token');"
        " if (!held || expOf(held) < expOf(auth['unsloth_auth_token'])) {"
        " for (const k of Object.keys(auth)) window.localStorage.setItem(k, auth[k]); }"
        " } catch (e) {} })();"
    )
