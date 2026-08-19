# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install, launch and authenticate a real Studio. VENDORED from `studio_test_kit`.

Vendored on purpose. The shipped artifact is a single `studiobench.pyz` a tester runs on a machine
that has a Studio and nothing else, so it cannot import a module that lives elsewhere in this
repository. The logic is `studio_test_kit.lifecycle` plus `studio_test_kit.auth`, with two
deliberate changes:

- **stdlib only.** `studio_test_kit.auth` uses `httpx`; this uses `urllib.request`, so `--doctor`
  and `--attach` work on a machine with nothing pip-installed but Playwright.
- **The password-change gate is handled.** A current Studio mints a bootstrap password and sets
  `must_change_password`, and until it is cleared EVERY authenticated route answers
  `403 Password change required` while `/healthz` answers 200 and login itself succeeds. That
  failure is silent one request too late, and it is what stops a thread from being seeded at all.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class StudioInstall:
    home: Path
    repo: Path
    branch: str
    bootstrap_password: Optional[str] = None
    port: Optional[int] = None
    pid: Optional[int] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class StudioAuth:
    access_token: str
    refresh_token: str
    base_url: str
    username: str
    password: str


@dataclass
class ProviderSeed:
    """One external provider entry for the SPA's localStorage.

    `provider_type` is **"custom"**, not "openai", and the difference decides whether this
    benchmark measures anything. The backend routes `openai` to `/v1/responses`; `custom` is the
    generic OpenAI-compatible entry and is routed to `{base_url}/chat/completions` with the body
    relayed and the SSE lines forwarded verbatim, which is the path a real llama.cpp, vLLM or
    Ollama server takes and the only one that puts our own event stream in front of the app's
    parser.
    """

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
    """Create the provider in the BACKEND and return the id IT assigned.

    LOCALSTORAGE SEEDING ALONE IS NOT ENOUGH ON A CURRENT STUDIO, and this is the second half of
    the reason nothing was ever generated. `studio_test_kit.auth` seeds three localStorage keys and
    its docstring says that is sufficient to drive any external provider end to end. It was; it is
    not now. The provider list the SPA validates a selection against comes from
    `GET /api/providers/`, so with a provider only in localStorage the model picker renders the
    model with the label "No longer offered" and pressing send throws `Connection not found`
    before any request is made. Both symptoms were observed directly against a shipped build.

    So the provider is created over REST and the id the BACKEND assigns is what the selection
    checkpoint must name. A client-generated id would not be found either.

    No API key is sent. For `custom` the backend omits the Authorization header entirely when the
    key is empty, which is what a local llama.cpp or vLLM server expects, and it saves this
    harness from having to RSA-encrypt a dummy secret against the server's published public key
    just to have it ignored.
    """
    existing = request_json(f"{base_url.rstrip('/')}/api/providers/", token = auth.access_token) or []
    for row in existing:
        # Idempotent across runs. Every run binds a NEW ephemeral pacer port, so a stale entry
        # from a previous run points at a port nothing is listening on, and leaving it there gives
        # the picker two identically named models of which one is dead.
        if row.get("display_name") == provider.name:
            try:
                request_json(
                    f"{base_url.rstrip('/')}/api/providers/{row['id']}",
                    method = "DELETE",
                    token = auth.access_token,
                )
            except HttpError:
                pass
    created = request_json(
        f"{base_url.rstrip('/')}/api/providers/",
        method = "POST",
        token = auth.access_token,
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
    """The app's own id for "this model, on this external provider".

    SEEDING THE PROVIDER IS NOT ENOUGH, and this is the difference between a benchmark and a page
    that does nothing. With the provider seeded but no model SELECTED, the composer accepts text,
    the send button is enabled, the message is stored to the thread -- and no completion request
    is ever made. Measured: the first end-to-end run finished every action, reported nine of
    sixteen ran, and recorded an assistant message with zero characters, because the reply that
    was supposed to be measured was never asked for. The pacer's request log was empty, which is
    the only reason it was caught.

    `chat-runtime-store.ts` restores the selection from
    `unsloth_chat_last_external_checkpoint`, in the format `buildExternalModelId` produces.
    """
    from urllib.parse import quote
    return f"external::{provider.id}::{quote(model_id, safe = '')}"


# ── HTTP, stdlib ────────────────────────────────────────────────────


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


# ── install and launch ──────────────────────────────────────────────


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
    if reuse_clone and (repo / ".git").exists():
        _run(["git", "fetch", "origin", branch], cwd = repo)
        _run(["git", "checkout", branch], cwd = repo)
        _run(["git", "reset", "--hard", f"origin/{branch}"], cwd = repo)
    else:
        if repo.exists():
            shutil.rmtree(repo)
        _run(["git", "clone", "--branch", branch, remote, str(repo)])
    install_sh = repo / "install.sh"
    if not install_sh.exists():
        raise FileNotFoundError(f"install.sh missing at {install_sh}")
    _run(
        ["bash", str(install_sh), "--local"],
        cwd = repo,
        env = {"UNSLOTH_STUDIO_HOME": str(home)},
        timeout = 60 * 45,
    )
    return StudioInstall(home = home, repo = repo, branch = branch)


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


def launch_studio(
    install: StudioInstall,
    port: int,
    log_path: Path,
    extra_env: Optional[dict] = None,
    healthz_timeout_s: int = 240,
    password_timeout_s: int = 30,
) -> StudioInstall:
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents = True, exist_ok = True)
    log_path.write_text("")
    bin_path = _find_unsloth_bin(install)
    env = {"UNSLOTH_STUDIO_HOME": str(install.home), **(extra_env or {})}
    # NOT `UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1`: that opt-in SSRF guard rejects any
    # non-global address, which is exactly what the pacer's 127.0.0.1 base URL is.
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
    if not wait_for_healthz(install.base_url, healthz_timeout_s):
        raise TimeoutError(f"Studio on :{port} did not pass /healthz within {healthz_timeout_s}s")
    try:
        out = _run(["pgrep", "-f", f"unsloth studio.*-p {port}"], check = False).stdout.strip()
        if out:
            install.pid = int(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    return install


def stop_studio(install: StudioInstall) -> None:
    if install.pid:
        try:
            os.killpg(os.getpgid(install.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass


# ── auth ────────────────────────────────────────────────────────────

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
    )


def authenticate(
    base_url: str,
    username: str,
    password: str,
    new_password: str = BENCH_PASSWORD,
) -> StudioAuth:
    """Log in and CLEAR the password-change gate if it is set.

    Not optional and not cosmetic. When `must_change_password` is set, login succeeds and returns
    a token, `/healthz` answers 200, and every route this harness needs -- create a thread, seed
    its messages, read them back -- answers `403 Password change required`. Nothing announces it;
    the run simply fails to seed a thread and reports an empty one. So the gate is read from
    `/api/auth/status` up front and cleared through the one endpoint that accepts the gated token.
    """
    # Try the supplied password, then the one a PREVIOUS studiobench run rotated to. Studio mints
    # a bootstrap password and demands it be changed; this function is what changes it, so the
    # second run against the same home is handed a bootstrap password that no longer exists and
    # gets a 401 whose message sends you to `reset-password`. Measured on the second run against
    # a home the first run had already set up. Trying both makes reruns and `--resume` work.
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
        )
    return auth


def seed_init_script(
    auth: StudioAuth,
    providers: list[ProviderSeed],
    extra_local_storage: Optional[dict] = None,
) -> str:
    """localStorage the SPA reads on its FIRST paint, so it boots already logged in and already
    holding the provider. The plaintext key is RSA-encrypted by the SPA per request against the
    server's published public key, so seeding it plainly here is the supported path."""
    payload = {
        "unsloth_auth_token": auth.access_token,
        "unsloth_refresh_token": auth.refresh_token,
        "unsloth_chat_external_providers": json.dumps([p.as_provider_entry() for p in providers]),
        "unsloth_chat_external_provider_keys": json.dumps(
            {p.id: p.api_key for p in providers if p.api_key}
        ),
        "unsloth_chat_connections_enabled": "true",
    }
    for k, v in (extra_local_storage or {}).items():
        payload[k] = v if isinstance(v, str) else json.dumps(v)
    return (
        "(() => { const seed = "
        + json.dumps(payload)
        + "; for (const k of Object.keys(seed)) { try { window.localStorage.setItem(k, seed[k]);"
        " } catch (e) {} } })();"
    )
