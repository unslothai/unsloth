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
import socket
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
    #: The commit `branch` RESOLVED TO, which is the build that was installed. `branch` is what
    #: the caller typed, and a branch or a movable tag is not a build. See
    #: `__main__.commit_problems`.
    commit: Optional[str] = None

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


def checkout_ref(repo: Path, ref: str) -> str:
    """Put `repo` on `ref`, whatever kind of ref it is. Returns the commit checked out.

    NOT `git clone --branch <ref>` AND NOT `reset --hard origin/<ref>`. `--branch` resolves against
    the remote's advertised branches and tags -- `git clone -h` calls it "checkout <branch> instead
    of the remote's HEAD" -- so a commit sha fails with `Remote branch <sha> not found in upstream
    origin` before anything is installed, and `origin/<sha>` is not a name that exists either.
    CONTRIBUTING-perf.md asks for exactly that ref: a change that has already merged is measured as
    `merge commit` against `merge commit^1`, and neither of those is a branch.

    So the ref is fetched and then resolved locally: `FETCH_HEAD` when the fetch could name it,
    `origin/<ref>` when it is a branch, and the ref itself when it is already an object here.
    """
    fetched = _run(["git", "fetch", "--tags", "origin", ref], cwd = repo, check = False)
    if fetched.returncode != 0:
        # A ref the remote will not serve by name (an old server, or `ref^1`, which is a local
        # expression rather than something to ask for). Fetch everything and resolve it here.
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


#: What `install.sh` is allowed. A multi-gigabyte download and build, documented in the README as
#: "budgeted at up to 45 minutes" and explicitly NOT part of a tier's wall clock. Named rather
#: than inlined because the run's watchdog has to add it to a deadline it would otherwise fire in
#: the middle of: see `watchdog_deadline_s`.
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
        # Cloned WITHOUT `--branch`, then moved onto the ref locally, so one code path serves a
        # branch, a tag and a commit sha instead of two that disagree about what a ref is.
        _run(["git", "clone", remote, str(repo)])
    # KEPT, not discarded. `checkout_ref` is the only place that knows which commit a ref names,
    # and the ref alone cannot tell a resumed run that `main` moved underneath it.
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


#: How long to keep looking for the launched server's pid. It appears once the server has forked
#: and exec'd, which is after `setsid -f` has already returned, so this is a poll rather than a
#: read. Named so a test can set it to zero instead of sleeping through it.
PID_DISCOVERY_TIMEOUT_S = 15.0


def _discover_pid(port: int, timeout_s: Optional[float] = None) -> Optional[int]:
    """The pid of the Studio serving `port`, or None. Polls, because it appears asynchronously.

    THE PROCESS WE LAUNCHED IS NOT THE PROCESS WE SPAWNED. `launch_studio` runs the server under
    `setsid -f`, which always forks and lets the parent exit without waiting, so the pid `Popen`
    returns belongs to a `setsid` that is gone by the time the server binds; the server itself is
    reparented into a session of its own and cannot be reached through our process group. `pgrep`
    is the only handle on it, and it can only be taken once the server exists.
    """
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
    """Is something already accepting connections on `port`?

    A plain connect, not a bind: the server we are about to launch is DETACHED and binds in a
    process of its own, so the only question this can answer is whether the port is already
    somebody's -- and if it is, it will not be ours.
    """
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
    # AN OCCUPIED PORT IS REFUSED BEFORE ANYTHING IS LAUNCHED, and this is the half of the
    # abandoned-server failure that no cleanup can reach. `--keep-studio` asks for a Studio to be
    # LEFT RUNNING on this port, so the next run's `unsloth studio -p <port>` finds one of our own
    # servers there and aborts rather than binding (`studio/backend/run.py`, `_resolve_port` with
    # `avoid_own_studio`); when the pid record is not readable it falls back to the NEXT port
    # instead. Either way nothing this run launched is on `port`.
    #
    # Everything downstream then agrees that the launch worked. `_discover_pid` pgreps for
    # `unsloth studio.*-p <port>` and finds the OLD process; `wait_for_healthz` takes its 200 from
    # it; and `authenticate` retries with `BENCH_PASSWORD`, which a previous studiobench run has
    # already rotated that Studio to, so the login succeeds as well. The run then measures the
    # build the PREVIOUS invocation installed while `run_meta` records the ref this one asked for,
    # and `stop_studio` kills the server the caller asked to keep. There is no reading anywhere
    # that says which build answered, so this is refused rather than reported.
    if port_is_busy(port):
        holder = _discover_pid(port, 0.0)
        raise RuntimeError(
            f"port {port} is already in use"
            + (f" by Studio pid {holder}" if holder else "")
            + ". A Studio launched here would abort or land on another port while this harness "
            "measured whatever is already answering. Stop it (`unsloth studio stop`, or the "
            "Studio a previous --keep-studio run left behind) or pass --port."
        )
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
    # BEFORE the health check, not after it. A Studio that starts and stays unhealthy used to raise
    # here with `install.pid` still None, and `stop_studio` has nothing to kill without it -- so the
    # detached server was left running on the requested port while the CLI unwound. It is not idle
    # there: the next attempt's `unsloth studio -p <port>` finds one of our own servers on the port
    # and aborts rather than binding (`run.py:_resolve_port`, `avoid_own_studio`), while
    # `wait_for_healthz` gets its 200 from the STALE process -- which by then may have finished
    # starting. That run measures the build the previous attempt installed and records the ref this
    # one asked for, which is the one failure this harness may never produce quietly.
    install.pid = _discover_pid(port)
    healthy = wait_for_healthz(install.base_url, healthz_timeout_s)
    if install.pid is None:
        # One more look: a server slow enough to miss the window above is exactly the one whose
        # health check just timed out, and it is the one that most needs to be terminated.
        install.pid = _discover_pid(port, 0.0)
    if not healthy:
        stop_studio(install)
        raise TimeoutError(f"Studio on :{port} did not pass /healthz within {healthz_timeout_s}s")
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
