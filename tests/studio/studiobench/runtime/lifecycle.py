# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install, launch and authenticate a real Unsloth. VENDORED from `studio_test_kit`.

Vendored on purpose. The shipped artifact is a single `studiobench.pyz` a tester runs on a machine
that has an Unsloth and nothing else, so it cannot import a module that lives elsewhere in this
repository. The logic is `studio_test_kit.lifecycle` plus `studio_test_kit.auth`, with two
deliberate changes:

- **stdlib only.** `studio_test_kit.auth` uses `httpx`; this uses `urllib.request`, so `--doctor`
  and `--attach` work on a machine with nothing pip-installed but Playwright.
- **The password-change gate is handled.** A current Unsloth mints a bootstrap password and sets
  `must_change_password`, and until it is cleared EVERY authenticated route answers
  `403 Password change required` while `/healthz` answers 200 and login itself succeeds. That
  failure is silent one request too late, and it is what stops a thread from being seeded at all.
"""

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
    #: The commit `branch` RESOLVED TO, which is the build that was installed. `branch` is what
    #: the caller typed, and a branch or a movable tag is not a build. See
    #: `__main__.commit_problems`.
    commit: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


#: What the backend gives an access token: `auth/authentication.py`, ACCESS_TOKEN_EXPIRE_MINUTES.
#: Only a FALLBACK. The expiry actually enforced is the `exp` claim in the token this run was
#: handed, which is read from the token itself; this number is what to assume when the server
#: hands back something that is not a readable JWT.
ACCESS_TOKEN_TTL_S = 60 * 60
#: How far ahead of `exp` a token is replaced. Seeding a 1M-token thread is ONE request with a
#: 900 second timeout, so a token that is merely valid at the moment the request is written is not
#: good enough: it has to still be valid when the server finishes reading the body.
TOKEN_REFRESH_MARGIN_S = 15 * 60


def jwt_expiry(token: str) -> Optional[float]:
    """The `exp` claim of a JWT, in unix seconds, WITHOUT verifying anything.

    This is not authentication, it is a clock: the harness holds a token the server issued and
    needs to know when the server will stop accepting it. Verification is the server's job and it
    does it on every request. Anything unreadable returns None and the caller falls back to
    `ACCESS_TOKEN_TTL_S`.
    """
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
    """The harness's own credentials, WHICH OUTLIVE THE TOKEN THEY WERE HANDED.

    THE RUN IS LONGER THAN THE TOKEN. An access token is good for one hour
    (`ACCESS_TOKEN_EXPIRE_MINUTES = 60`) and this harness authenticated once per arm, at setup,
    before the first install. A standard A/B at four repetitions is 24 films of 243 seconds --
    97 minutes of cells alone, before seeding, calibration and two installs -- so the token the
    seeder holds expires PART WAY THROUGH, and every request after that answers 401: the next
    `create_thread` fails and the cell dies. In the browser it is worse than an error, because the
    SPA's own 401 path clears its tokens and navigates to the login route, which reaches Playwright
    as `Execution context was destroyed, most likely because of a navigation` -- the exact shape of
    a flake, arriving on a schedule.

    So the token is REPLACED BEFORE IT EXPIRES rather than after it fails. `token()` is what every
    authenticated request asks for and it re-authenticates whenever `exp` is inside
    `TOKEN_REFRESH_MARGIN_S`; `auth_request_json` additionally recovers from a 401 that arrives
    anyway, because a clock skew between this process and the server is exactly the case a
    deadline computed here cannot see.

    RE-LOGIN, NOT REFRESH, and the difference matters. `POST /api/auth/refresh` is SINGLE USE
    (`storage.consume_refresh_token`) and the refresh token this object holds is the same one the
    page was seeded with. Spending it here would invalidate the copy in the SPA's localStorage and
    log the page out to fix the harness -- trading this defect for itself. A password login mints a
    fresh pair and touches nothing the page owns.
    """

    access_token: str
    refresh_token: str
    base_url: str
    username: str
    password: str
    #: Unix seconds, from the token's own `exp`. None means "unknown", treated as `ACCESS_TOKEN_TTL_S`
    #: from the moment this object was built.
    expires_at: Optional[float] = None
    #: Called with `self` after the token is replaced. The browser context seeds its localStorage
    #: from a SNAPSHOT of these values, so whoever owns that context re-seeds it here.
    on_rotate: Optional[Callable[["StudioAuth"], None]] = None
    rotations: int = field(default = 0, init = False)
    #: Turned off the first time a FRESH token still reads as expiring, which means the `exp` this
    #: process reads and the server's own clock do not agree. See `rotate`.
    proactive: bool = field(default = True, init = False)
    #: The last `on_rotate` failure, kept rather than raised. See `rotate`.
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
        """Log in again and adopt the new pair. Raises if the server will not have us.

        The password is the one THIS harness rotated to (`authenticate` clears the
        password-change gate up front), so there is a credential to log in with for as long as the
        run lasts.

        A FRESH TOKEN THAT IS ALREADY INSIDE THE MARGIN TURNS THE PROACTIVE HALF OFF, and this is
        the guard against the one way "refresh before `exp`" can run away. `needs_refresh` compares
        the server's `exp` against THIS PROCESS'S clock, and the two are not required to agree: a
        Unsloth running 45 minutes behind, or a deployment that shortens
        `ACCESS_TOKEN_EXPIRE_MINUTES` below the margin, makes every token ever issued look like it
        is about to expire. Without this, every single request would log in again and append
        another init script to the browser context for the rest of the run. So the condition is
        tested against a token known to be one second old: if even that one is inside the margin,
        the margin is not usable here and the token is left to `auth_request_json`'s 401 recovery,
        which asks the server rather than the clock.

        THE HOOK CANNOT FAIL THE CREDENTIAL. `on_rotate` re-seeds a Playwright context, which can
        throw for reasons that have nothing to do with authentication -- a closed context, a page
        that crashed. The token has already been replaced by then and the caller's request must go
        out; the failure is recorded on `hook_error` instead of being raised through a function
        whose job was to hold a credential.
        """
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
    """`request_json` with credentials that survive a long run: refreshed BEFORE `exp`, and
    re-minted once more if the server rejects the token anyway.

    Both halves are needed. The proactive half is what keeps a 900 second seeding PUT from dying
    half way through a request that was valid when it started. The reactive half covers what this
    process cannot compute: a clock offset against the server, an Unsloth restarted underneath the
    run, or a token invalidated by something else. One retry only -- a 401 that survives a fresh
    login is a real refusal and must be raised, not looped on.

    The token is fetched OUTSIDE the `try`, so a 401 raised by the login inside `token()` is not
    mistaken for a 401 from this request and answered with a second login. The backend locks an
    account out after five failures in a minute (`routes/auth.py`, `_LOGIN_MAX_FAILS`), so a wrong
    credential retried at double rate reaches the lockout twice as fast and the run then dies on a
    429 that says nothing about the password.
    """
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
    existing = auth_request_json(auth, f"{base_url.rstrip('/')}/api/providers/") or []
    for row in existing:
        # Idempotent across runs. Every run binds a NEW ephemeral pacer port, so a stale entry
        # from a previous run points at a port nothing is listening on, and leaving it there gives
        # the picker two identically named models of which one is dead.
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
    """The pid of the Unsloth serving `port`, or None. Polls, because it appears asynchronously.

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
    # abandoned-server failure that no cleanup can reach. `--keep-studio` asks for an Unsloth to be
    # LEFT RUNNING on this port, so the next run's `unsloth studio -p <port>` finds one of our own
    # servers there and aborts rather than binding (`studio/backend/run.py`, `_resolve_port` with
    # `avoid_own_studio`); when the pid record is not readable it falls back to the NEXT port
    # instead. Either way nothing this run launched is on `port`.
    #
    # Everything downstream then agrees that the launch worked. `_discover_pid` pgreps for
    # `unsloth studio.*-p <port>` and finds the OLD process; `wait_for_healthz` takes its 200 from
    # it; and `authenticate` retries with `BENCH_PASSWORD`, which a previous studiobench run has
    # already rotated that Unsloth to, so the login succeeds as well. The run then measures the
    # build the PREVIOUS invocation installed while `run_meta` records the ref this one asked for,
    # and `stop_studio` kills the server the caller asked to keep. There is no reading anywhere
    # that says which build answered, so this is refused rather than reported.
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
    # BEFORE the health check, not after it. An Unsloth that starts and stays unhealthy used to raise
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
        raise TimeoutError(f"Unsloth on :{port} did not pass /healthz within {healthz_timeout_s}s")
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
        expires_at = jwt_expiry(body["access_token"]),
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
    # Try the supplied password, then the one a PREVIOUS studiobench run rotated to. Unsloth mints
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
            expires_at = jwt_expiry(body["access_token"]),
        )
    return auth


def seed_init_script(
    auth: StudioAuth,
    providers: list[ProviderSeed],
    extra_local_storage: Optional[dict] = None,
) -> str:
    """localStorage the SPA reads on its FIRST paint, so it boots already logged in and already
    holding the provider. The plaintext key is RSA-encrypted by the SPA per request against the
    server's published public key, so seeding it plainly here is the supported path.

    THE REFRESH TOKEN GOES UNDER THE KEY THE APP READS, which is
    `unsloth_auth_refresh_token` (`features/auth/session.ts`, `AUTH_REFRESH_TOKEN_KEY`) and not
    `unsloth_refresh_token`, which nothing in the frontend has ever read. Seeded under the wrong
    name the page boots with an access token and NO way to renew it, so one hour in --
    `ACCESS_TOKEN_EXPIRE_MINUTES = 60`, shorter than a standard A/B -- the first request to answer
    401 takes `authFetch` down the branch that clears the tokens and navigates to the login route.
    Playwright reports that as `Execution context was destroyed, most likely because of a
    navigation`, which reads as a flake and is not one: it is the clock. With the right key the
    SPA rotates its own pair through `POST /api/auth/refresh` and the film carries on.

    That endpoint is SINGLE USE, which is why the harness re-authenticates by password instead of
    spending this token itself -- see `StudioAuth`. This copy belongs to the page.

    AND THE AUTH KEYS ARE WRITTEN ONLY BY THE FRESHEST WRITER. An init script is a snapshot that
    re-runs on EVERY navigation, and there is one navigation per cell, so a script carrying the
    token this run started with would keep putting it back over whatever the SPA had rotated to.
    The harness re-seeds after `StudioAuth.rotate`, but Playwright says outright that "the order of
    evaluation of multiple scripts installed via browser_context.add_init_script() and
    page.add_init_script() is not defined", so "the newest one was added last" decides nothing.
    Each script therefore compares its own token's `exp` against the one already in storage and
    writes only if it is carrying the later one. That converges on the freshest token whatever
    order they run in, which is the only property worth having here.
    """
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
    # `exp` out of an unverified JWT payload, base64url with the padding this token does not carry.
    # Anything unreadable scores 0, so a token nobody can date never displaces one that can be.
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
