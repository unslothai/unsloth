# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does /api/health ever publish a Mac capability verdict it is about to take back?

The unit tests around this invariant drive main._superseded_by_mlx_repair with a fake
clock and a stand-in worker. Nothing there boots a server, and nothing there runs on a
host where `import mlx.core` succeeds or fails for real. This does: it boots the real
`unsloth studio` on Apple Silicon, polls /api/health from the first reply the socket
gives, and judges the whole sequence rather than the last answer.

Three scenarios, one per boot (the self-heal is once per process, so they cannot share
one):

  real-mlx        A healthy Apple Silicon host. The verdict must settle to chat_only
                  false with device_backend "mlx", and no reply on the way there may
                  publish chat_only true. This is the regression that matters most:
                  the hold must not fire on a Mac that can train.

  no-mlx-settles  MLX made unimportable, self-heal opted out with
                  UNSLOTH_DISABLE_MLX_AUTOREPAIR=1. The verdict must settle promptly to
                  chat_only true / "mlx_unavailable". Nothing is coming to overturn it,
                  so a hold here would spin Train and Video for the whole session --
                  the failure mode the hold's bounds exist to prevent.

  no-mlx-repair   MLX made unimportable, self-heal left on. While the reinstall is
                  running, no reply may publish the verdict; once it has failed, the
                  verdict must settle exactly as in no-mlx-settles.

What is real here and what is not, stated plainly, because an integration test that
quietly asserts against its own mock is worth less than no test:

  * The host, the Apple Silicon check, hardware detection, the torch warm, the
    post-warm scheduler, the self-heal thread and its latch, the /api/health route and
    every window in main.py are all the shipped code, running in a real server process.
  * "MLX is missing" is simulated the same way tests/studio/test_hardware_dispatch_matrix.py
    simulates it, by making the import fail. That test patches inside its own process;
    this one has a server to boot, so the block is a sitecustomize on PYTHONPATH. It
    hides `mlx` from PathFinder, which is what an uninstalled package looks like: the
    spec lookup unsloth/__init__.py does returns None instead of raising, and every
    `import mlx.core` in the stack check raises ModuleNotFoundError.
  * In no-mlx-repair the installer is a stub `uv` first on PATH that records its argv,
    sleeps, and fails. utils.mlx_repair is untouched: it finds uv through
    shutil.which(), builds the real command, spawns the real subprocess and waits for
    it on the real worker thread. Only the third-party binary at the far end of that
    subprocess is ours, and it is a stub because a genuine `uv pip install mlx mlx-lm
    mlx-vlm --reinstall-package ...` on a CI runner is minutes of network for a result
    the shim would reject anyway. The stub's recorded argv is asserted, so a run where
    the self-heal never actually reached the installer fails instead of passing quietly.

Boot and the health wait are the repo's own scripts (.github/scripts/boot-studio-api-only.sh,
.github/scripts/wait-for-health.sh). The poll below is not a reimplementation of the
latter: wait-for-health.sh answers "is it up yet" and keeps no history, and every
assertion here is about replies it would have thrown away.

Apple Silicon only; invoked from .github/workflows/mlx-ci.yml. No GPU work, no weights,
and no network beyond localhost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOOT_SCRIPT = REPO / ".github" / "scripts" / "boot-studio-api-only.sh"
WAIT_FOR_HEALTH = REPO / ".github" / "scripts" / "wait-for-health.sh"
AUTH_DIR = Path(
    os.environ.get("STUDIO_AUTH_DIR", str(Path.home() / ".unsloth" / "studio" / "auth"))
)

# Fast enough that a verdict published for a moment cannot slip between two reads, slow
POLL_INTERVAL_S = float(os.environ.get("STUDIO_MAC_VERDICT_POLL_S", "0.25"))
# How long a verdict is watched after it settles.
STABLE_HOLD_S = float(os.environ.get("STUDIO_MAC_VERDICT_STABLE_S", "15"))
# Detection is stage one of the warm, so this covers a cold `import torch` and no more.
SETTLE_BUDGET_S = float(os.environ.get("STUDIO_MAC_VERDICT_SETTLE_S", "300"))
# The self-heal is scheduled after join_background_warm(), i.e.
# behind transformers, datasets and unsloth_zoo as well.
REPAIR_START_BUDGET_S = float(os.environ.get("STUDIO_MAC_VERDICT_REPAIR_START_S", "600"))
# How long the stub installer runs.
UV_SLEEP_S = float(os.environ.get("STUDIO_MAC_VERDICT_UV_SLEEP_S", "75"))
# breath. Comfortably past the 30s grace so a slow scheduler cannot blur the two.
# Where "the grace has certainly expired" starts, measured from the installer's first breath.
WORKER_PROOF_MARGIN_S = float(os.environ.get("STUDIO_MAC_VERDICT_WORKER_PROOF_S", "45"))
# Once the worker is gone nothing holds the verdict, so this is a settle, not a wait.
SETTLE_AFTER_REPAIR_S = float(os.environ.get("STUDIO_MAC_VERDICT_POST_REPAIR_S", "120"))

_failed: list[str] = []


def info(msg: str) -> None:
    print(f"[verdict] {msg}", flush = True)


def step(msg: str) -> None:
    print(f"[verdict] STEP {msg}", flush = True)


def fail(msg: str) -> None:
    """Record a failure and keep going, so one run reports every broken invariant."""
    print(f"[verdict] FAIL {msg}", flush = True)
    _failed.append(msg)


def ok(msg: str) -> None:
    print(f"[verdict]   OK {msg}", flush = True)




def _get(
    url: str,
    token: str | None = None,
    timeout: float = 15.0,
):
    """(status, parsed body) for a GET, or (None, None) when the socket is not there yet."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, method = "GET", headers = headers)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return None, None


def _post(
    url: str,
    body: dict,
    token: str | None = None,
    timeout: float = 30.0,
):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data = data, method = "POST", headers = headers)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        # Connection refused before the socket binds, and read timeouts while a C-extension import holds the GIL.
        return None, None




@dataclass(frozen = True)
class Sample:
    t: float
    token_sent: bool
    body: dict

    @property
    def authed(self) -> bool:
        """True when the bearer was accepted, read off the reply rather than assumed.

        /api/health answers an unusable bearer with the unauthenticated body instead of
        401, and that body carries no chat_only_reason -- which reads exactly like a
        training-capable Mac. `version` is authed-only and present in both the settled
        and the provisional authed shapes, so it is the marker that cannot lie.
        """
        return "version" in self.body

    @property
    def published(self) -> bool:
        """True when this reply is a measurement rather than a placeholder.

        `hardware_detecting` is exactly the marker config/hardware-verdict.ts reads to
        keep the UI provisional, so its absence is what "the frontend will act on this"
        means. chat_only is present either way and is the pre-detection default (True)
        while the marker is up, which is why its value alone proves nothing.
        """
        return "hardware_detecting" not in self.body

    @property
    def chat_only(self):
        return self.body.get("chat_only")

    @property
    def reason(self):
        return self.body.get("chat_only_reason")

    def state(self) -> tuple:
        return (
            self.published,
            self.authed,
            self.token_sent,
            self.chat_only,
            self.reason,
            self.body.get("device_type"),
            self.body.get("hardware_detection_deferred"),
            self.body.get("torch_warm_in_progress"),
        )


class HealthPoller(threading.Thread):
    """Reads /api/health in a loop and keeps every reply.

    Started before the boot, so the first reply the server is capable of giving is the
    first one recorded. A verdict that is wrong for two seconds and right afterwards is
    the bug being guarded against, and only the sequence can show it.
    """

    def __init__(self, base: str) -> None:
        super().__init__(daemon = True, name = "health-poller")
        self.base = base
        self.samples: list[Sample] = []
        self.token: str | None = None
        self.first_reply_at: float | None = None
        # Filled in once the boot script reports it, so the waits below can tell "still starting" from "already dead".
        self.server_pid: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.t0 = time.monotonic()

    def run(self) -> None:
        while not self._stop.is_set():
            token = self.token
            status, body = _get(f"{self.base}/api/health", token)
            if status == 200 and isinstance(body, dict):
                now = time.monotonic() - self.t0
                with self._lock:
                    if self.first_reply_at is None:
                        self.first_reply_at = now
                    self.samples.append(Sample(now, token is not None, body))
            elif status is not None and status != 200:
                # A non-200 from /api/health is not a verdict, but it is worth seeing:
                info(f"/api/health answered {status}")
            self._stop.wait(POLL_INTERVAL_S)

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> list[Sample]:
        with self._lock:
            return list(self.samples)

    def wait_for(self, predicate, budget: float, what: str) -> Sample | None:
        """First sample satisfying ``predicate``, or None once ``budget`` is spent.

        Gives up early on a server that has exited. A boot that dies on a missing
        dependency otherwise burns the whole budget answering nothing, and the timeout
        message blames the verdict for a process that was never there to publish one.
        """
        deadline = time.monotonic() + budget
        seen = 0
        while time.monotonic() < deadline:
            samples = self.snapshot()
            for sample in samples[seen:]:
                if predicate(sample):
                    return sample
            seen = len(samples)
            if self.server_pid and not _process_alive(self.server_pid):
                info(f"the server (pid {self.server_pid}) exited while waiting for {what}")
                return None
            time.sleep(POLL_INTERVAL_S)
        info(f"gave up waiting {budget:.0f}s for {what}")
        return None

    def report(self) -> None:
        """Print one line per state change. Thousands of identical replies say nothing;
        the transitions between them are the whole evidence."""
        samples = self.snapshot()
        first = f"t+{self.first_reply_at:.2f}s" if self.first_reply_at is not None else "never"
        info(f"{len(samples)} health replies recorded, first at {first}")
        previous = None
        for sample in samples:
            state = sample.state()
            if state == previous:
                continue
            previous = state
            info(
                f"  t+{sample.t:7.2f}s "
                f"{'PUBLISHED ' if sample.published else 'provisional'} "
                f"{'authed  ' if sample.authed else 'unauthed'} "
                f"chat_only={sample.chat_only} reason={sample.reason} "
                f"device_type={sample.body.get('device_type')} "
                f"deferred={sample.body.get('hardware_detection_deferred')} "
                f"warm={sample.body.get('torch_warm_in_progress')}"
            )


class TokenGetter(threading.Thread):
    """Logs in with the bootstrap password and hands the poller a bearer.

    device_type and chat_only_reason are authed-only fields, so without this the run can
    see THAT a verdict was published but not WHICH one. Runs concurrently with the poll
    rather than before it: waiting for a login would spend the opening seconds of the
    very window under test, and an unauthed reply already reveals a published verdict.
    """

    def __init__(self, base: str, poller: HealthPoller) -> None:
        super().__init__(daemon = True, name = "token-getter")
        self.base = base
        self.poller = poller
        self._stop = threading.Event()
        self.error: str | None = None

    def _login(self, password: str) -> str | None:
        status, body = _post(
            f"{self.base}/api/auth/login", {"username": "unsloth", "password": password}
        )
        if status == 200 and isinstance(body, dict):
            return body.get("access_token")
        return None

    def run(self) -> None:
        pw_file = AUTH_DIR / ".bootstrap_password"
        deadline = time.monotonic() + 240
        password = None
        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                password = pw_file.read_text(encoding = "utf-8").strip()
            except OSError:
                self._stop.wait(0.5)
                continue
            if password:
                break
            self._stop.wait(0.5)
        if not password:
            self.error = f"no bootstrap password appeared at {pw_file}"
            return
        token = None
        while time.monotonic() < deadline and not self._stop.is_set():
            token = self._login(password)
            if token:
                break
            self._stop.wait(1.0)
        if not token:
            self.error = "bootstrap login never returned a token"
            return
        # The seeded password is must-change, and its token only opens /api/auth/change-password: everything else,
        # /api/health's authed half included, sees no subject and silently answers with the unauthenticated body.
        # Read chat_only_reason off that and it is always None, which looks exactly like a training-capable Mac.
        # So rotate first, the same way studio_api_smoke.py does.
        rotated = f"{password}-Rotated1!"
        status, _ = _post(
            f"{self.base}/api/auth/change-password",
            {"current_password": password, "new_password": rotated},
            token = token,
        )
        if status == 200:
            token = self._login(rotated) or token
        self.poller.token = token

    def stop(self) -> None:
        self._stop.set()




def _write_mlx_block_shim(directory: Path) -> Path:
    """A sitecustomize that hides mlx from the import system, for the server we boot.

    tests/studio/test_hardware_dispatch_matrix.py does this with monkeypatch inside one
    process. A booted server is a different process (and spawns more), so the block has
    to arrive through PYTHONPATH, which `site` reads for every interpreter that starts
    under it.

    PathFinder rather than a meta_path finder that raises: unsloth/__init__.py decides
    _IS_MLX with importlib.util.find_spec("mlx") and does not guard it, so a finder that
    raised would turn `import unsloth` into a crash instead of a Mac without MLX.
    Returning None from the path finder is what an uninstalled package genuinely looks
    like -- find_spec answers None, and `import mlx.core` raises ModuleNotFoundError,
    which is the ImportError both _has_mlx() and the stack check already handle.
    """
    directory.mkdir(parents = True, exist_ok = True)
    shim = directory / "sitecustomize.py"
    shim.write_text(
        "# Generated by tests/studio/mac_capability_verdict_smoke.py. Simulates a Mac\n"
        "# whose MLX stack is missing, on a runner where it is installed.\n"
        "import importlib.machinery\n"
        "import sys\n"
        "\n"
        "_real_find_spec = importlib.machinery.PathFinder.find_spec\n"
        "\n"
        "\n"
        "def _find_spec(fullname, path = None, target = None):\n"
        "    if fullname == 'mlx' or fullname.startswith('mlx.'):\n"
        "        return None\n"
        "    return _real_find_spec(fullname, path, target)\n"
        "\n"
        "\n"
        "importlib.machinery.PathFinder.find_spec = staticmethod(_find_spec)\n"
        "for _name in [n for n in sys.modules if n == 'mlx' or n.startswith('mlx.')]:\n"
        "    del sys.modules[_name]\n",
        encoding = "utf-8",
    )
    return shim


def _write_stub_uv(directory: Path, marker: Path) -> Path:
    """A `uv` that records how it was called, takes its time, and fails.

    utils.mlx_repair reaches its installer through shutil.which("uv"), so putting this
    first on PATH is enough to drive the real self-heal without a real install. It fails
    (exit 1) because a success would re-run the stack check, which the shim above still
    refuses -- so the verdict settles chat-only either way, and failing gets there
    without pretending an install happened.

    The marker is written twice, atomically, so a reader can tell "installing" from
    "finished" without racing a partial file. It carries argv because a run where the
    self-heal never reached the installer would otherwise pass this file's assertions
    while proving nothing.
    """
    directory.mkdir(parents = True, exist_ok = True)
    stub = directory / "uv"
    stub.write_text(
        f"#!{sys.executable}\n"
        "# Generated by tests/studio/mac_capability_verdict_smoke.py.\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        "\n"
        f"MARKER = {str(marker)!r}\n"
        f"SLEEP = {UV_SLEEP_S!r}\n"
        "\n"
        "\n"
        "def write(record):\n"
        "    tmp = MARKER + '.tmp'\n"
        "    with open(tmp, 'w', encoding = 'utf-8') as handle:\n"
        "        json.dump(record, handle)\n"
        "    os.replace(tmp, MARKER)\n"
        "\n"
        "\n"
        "record = {'argv': sys.argv[1:], 'started': time.time()}\n"
        "write(record)\n"
        "time.sleep(SLEEP)\n"
        "record['ended'] = time.time()\n"
        "write(record)\n"
        "sys.exit(1)\n",
        encoding = "utf-8",
    )
    stub.chmod(0o755)
    return stub


def _read_marker(marker: Path) -> dict | None:
    try:
        return json.loads(marker.read_text(encoding = "utf-8"))
    except Exception:
        return None




def boot(port: int, log: Path, env: dict) -> int | None:
    """Run the repo's boot script and return the server pid.

    --api-only because mlx-ci.yml boots on a bare `pip install -e .` with no built
    studio/frontend/dist, and without the flag the server prints "Unsloth frontend build
    not found" and exits before it binds. The workflows that run install.sh do have a
    dist and deliberately serve it, which is why the flag is theirs to pass, not the
    script's to assume.

    GITHUB_ENV is dropped for this call on purpose: with it set the script exports the
    pid into the workflow environment instead of printing it, and this file needs the pid
    in hand to stop the server before the next scenario boots on the next port.
    """
    child_env = dict(env)
    child_env.pop("GITHUB_ENV", None)
    log.parent.mkdir(parents = True, exist_ok = True)
    result = subprocess.run(
        ["bash", str(BOOT_SCRIPT), "--port", str(port), "--log", str(log), "--api-only"],
        env = child_env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    # Flushed here, not left to the next print:
    sys.stdout.flush()
    sys.stderr.flush()
    match = re.search(r"pid (\d+)", result.stdout)
    return int(match.group(1)) if match else None


def wait_for_health(port: int, log: Path) -> bool:
    """The repo's own health wait, used for what this file has no answer for: when the
    server never comes up at all, it tails that server's log and says so."""
    result = subprocess.run(
        [
            "bash",
            str(WAIT_FOR_HEALTH),
            "--port",
            str(port),
            "--log",
            str(log),
            "--tmp",
            str(log.with_suffix(".health.json")),
        ],
        capture_output = True,
        text = True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode == 0


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.25)
    try:
        os.kill(pid, 9)
    except OSError:
        pass


def tail_log(log: Path, lines: int = 80) -> None:
    try:
        content = log.read_text(encoding = "utf-8", errors = "replace").splitlines()
    except OSError:
        info(f"no server log at {log}")
        return
    info(f"last {min(lines, len(content))} lines of {log}:")
    for line in content[-lines:]:
        print(f"  | {line}", flush = True)




def assert_watched_the_window(poller: HealthPoller) -> None:
    """The poll has to have produced replies, or nothing below was actually tested."""
    samples = poller.snapshot()
    if not samples:
        fail("no /api/health reply was ever recorded; nothing below was actually tested")
        return
    provisional = [s for s in samples if not s.published]
    if provisional:
        ok(f"{len(provisional)} provisional replies seen before the verdict settled")
        return
    # Not a failure. Detection is stage one of the warm and the socket binds right after the lifespan starts it, so on
    info(
        f"no provisional reply was observed; detection settled within "
        f"{poller.first_reply_at:.2f}s, before the socket answered"
    )


def assert_never_published(poller: HealthPoller, predicate, what: str) -> None:
    for sample in poller.snapshot():
        if sample.published and predicate(sample):
            fail(
                f"{what} was published at t+{sample.t:.2f}s "
                f"(chat_only={sample.chat_only} reason={sample.reason} "
                f"authed={sample.authed}); the frontend treats this as measured"
            )
            return
    ok(f"{what} was never published")


def assert_settled_and_stable(poller: HealthPoller, chat_only: bool, reason, label: str) -> None:
    """Watch a settled verdict for a while and require it to stay settled and unchanged."""
    step(f"holding {STABLE_HOLD_S:.0f}s to confirm the {label} verdict stays settled")
    mark = len(poller.snapshot())
    time.sleep(STABLE_HOLD_S)
    tail = poller.snapshot()[mark:]
    if not tail:
        fail(f"the poller stopped producing replies while holding the {label} verdict")
        return
    for sample in tail:
        if not sample.published:
            fail(
                f"the {label} verdict went back to provisional at t+{sample.t:.2f}s; "
                "the rows would spin again after settling"
            )
            return
        if sample.chat_only != chat_only:
            fail(f"chat_only flipped to {sample.chat_only} at t+{sample.t:.2f}s")
            return
        if sample.authed and sample.reason != reason:
            fail(f"chat_only_reason became {sample.reason!r} at t+{sample.t:.2f}s")
            return
    ok(f"the {label} verdict held for {STABLE_HOLD_S:.0f}s across {len(tail)} replies")


def first_authed_published(
    poller: HealthPoller, getter: TokenGetter, budget: float
) -> Sample | None:
    """The first published reply that also carries the authed-only fields.

    chat_only_reason and device_type are what the greyed-row tooltip is built from, so a
    scenario that only ever saw an unauthenticated published reply has not checked the
    thing the user actually reads.
    """
    sample = poller.wait_for(
        lambda s: s.published and s.authed,
        budget,
        "a published reply on an authenticated read",
    )
    if sample is None:
        fail(f"no authenticated published reply ({getter.error or 'login had not landed yet'})")
    return sample


def assert_settled_verdict(sample: Sample, chat_only: bool, reason, label: str) -> bool:
    """The fields the sidebar renders a row from, checked together."""
    before = len(_failed)
    if sample.chat_only is not chat_only:
        fail(f"the settled {label} verdict is chat_only={sample.chat_only}, expected {chat_only}")
    if sample.reason != reason:
        fail(f"chat_only_reason is {sample.reason!r}, expected {reason!r}")
    if sample.body.get("device_type") != "mac":
        fail(f"device_type is {sample.body.get('device_type')!r}, expected 'mac'")
    if sample.body.get("hardware_detection_deferred"):
        fail(
            "the settled reply is marked detection-deferred; env.ts reads that as "
            "'nothing will ever measure this' and stores the conservative chat_only"
        )
    return len(_failed) == before


@contextmanager
def booted(port: int, log: Path, env: dict):
    """Boot an Unsloth with ``env``, polling /api/health from before it can answer."""
    base = f"http://127.0.0.1:{port}"
    poller = HealthPoller(base)
    poller.start()
    pid = None
    getter = TokenGetter(base, poller)
    try:
        pid = boot(port, log, env)
        poller.server_pid = pid
        # spend its retries on a credential this server has never heard of.
        # Started only now: the boot script wipes the auth directory, so a login thread running before it would read
        getter.start()
        yield poller, getter
    finally:
        getter.stop()
        poller.stop()
        poller.report()
        stop(pid)




def scenario_real_mlx(port: int, log: Path) -> None:
    """A healthy Apple Silicon host must never look chat-only, not even briefly."""
    step("scenario real-mlx: the verdict on a Mac whose MLX stack works")

    # Preflight in this process, with the product's own criterion.
    sys.path.insert(0, str(REPO / "studio" / "backend"))
    from utils.mlx_repair import mlx_stack_available  # noqa: PLC0415

    if not mlx_stack_available():
        fail(
            "the MLX stack on this host is missing or below unsloth-zoo's minimums, so "
            "the healthy-Mac scenario cannot run here; fix the runner's install rather "
            "than reading this run as a pass"
        )
        return

    with booted(port, log, dict(os.environ)) as (poller, getter):
        settled = poller.wait_for(lambda s: s.published, SETTLE_BUDGET_S, "the verdict to settle")
        if settled is None:
            wait_for_health(port, log)
            tail_log(log)
            fail("the verdict never settled on a host with a working MLX stack")
            return
        info(f"verdict settled at t+{settled.t:.2f}s")

        assert_watched_the_window(poller)
        # The reported bug in the exact shape it reached users:
        assert_never_published(
            poller,
            lambda s: s.chat_only is True,
            "a chat-only verdict on a training-capable Mac",
        )

        authed = first_authed_published(poller, getter, 120)
        if authed is not None and assert_settled_verdict(authed, False, None, "training-capable"):
            ok("the settled verdict is chat_only=false with device_type mac")

        # device_type names the platform;
        if poller.token:
            status, body = _get(f"http://127.0.0.1:{port}/api/system", poller.token, timeout = 60)
            if status != 200 or not isinstance(body, dict):
                fail(f"/api/system returned {status}")
            elif body.get("device_backend") != "mlx":
                fail(
                    f"/api/system reports device_backend={body.get('device_backend')!r}; "
                    "detection did not select the MLX backend on Apple Silicon"
                )
            else:
                ok("/api/system reports device_backend mlx")

        assert_settled_and_stable(poller, False, None, "training-capable")


def scenario_no_mlx_settles(port: int, log: Path, work: Path) -> None:
    """A Mac nothing is coming to repair must still get a final answer, and quickly."""
    step("scenario no-mlx-settles: MLX blocked, self-heal opted out")

    shim_dir = work / "mlx_block"
    _write_mlx_block_shim(shim_dir)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shim_dir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["UNSLOTH_DISABLE_MLX_AUTOREPAIR"] = "1"

    with booted(port, log, env) as (poller, getter):
        if not wait_for_health(port, log):
            fail("the server never became healthy")
            return
        settled = poller.wait_for(lambda s: s.published, SETTLE_BUDGET_S, "the verdict to settle")
        if settled is None:
            tail_log(log)
            fail(
                "the verdict never settled with the self-heal opted out; Train and Video "
                "would spin for the whole session on a Mac that has no repair coming"
            )
            return
        info(f"verdict settled at t+{settled.t:.2f}s")

        assert_watched_the_window(poller)
        if settled.chat_only is not True:
            fail(
                f"the settled verdict is chat_only={settled.chat_only} with mlx blocked; "
                "the import shim did not take, so this scenario tested nothing"
            )
        authed = first_authed_published(poller, getter, 120)
        if authed is not None and assert_settled_verdict(
            authed, True, "mlx_unavailable", "chat-only"
        ):
            ok("chat_only true / mlx_unavailable, settled and explained")
        assert_settled_and_stable(poller, True, "mlx_unavailable", "chat-only")


def scenario_no_mlx_repair(port: int, log: Path, work: Path) -> None:
    """While the self-heal is installing, the verdict it would overturn stays unpublished."""
    step("scenario no-mlx-repair: MLX blocked, self-heal running")

    shim_dir = work / "mlx_block"
    _write_mlx_block_shim(shim_dir)
    stub_dir = work / "stub_bin"
    marker = work / "uv_invocation.json"
    marker.unlink(missing_ok = True)
    _write_stub_uv(stub_dir, marker)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shim_dir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PATH"] = os.pathsep.join([str(stub_dir), env.get("PATH", "")])
    env.pop("UNSLOTH_DISABLE_MLX_AUTOREPAIR", None)

    with booted(port, log, env) as (poller, getter):
        # The window under test opens once detection settles, not at the first byte, so gating on the repo's health
        if not wait_for_health(port, log):
            fail("the server never became healthy")
            return

        step(f"waiting up to {REPAIR_START_BUDGET_S:.0f}s for the self-heal to reach the installer")
        deadline = time.monotonic() + REPAIR_START_BUDGET_S
        record = None
        while time.monotonic() < deadline:
            record = _read_marker(marker)
            if record and "started" in record:
                break
            time.sleep(0.5)
        if not record or "started" not in record:
            tail_log(log)
            fail(
                "the self-heal never invoked the installer, so nothing here exercised the "
                "in-flight hold; check that the post-warm scheduler ran"
            )
            return
        # Anchor the installer's clock to the poller's, so the two timelines can be compared.
        uv_started = record["started"] - (time.time() - time.monotonic()) - poller.t0
        argv = " ".join(record.get("argv", []))
        info(f"installer invoked at t+{uv_started:.2f}s: uv {argv[:300]}")
        for token in ("pip", "install", "mlx", "mlx-lm", "mlx-vlm"):
            if token not in argv:
                fail(f"the installer was invoked without {token!r}; this is not the MLX self-heal")

        step(f"waiting for the installer to finish ({UV_SLEEP_S:.0f}s)")
        deadline = time.monotonic() + UV_SLEEP_S + 120
        while time.monotonic() < deadline:
            record = _read_marker(marker)
            if record and "ended" in record:
                break
            time.sleep(0.5)
        if not record or "ended" not in record:
            fail("the stub installer never finished; the run below would prove nothing")
            return
        uv_ended = record["ended"] - (time.time() - time.monotonic()) - poller.t0
        info(f"installer finished at t+{uv_ended:.2f}s")

        in_flight = [s for s in poller.snapshot() if uv_started <= s.t <= uv_ended]
        if not in_flight:
            fail("no health reply landed while the installer was running")
        else:
            published = [s for s in in_flight if s.published]
            if published:
                first = published[0]
                fail(
                    f"the verdict was published at t+{first.t:.2f}s while the self-heal was "
                    f"still installing (chat_only={first.chat_only} reason={first.reason}); "
                    "this is the greyed-out Train and Video the repair then takes back"
                )
            else:
                ok(f"{len(in_flight)} replies during the install, none published")

            # The pre-start window would have expired 30s after the warm ended, and the warm ends before the scheduler
            # Must clear main._MLX_PRESTART_GRACE_AFTER_WARM_S with room to spare: samples taken after that grace would
            # have expired are the only ones that prove the LIVE WORKER is holding the verdict rather than the pre-start
            # window.
            proof = [s for s in in_flight if s.t >= uv_started + WORKER_PROOF_MARGIN_S]
            if not proof:
                fail(
                    f"no reply landed more than {WORKER_PROOF_MARGIN_S:.0f}s into the install, "
                    "so the hold observed above cannot be told apart from the pre-start grace"
                )
            elif any(s.published for s in proof):
                fail("the verdict was published while a live repair worker was still installing")
            else:
                ok(
                    f"{len(proof)} replies held back past the pre-start grace, "
                    "so a live worker is what is holding them"
                )

        step("the install has failed; the verdict must now settle on its own")
        settled = poller.wait_for(
            lambda s: s.published, SETTLE_AFTER_REPAIR_S, "the verdict to settle"
        )
        if settled is None:
            tail_log(log)
            fail(
                "the verdict never settled after the self-heal finished; the hold is "
                "unbounded, which spins Train and Video for the rest of the session"
            )
            return
        info(
            f"verdict settled at t+{settled.t:.2f}s, "
            f"{settled.t - uv_ended:.2f}s after the install gave up"
        )
        authed = first_authed_published(poller, getter, 120)
        if authed is not None:
            assert_settled_verdict(authed, True, "mlx_unavailable", "post-repair chat-only")
        assert_settled_and_stable(poller, True, "mlx_unavailable", "post-repair chat-only")


SCENARIOS = {
    "real-mlx": scenario_real_mlx,
    "no-mlx-settles": scenario_no_mlx_settles,
    "no-mlx-repair": scenario_no_mlx_repair,
}


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("scenario", choices = sorted(SCENARIOS))
    parser.add_argument("--port", type = int, required = True)
    parser.add_argument("--log", default = None, help = "server log path")
    parser.add_argument(
        "--workdir",
        default = None,
        help = "where the generated import shim and stub installer are written",
    )
    args = parser.parse_args()

    log = Path(args.log) if args.log else Path("logs") / f"studio_verdict_{args.scenario}.log"
    work = Path(args.workdir) if args.workdir else Path("logs") / f"verdict_{args.scenario}"
    work.mkdir(parents = True, exist_ok = True)

    if args.scenario == "real-mlx":
        scenario_real_mlx(args.port, log)
    else:
        SCENARIOS[args.scenario](args.port, log, work)

    if _failed:
        print(f"[verdict] {len(_failed)} FAILURE(S)", flush = True)
        for message in _failed:
            print(f"[verdict]   - {message}", flush = True)
        return 1
    print("[verdict] PASS", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
