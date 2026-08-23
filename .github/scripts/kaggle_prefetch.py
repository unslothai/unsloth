# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The model prefetch that both Kaggle kernels paste into a generated cell.

A Kaggle GPU session is 2xT4 and the kernel keeps both busy training, but
downloading a model is CPU and network work that holds a card idle for its
whole duration. This module is the body of a lane that does that downloading
EARLY, beside the training legs, so the leg that needs the model finds it
already on disk.

Why a module and not a copy in each builder: `kaggle_t4_ci` prefetches the leg
models into the Kaggle image's default cache, and `kaggle_studio_ci` prefetches
Studio's two models into Studio's own private ``HF_HOME``. Same retry policy,
same reporting, two different cache roots. One copy that takes the root as an
argument is the only version of that which stays in agreement with itself.

Load it BY PATH (``importlib.util.spec_from_file_location``), never with a
plain ``import``. Both script directories already ship a ``build_kernel.py``
and a ``report.py``, the test suite puts both on ``sys.path``, and a plain
import therefore resolves to whichever reached ``sys.modules`` first -- which
is decided by test order rather than by intent. That collision has been paid
for here once already: one ``sys.path.insert`` added for a single test took
nine unrelated tests down with it.

WHAT THIS IS NOT: it is not a correctness mechanism. Every caller must treat a
failed prefetch as a no-op, because the payload that wants the model downloads
it for itself exactly as it did before this existed. A prefetch that fails the
kernel would be a new way to go red for something that is not under test.
"""

from __future__ import annotations

# The sentinel the driver and the reporters grep for. One record per repo, on
# its own line, so `kernel.log` ALONE measures the download -- which is the
# number the whole schedule is built around and the one thing no artifact has
# ever separated from weight-load time.
PREFETCH_SENTINEL = "KAGGLE_CI_PREFETCH"


def _normalise(repos):
    """``["a", ("b", ["*.gguf"])]`` -> ``[("a", None), ("b", ["*.gguf"])]``.

    A bare string means the WHOLE repo, which is right for a small model whose
    every file gets loaded and wrong for anything with variants. Run
    32667451396 fetched 69.1 GB of ``Qwen3.5-2B-GGUF`` -- every quant in the
    repo -- so that Studio could load one UD-Q4_K_XL file, and 55.1 GB of a
    checkpoint that was never opened at all. On a 4-core Kaggle box that is not
    just wasted bandwidth: it is CPU stolen from the payloads the prefetch
    exists to speed up, and it pushed the Studio install from 258s to 673.5s.
    """
    out = []
    for entry in repos:
        if isinstance(entry, str):
            out.append((entry, None))
            continue
        repo, patterns = entry
        out.append((repo, list(patterns) if patterns else None))
    return out


def prefetch_cell(
    repos: list,
    *,
    hf_home: str | None = None,
    attempt_timeout: int = 900,
    total_timeout: int = 1800,
) -> str:
    """Source for a cell (or a driver thread) that warms ``repos``, in order.

    ``repos`` is ordered and the order is load bearing: the caller puts the
    repo with the longest lead time first, because a prefetch only pays for
    the work it finishes BEFORE the payload that wants it starts.

    ``hf_home`` of None means "do not touch HF_HOME", which is what the leg
    prefetch needs -- the legs read the Kaggle image's default cache and the
    entire point is to land in the cache they read. Setting it to a private
    directory there would produce a perfectly healthy prefetch that no payload
    can see, a full 12 GB of work thrown away, and a green run.
    """
    # repr(), NOT json.dumps(). This text is Python, and `json.dumps(None)` is
    # `null`, which parses fine and dies with a NameError the first time the
    # cell RUNS -- on a Kaggle session, minutes in, having already paid for the
    # box. `test_the_generated_prefetch_cell_runs` exists because compiling the
    # cell did not catch exactly that.
    return f'''
import json, os, threading, time

_REPOS = {_normalise(repos)!r}
_HF_HOME = {hf_home!r}
_ATTEMPT_TIMEOUT = {attempt_timeout}
_TOTAL_TIMEOUT = {total_timeout}
_DEADLINE = time.time() + _TOTAL_TIMEOUT

if _HF_HOME:
    os.environ["HF_HOME"] = _HF_HOME


def _repo_bytes(repo):
    """Size of THIS repo's directory in the hub cache.

    Measured rather than taken from the return value, because
    `snapshot_download` reports a path and not a transfer size, and a repo that
    was ALREADY warm has to read as zero new bytes rather than as its own size
    or every rerun looks like a full download.

    Scoped to the one repo rather than differencing the whole cache, for two
    reasons. The cheap one: walking a cache holding a 12 GB model, twice per
    repo, is real time charged to the very measurement it is taking. The one
    that matters: the legs are downloading into this same cache CONCURRENTLY,
    so a whole-cache delta silently credits their bytes to this lane and
    reports a download rate the Hub never delivered.
    """
    root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    folder = "models--" + repo.replace("/", "--")
    total = 0
    for dirpath, _dirnames, filenames in os.walk(os.path.join(root, "hub", folder)):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _attempt(repo, patterns, disable_xet):
    """One `snapshot_download`, in a thread, under a wall-clock watchdog.

    The watchdog is the point, and it is not the same thing as a retry. Xet
    classifies 408/429/5xx as transient and retries them itself with backoff
    (5 attempts, 3s base, a six-minute cap per delay), so a throttled or
    stalling transfer can sit inside ONE call for many minutes without ever
    raising anything for an ordinary `except` to catch. There are documented
    cases of Xet stalling where classic HTTP ran at line speed. So an attempt
    that stops making progress has to be abandoned on the clock and retried on
    a DIFFERENT transport; retrying the same stalling transport is how a retry
    loop turns into a way to spend the whole session.

    The thread is left running (daemon) rather than killed -- Python cannot
    kill it -- and its bytes are not lost: since huggingface_hub 1.18.0 each
    file lands through a process-unique temporary and an atomic move, and the
    files it already completed stay valid in the cache for the next attempt.
    """
    env_backup = os.environ.get("HF_HUB_DISABLE_XET")
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    box = {{}}

    def _run():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, allow_patterns=patterns)
            box["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            box["error"] = f"{{type(exc).__name__}}: {{exc}}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    budget = min(_ATTEMPT_TIMEOUT, max(1.0, _DEADLINE - time.time()))
    t.join(budget)
    if env_backup is None:
        os.environ.pop("HF_HUB_DISABLE_XET", None)
    else:
        os.environ["HF_HUB_DISABLE_XET"] = env_backup
    if t.is_alive():
        return False, f"no result within {{budget:.0f}}s"
    if box.get("ok"):
        return True, None
    return False, box.get("error", "unknown")


def prefetch_all():
    for repo, patterns in _REPOS:
        started = time.time()
        before = _repo_bytes(repo)
        ok, error, transport, attempts = False, None, None, 0
        # Time inside the attempt that WORKED, separately from the wall clock
        # of the whole repo. They differ by the backoff sleeps and the failed
        # attempts, and conflating them corrupts the one number this lane
        # exists to produce: a run that retried twice would report its 12 GB as
        # having taken 9s longer than it did and understate the achieved rate
        # accordingly. `seconds` is what the SCHEDULE waits for; `download_
        # seconds` is how fast the Hub actually was.
        download_seconds = None
        # Last attempt forces classic HTTP. The first two keep whatever
        # transport the hub chose (Xet when the repo has Xet metadata and
        # hf_xet is installed, which since huggingface_hub 0.32 is the default
        # and is therefore live here whether or not anyone chose it).
        _plan = (False, False, True)
        for _i, disable_xet in enumerate(_plan):
            if time.time() >= _DEADLINE:
                error = "the prefetch budget was spent before this repo"
                break
            attempts += 1
            _t0 = time.time()
            ok, error = _attempt(repo, patterns, disable_xet)
            transport = "http" if disable_xet else "auto"
            if ok:
                download_seconds = round(time.time() - _t0, 1)
                break
            print(f"{PREFETCH_SENTINEL}_RETRY " + json.dumps(
                {{"repo": repo, "attempt": attempts, "error": str(error)[:300]}}),
                flush=True)
            # Backoff only when something will actually follow it. Sleeping
            # after the LAST attempt buys nothing -- there is no retry left to
            # space out -- and it is the difference between this lane giving up
            # in 9s and giving up in 18s, per repo, on the path where the
            # network is already known to be unhappy.
            #
            # Bounded by what is LEFT of the budget too, not just by the curve:
            # a lane already at its deadline that still sleeps 15s is spending
            # session time to accomplish nothing.
            if _i + 1 < len(_plan):
                time.sleep(max(0.0, min(15.0, 3.0 * attempts, _DEADLINE - time.time())))
        seconds = round(time.time() - started, 1)
        moved = max(0, _repo_bytes(repo) - before)
        print(f"{PREFETCH_SENTINEL} " + json.dumps({{
            "repo": repo, "ok": bool(ok), "seconds": seconds,
            "download_seconds": download_seconds, "bytes": moved,
            "mb_per_s": (round(moved / 1e6 / download_seconds, 1)
                         if download_seconds else None),
            # Reported so an over-narrow filter is visible rather than silent.
            # A pattern that matches nothing downloads nothing, reports ok and
            # leaves the payload to fetch the model itself -- a prefetch that
            # looks perfect and does nothing. `bytes` next to `patterns` is
            # what makes that readable in the summary.
            "patterns": patterns,
            "transport": transport, "attempts": attempts,
            "error": None if ok else str(error)[:300],
        }}), flush=True)


# Never raises. A prefetch is an optimisation: the payload that wants the model
# still downloads it for itself, so a failure here costs seconds, and letting
# it propagate would invent a way for the kernel to go red for something that
# is not under test.
try:
    prefetch_all()
except BaseException as exc:  # noqa: BLE001
    print(f"{PREFETCH_SENTINEL}_ABORTED " + json.dumps(
        {{"error": f"{{type(exc).__name__}}: {{exc}}"}}), flush=True)
'''
