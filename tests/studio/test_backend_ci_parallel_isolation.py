# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards the files Backend CI deliberately keeps out of the parallel pytest run.

The repo-cpu-tests job runs tests/ under `-n 4`. Two groups cannot go through it:

- tests/studio/load_freeze asserts UPPER bounds on real elapsed time (50 concurrent
  probes under 15s, a fast-shim probe under 2s, five sequential probes under 10s, a
  not-loaded short circuit under 50 ms), and a pytest worker descheduled by the other
  three inflates them. The two tightest bounds it used to carry, 250 ms for a /health
  burst and 350 ms for a 100-request burst, are gone: those two tests now hold the
  blocking call open on an event and assert that /health answers while it is held,
  which is the property the bounds were standing in for and does not move with load.
- the hardware-spoof files mutate hardware.py module globals, so they leak into
  whatever shares their worker.

Both are ignored from the parallel invocation and run again in their own serial
step. That is two edits held together by nothing, and dropping the second one is
silent: the job stays green while the tests stop running. These tests fail if the
ignore appears without a step that runs the same path, or the other way round.
"""

import ast
import fnmatch
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "studio-backend-ci.yml"

# (ignored path, why it cannot share a worker)
ISOLATED = [
    ("tests/studio/load_freeze", "wall-clock latency bounds"),
    ("tests/studio/test_hardware_dispatch_matrix.py", "mutates hardware.py globals"),
    ("tests/studio/test_is_mlx_dispatch_gate.py", "mutates hardware.py globals"),
    ("tests/studio/test_xpu_spoof_pipeline.py", "mutates hardware.py globals"),
    ("tests/studio/test_mlx_context_platform_matrix.py", "mutates hardware.py globals"),
]


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]


def _selections(job_name: str) -> list[str]:
    """The `selection` of each matrix entry of one job, whitespace-normalised.

    Not every entry has one: the `pytest` job's 3.11 floor-spot-check leg names three
    files directly and is not a shard of anything, so it carries no selection.
    """
    job = _jobs()[job_name]
    include = job.get("strategy", {}).get("matrix", {}).get("include", [])
    return [" ".join(entry["selection"].split()) for entry in include if "selection" in entry]


def _commands_in(job_name: str) -> list[str]:
    """Every `python -m pytest ...` invocation of one job, line joins and matrix resolved.

    Read off the raw text of each `run:` scalar rather than off more parsed YAML: a
    `run:` block is one scalar and the interesting structure is inside it, so parsing
    further buys nothing and would make this depend on step layout instead of commands.

    The one thing that cannot be read off the text is `${{ matrix.selection }}`, because
    the paths live in the matrix rather than in the command. BOTH parallel jobs are
    sharded now and both write their step that way, so the substitution has to know whose
    matrix to read: expanding a command with the other job's selections would build runs
    that are not in the workflow and ask the isolation questions below of those instead.
    Hence per job, which is also how the caller knows the owner without guessing from a
    flag that happens to appear in one of them.
    """
    job = _jobs()[job_name]
    selections = _selections(job_name)
    commands = []
    for step in job.get("steps", []):
        joined = re.sub(r"\\\s*\n\s*", " ", str(step.get("run", "")))
        for line in joined.splitlines():
            line = line.strip()
            if "python -m pytest" not in line or line.startswith("#"):
                continue
            if "${{ matrix.selection }}" in line:
                commands.extend(
                    line.replace("${{ matrix.selection }}", selection) for selection in selections
                )
            else:
                commands.append(line)
    return commands


def _pytest_commands() -> list[str]:
    """Every pytest invocation in the workflow, across every job."""
    return [command for job_name in _jobs() for command in _commands_in(job_name)]


def _collects(command: str, path: str) -> bool:
    """Whether a pytest command would collect `path`, by its roots and its ignore flags.

    An isolated path used to be kept out of the parallel run by naming it in an --ignore.
    A shard that does not name its directory at all keeps it out just as effectively, so
    the question the guard asks is whether the run reaches the path, not how.

    --ignore-glob is read as well as --ignore, because the backend shards are told apart
    by nothing else: all three root at `tests/` and differ only in which glob they
    exclude. Treating those flags as noise would have every backend shard appear to
    collect every backend file, and "not collected by any parallel run" would then be
    unfalsifiable for the whole suite.
    """
    tokens = command.split()
    roots, ignores, globs = [], [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--ignore-glob="):
            globs.append(token.split("=", 1)[1].strip("'\""))
        elif token.startswith("--ignore="):
            ignores.append(token.split("=", 1)[1].rstrip("/"))
        elif token == "--deselect":
            index += 1
        elif token.startswith("tests/") or token == "tests":
            roots.append(token.rstrip("/"))
        index += 1

    def under(prefix, candidate):
        return candidate == prefix or candidate.startswith(prefix + "/")

    if any(under(ignore, path) for ignore in ignores):
        return False
    if any(_ignore_glob_hits(pattern, path) for pattern in globs):
        return False
    return any(under(root, path) for root in roots)


def _ignore_glob_hits(pattern: str, path: str) -> bool:
    """pytest's own --ignore-glob rule, for a repo-relative path.

    pytest matches with `_pytest.pathlib.fnmatch_ex`, which fnmatches the ABSOLUTE path
    against the pattern with `*/` prepended when the pattern contains a separator and is
    relative. fnmatch's `*` crosses `/`, so the prefix is free and matching the suffix of
    the relative path is the same question. Verified against pytest itself rather than
    assumed: `--ignore-glob=tests/test_[a-h]*.py` does exclude tests/test_a.py and does
    NOT exclude tests/sub/test_a.py, which is the property the shards below rely on.
    """
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"*/{pattern}")


# Two different trees are run in parallel now, and the isolation below belongs to exactly one of them. The repo-root
# job runs `tests/` from the checkout; the matrix job runs the backend's own suite with
# `working-directory: studio/backend`, so `tests/studio/...` is not a path that exists for it.
# Told apart by the selection they carry. This used to be one flag, `--ignore=tests/qlora`, which only the repo-root
# run had; the repo-root run is three shards now and only one of them carries that flag, so the test is membership of
# the matrix instead. The backend matrix run is not in it.
#
# Both sides are membership now. The backend run used to be told apart by one of its --ignore flags, which stopped
# identifying it once it was sharded the same way and its flags moved onto a shared step. Reading each side out of its
# own job's matrix is the same question asked the same way twice, and it cannot start matching the other job by
# accident the way a shared flag can; the shape test below asserts the two sets stay disjoint.


def _over_the_repo_tests(command: str) -> bool:
    return any(selection in command for selection in _selections("repo-cpu-tests"))


# The same pairing, for the backend matrix run. Ignoring a file from the parallel run and running it again serially is
# two edits held together by nothing, and dropping the second is silent: the job stays green while the tests stop
# running.
BACKEND_ISOLATED = [
    ("tests/test_streaming_stripper.py", "times itself against a reference in the same process"),
    ("tests/test_llama_cpp_wait_for_vram_settle.py", "asserts elapsed < 0.05"),
    ("tests/test_tool_xml_strip.py", "asserts a regex benchmark under 0.1s"),
    ("tests/test_diffusion_checkpoint_resume.py", "compares one duration against another"),
    (
        "tests/test_tool_output_streaming.py",
        "compares when a callback fired against when the child exited",
    ),
    ("tests/test_web_fetch_extraction.py", "compares parse time at two input sizes"),
    ("tests/test_tool_call_parser_strict.py", "compares parse time at two nesting depths"),
    ("tests/test_pr5624_regressions.py", "R1 parser's 1s bound exceeded under CPU contention"),
    # Found by staging rather than by the scan, and the scan cannot find it: see below.
    (
        "tests/test_tunnel_safe_long_post.py",
        ":101 requires len(chunks) > 2, which one 100ms stall falsifies",
    ),
    ("tests/test_scan_loras_off_event_loop.py", "counts heartbeats during a 0.3s sleep"),
    ("tests/test_anthropic_messages.py", "counts SSE keepalives emitted during a 0.24s stall"),
    # The tick count is not the tight part of this file: it has 5x margin. :400 is,
    # `assert elapsed < 0.2` around a join that already costs 0.03s on an idle box.
    ("tests/test_profile_stats.py", ":400 asserts elapsed < 0.2 around a 0.03s join"),
    # Was ignored by the parallel run and rerun serially and named in NEITHER direction
    # here, so the file was the one thing this guard cannot see: deleting it from the
    # serial step would have left it running nowhere with the job green. It qualifies
    # twice over. Timing-tight at :1101 `assert started.is_set()` under a patched
    # _SWITCH_BUDGET_S = 0.3, against a cold path the file itself records at 1.98s -- and
    # like test_tunnel_safe_long_post, the assertion is on a RESULT rather than on a
    # duration, so the scan below cannot reach it. State-mutating as well: it writes
    # keepwarm globals directly at :2628-2630. It has already failed this way once on
    # 3.13 while 3.10 passed the same commit.
    (
        "tests/test_media_auto_switch.py",
        ":1101 asserts started.is_set() under a patched 0.3s budget on a 1.98s cold path, "
        "and writes keepwarm globals at :2628-2630",
    ),
]

# What the scan above does NOT cover, recorded because the gap is structural rather than a missing case. It finds
# assertions that COMPARE clock-derived values. A test can depend on timing without any clock in it at all:
# test_tunnel_safe_long_post patches the keepalive threshold to 0.05s and makes the work sleep 0.2s, then asserts on
# the RESULT -- that the response starts with padding -- so whether it passes turns on which of two timers fired
# first, and nothing in the expression is a duration. It failed exactly that way on a staging 3.13 leg that had been
# green.
#
# test_scan_loras_off_event_loop is the same shape from the other direction: it counts how many times a heartbeat
# coroutine ticked during a 0.3s sleep and requires at least three. Descheduling the worker costs ticks without the
# scan being wrong, and the assertion compares a COUNT, so again there is no duration to find.
#
# Ten backend files pair a sub-second sleep with a small threshold constant. Four times the threshold was not enough
# margin for the one that failed, so the ratio is not a usable rule, and flagging all ten would serialise a large part
# of the suite on a guess.
#
# So this class is found by reading rather than by scanning. The first arrived from a staging failure, the second from
# review, and the third from reading the other eight candidates once the shape was clear: test_anthropic_messages
# counts SSE keepalives emitted during a 0.24s stall, which loses keepalives to a descheduled worker exactly as the
# heartbeat test loses ticks.
#
# That same pass turned up one false positive worth naming, because the grep that finds these is crude:
# test_diffusion_backend asserts len(staged) > 1 near a 0.2s sleep, but `staged` is a list comprehension over cached
# filenames and has no timing in it at all. It also costs 152s, so isolating it on the strength of a pattern match
# would have been expensive as well as wrong. Read the assertion before adding a file here.

# Below this, an elapsed-time bound is inside the range of a single scheduler quantum, so under four workers on four
# vCPUs it measures the scheduler as much as the code. Above it there is enough headroom to survive being descheduled.
# Twenty-two backend files assert some elapsed bound and serialising all of them would give back most of what -n 4
# buys, so the line is drawn where the measurement stops being about the code.
TIGHT_BOUND_S = 0.1


def _over_the_backend(command: str) -> bool:
    return any(selection in command for selection in _selections("pytest"))


BACKEND_TESTS = Path(__file__).resolve().parents[2] / "studio" / "backend" / "tests"
_CLOCKS = ("monotonic", "perf_counter", "process_time", "time")


# Sites the scan finds and a human has read. The scan looks for a comparison between two clock-derived quantities,
# which is the right net to cast, but not every such comparison is a performance claim. None of these can be broken
# by descheduling:
#
#   a SANDWICH, `before <= recorded <= after`, asserts a stamp was taken between two reads. Widening the gap cannot
#   falsify it.
#   a POLL DEADLINE, `time.monotonic() < limit` inside a wait-for-condition loop, is the pattern that replaces a
#   guessed sleep. Its 5s budget is a timeout, not a measurement.
#   a SENTINEL, `stamp < 0.0`, compares against a magic value rather than a duration.
#
# Keyed on the enclosing function rather than a line number, so an edit above it does not silently move the exemption
# onto something else.
BENIGN_TIMING = {
    ("test_media_auto_switch.py", "_until"),
    ("test_openai_auto_switch.py", "test_any_finished_download_drops_the_resolver_cache"),
    # A 600-second expiry checked against the wall clock.
    # Reading both sides of that gap late by whole seconds still leaves it true, and it only reaches this scan at all
    # because the widened operand walk now reads `x > time.time()` as a bound.
    (
        "test_openai_codex_subscription.py",
        "test_account_claim_and_token_response_are_validated_without_returning_raw_body",
    ),
}


def _reads_a_clock(node: ast.AST) -> bool:
    return any(
        isinstance(inner, ast.Call) and getattr(inner.func, "attr", "") in _CLOCKS
        for inner in ast.walk(node)
    )


def _calls_a_helper(node: ast.AST, helpers: set) -> bool:
    return any(
        isinstance(inner, ast.Call) and getattr(inner.func, "id", None) in helpers
        for inner in ast.walk(node)
    )


def _timing_helpers(tree: ast.AST) -> set:
    """Functions that hand back a clock value, however indirectly.

    Not just ``return time.perf_counter() - t0``. test_tool_call_parser_strict has

        def best_ms(depth):
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ...
                best = min(best, time.perf_counter() - t0)
            return best

    where the return reads no clock at all: the duration arrives through a local name. So
    a function counts if it returns anything containing one of its OWN timed names, and
    the whole thing runs to a fixpoint, so a helper that returns another helper's result
    is found on the next pass rather than missed.
    """
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    helpers: set = set()
    while True:
        grown = False
        for node in functions:
            if node.name in helpers:
                continue
            # With the helpers found so far, not without them: `value = base()` inside a wrapper only counts as
            # timed once `base` is known, and the pass that learns `base` is not the pass that reads the wrapper.
            local = _timed_names(node, helpers)
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Return) or inner.value is None:
                    continue
                if _is_timed(inner.value, local, helpers):
                    helpers.add(node.name)
                    grown = True
                    break
        if not grown:
            return helpers


def _timed_names(tree: ast.AST, helpers: set = frozenset()) -> set:
    """Anything holding a clock value: a duration, an instant, or a list of them.

    Three ways one gets there, all present in this suite:
        elapsed = time.monotonic() - start      a difference
        started = time.monotonic()              an instant, subtracted later
        first_seen_at.append(time.monotonic())  an instant parked in a container,
                                                usually from inside a callback

    Instants count, not only differences. test_tool_output_streaming compares
    `first_seen_at[0] - started` against `finished - started - 0.5`, where every term is
    an instant and no single name ever holds a duration.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and (
            _reads_a_clock(node.value) or _calls_a_helper(node.value, helpers)
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("append", "add", "insert") and _reads_a_clock(node):
                holder = node.func.value
                if isinstance(holder, ast.Name):
                    names.add(holder.id)
    return names


def _is_timed(node: ast.AST, names: set, helpers: set) -> bool:
    """Whether this expression is a duration, however it was spelled.

    Three forms, all of which appear in this suite:
      elapsed < 0.05                          a name assigned from a difference
      time.monotonic() - started < 0.2        the difference written inline
      _elapsed(big) < 8 * _elapsed(small)     a helper that returns a difference
    """
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in names:
            return True
        if isinstance(inner, ast.Call):
            if getattr(inner.func, "attr", "") in _CLOCKS:
                return True
            if getattr(inner.func, "id", None) in helpers:
                return True
    return False


_FRAGILE_CACHE: dict = {}


def _fragile_timing_asserts(path: Path) -> list:
    """Assertions whose outcome depends on how the process was scheduled.

    Two kinds, and the second has no threshold to be under:
      * ABSOLUTE, at or below TIGHT_BOUND_S. A bound that small is inside one scheduler
        quantum, so four workers on four vCPUs measure the scheduler as much as the code.
      * RELATIVE, comparing one duration against another. Descheduling one side and not
        the other breaks it at ANY magnitude, which is what took test_streaming_stripper
        out of the parallel run.

    Read with ast, not a regex: grepping `< 0.05` matches a float tolerance, and grepping
    `elapsed` matches whatever a variable happens to be called.

    Memoised on (resolved path, file text). Two tests below scan all 924 backend test
    files and the dict comprehension in one of them calls this twice per path, so the
    same parse-and-walk ran roughly three times over: 19.0s + 20.6s of the file's 37.8s.
    The read is deliberately still done every call and the text is part of the key, so a
    file rewritten mid-session is rescanned rather than served a stale verdict; only the
    parse and the walks are shared. The stored list is copied out, so no caller can
    mutate another's result, and the tree never leaves this function.
    """
    source = path.read_text(encoding = "utf-8", errors = "replace")
    key = (str(path.resolve()), source)
    cached = _FRAGILE_CACHE.get(key)
    if cached is not None:
        return list(cached)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        _FRAGILE_CACHE[key] = []
        return []
    # Helpers first: a name can hold a duration only because a helper returned one.
    helpers = _timing_helpers(tree)
    names = _timed_names(tree, helpers)
    enclosing = {}
    for holder in ast.walk(tree):
        if isinstance(holder, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(holder):
                enclosing.setdefault(inner, holder.name)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        where = enclosing.get(node, "<module>")
        if (path.name, where) in BENIGN_TIMING:
            continue
        for cmp_node in ast.walk(node.test):
            if not isinstance(cmp_node, ast.Compare):
                continue
            # Every adjacent pair, not just the one starting at cmp_node.left.
            # A chained `0.3 <= elapsed < 2.0` is a single Compare whose first operand is a literal, so requiring the
            # leftmost operand to be timed skipped the `elapsed < 2.0` link and let the file stay in the -n 4 run with
            # the guard still green.
            # test_llama_cpp_wait_for_vram_settle.py already writes bounds that way.
            operands = [cmp_node.left, *cmp_node.comparators]
            for index, op in enumerate(cmp_node.ops):
                lower, upper = operands[index], operands[index + 1]
                if isinstance(op, (ast.Gt, ast.GtE)):
                    # `0.05 > elapsed` bounds the same thing from the same side.
                    lower, upper = upper, lower
                elif not isinstance(op, (ast.Lt, ast.LtE)):
                    continue
                if not _is_timed(lower, names, helpers):
                    continue
                if _is_timed(upper, names, helpers):
                    found.append(f"{path.name}:{node.lineno} one duration against another")
                elif isinstance(upper, ast.Constant) and isinstance(upper.value, (int, float)):
                    if upper.value <= TIGHT_BOUND_S:
                        found.append(f"{path.name}:{node.lineno} duration < {upper.value}")
    _FRAGILE_CACHE[key] = found
    return list(found)


@pytest.mark.parametrize("path, reason", ISOLATED, ids = [p for p, _ in ISOLATED])
def test_an_isolated_path_is_ignored_by_every_parallel_pytest_run(path, reason):
    for command in _pytest_commands():
        if " -n " not in f" {command} " or not _over_the_repo_tests(command):
            continue
        assert not _collects(command, path), (
            f"{path} ({reason}) is collected by a parallel pytest run in "
            f"{WORKFLOW.name}, so it shares four workers on the runner's four vCPUs: {command}"
        )


@pytest.mark.parametrize("path, reason", ISOLATED, ids = [p for p, _ in ISOLATED])
def test_an_isolated_path_still_runs_in_a_serial_step(path, reason):
    """Ignoring it is half the change. Without this, the tests silently stop running."""
    serial = [
        command
        for command in _pytest_commands()
        if " -n " not in f" {command} "
        and re.search(rf"(?<![\w/]){re.escape(path)}(?![\w/])", command)
    ]
    assert serial, (
        f"{path} is ignored from the parallel run ({reason}) and no serial pytest step runs "
        f"it, so it runs nowhere in {WORKFLOW.name} while the job stays green."
    )


def test_the_command_scan_sees_the_parallel_run_and_the_serial_steps():
    """Pin the parser: a scan that matched nothing would pass both tests above."""
    commands = _pytest_commands()
    parallel = [command for command in commands if " -n " in f" {command} "]
    assert len(parallel) == 6, (
        f"expected six parallel pytest runs, three shards of the backend matrix and three "
        f"of repo-cpu-tests, got {parallel}. If a job stopped running in parallel, or a "
        f"shard was added or removed, say so here rather than letting this scan quietly "
        f"cover fewer runs."
    )
    root = [command for command in parallel if _over_the_repo_tests(command)]
    assert len(root) == 3, (
        f"expected the three repo-root shards, got {root}. The isolation checks above apply "
        f"to those, and a scan that matched none of them would pass on nothing."
    )
    backend = [command for command in parallel if _over_the_backend(command)]
    assert len(backend) == 3, (
        f"expected the three backend shards, got {backend}. Same reason: the backend "
        f"isolation checks apply to those."
    )
    # The two jobs are told apart by whose matrix a command's paths came out of, so the sets have to be disjoint or
    # each job's isolation rules would be applied to the other's runs -- which is how tests/studio/... would come to be
    # asked of a run whose working directory is studio/backend.
    assert not set(root) & set(
        backend
    ), f"a command reads as belonging to both jobs: {set(root) & set(backend)}"
    assert len(parallel) == len(root) + len(backend), (
        f"a parallel run belongs to neither job's matrix, so nothing below checks it: "
        f"{[command for command in parallel if command not in root + backend]}"
    )
    # The line joins and the matrix substitution both have to be resolved, or a shard command reads as
    # `pytest ${{ matrix.selection }} -q` with no paths at all and the first test above passes on nothing.
    assert all("${{" not in command for command in root + backend)
    assert any("--ignore=" in command for command in root)
    assert {command for command in root} == set(root), "a shard selection appears twice"
    assert len(set(backend)) == 3, "a backend shard selection appears twice"
    # Non-vacuous for the backend side too: the shards between them must reach the backend suite.
    assert any(_collects(command, "tests/test_account_contract.py") for command in backend)
    # Non-vacuous the other way too: the shards between them must reach the repo's test root, or "not collected by any
    # parallel run" would be true of every path in the repo.
    assert any(_collects(command, "tests/test_model_registry.py") for command in root)
    assert len(commands) > 1, "no serial pytest steps found; the ignore checks cannot fail"


def test_the_backend_matrix_still_runs_in_parallel():
    """The matrix leg was 23.3 minutes serial and is the longest job in the repo.

    Measured over the same tree before it was turned on: 1322.6s serial against 343.0s at
    -n 4, with the two failure sets equal name for name, so nothing in the backend suite
    depends on the order it runs in. Asserted here because dropping the flag would show up
    only as CI slowly getting slower again, which nothing reports.
    """
    backend = [command for command in _pytest_commands() if _over_the_backend(command)]
    assert backend, "the backend matrix pytest step is gone or was renamed past this scan"
    for command in backend:
        assert " -n " in f" {command} ", (
            f"a backend matrix shard is running serially again, which costs about 17 "
            f"minutes on every pull request and every push to main: {command}"
        )


@pytest.mark.parametrize("path, reason", BACKEND_ISOLATED, ids = [p for p, _ in BACKEND_ISOLATED])
def test_a_backend_isolated_path_is_ignored_by_the_parallel_run(path, reason):
    """Relative timing cannot survive four workers on four vCPUs.

    Observed on staging: the 3.10 leg reported "early markup cost 1.354s against the
    reference's 0.854s" while 3.13 passed the same commit. One side of the ratio was
    descheduled, not slower.
    """
    parallel = [
        command
        for command in _pytest_commands()
        if " -n " in f" {command} " and _over_the_backend(command)
    ]
    assert parallel, "the backend parallel run is gone or was renamed past this scan"
    # Asked of EVERY shard, as "does this run reach the path" rather than "does it spell
    # this --ignore", so a shard that quietly grew its own root still has to get past it.
    for command in parallel:
        assert not _collects(command, path), (
            f"{path} ({reason}) is back in a backend parallel run, where its measurements "
            f"compare a descheduled worker against an undescheduled one: {command}"
        )


@pytest.mark.parametrize("path, reason", BACKEND_ISOLATED, ids = [p for p, _ in BACKEND_ISOLATED])
def test_a_backend_isolated_path_still_runs_serially(path, reason):
    """Ignoring it is half the change; without this it runs nowhere and the job is green."""
    serial = [
        command
        for command in _pytest_commands()
        if " -n " not in f" {command} "
        and re.search(rf"(?<![\w/]){re.escape(path)}(?![\w/])", command)
    ]
    assert serial, (
        f"{path} is ignored from the backend parallel run ({reason}) and no serial step "
        f"runs it, so it runs nowhere in {WORKFLOW.name} while the job stays green."
    )


def test_every_tight_elapsed_bound_is_isolated():
    """The rule, applied by scanning rather than by memory.

    Two entries above were found by review, not CI: they passed on staging and would have
    flaked later. This finds them, so adding one forces the isolation instead of a flake.
    """
    isolated = {path for path, _ in BACKEND_ISOLATED}
    stray = {}
    for path in sorted(BACKEND_TESTS.glob("*.py")):
        bounds = _fragile_timing_asserts(path)
        if bounds and f"tests/{path.name}" not in isolated:
            stray[path.name] = bounds
    assert not stray, (
        f"these backend tests compare clock-derived values and still run under -n 4, "
        f"where four workers share four vCPUs: {stray}.\n"
        f"\n"
        f"Three ways out, in the order worth trying:\n"
        f"  1. If it is a PERFORMANCE claim -- one measurement against another, or an "
        f"absolute bound at or below {TIGHT_BOUND_S}s -- add the file to "
        f"BACKEND_ISOLATED and to BOTH halves of studio-backend-ci.yml: the --ignore on "
        f"the parallel run and the serial step that reruns it.\n"
        f"  2. If descheduling cannot falsify it, add (file, enclosing function) to "
        f"BENIGN_TIMING with a one-line reason. A sandwich (`before <= x <= after`), a "
        f"poll deadline, and a sentinel comparison are all already there. This net is "
        f"cast wide on purpose, so landing here does not mean the test is wrong.\n"
        f"  3. If it is an absolute bound that is simply too tight, give it enough "
        f"headroom to survive being descheduled."
    )


def test_the_scan_finds_all_three_shapes(tmp_path):
    """A scan that matched nothing would pass the test above on an empty set.

    One of each form the suite actually uses, because each needed its own handling and
    the first version of this scan only understood the first:
      elapsed < 0.05                        a name assigned from a difference
      time.monotonic() - started < 0.2      the difference written inline
      _elapsed(big) < 8 * _elapsed(small)   a helper that returns a difference

    Written out here rather than named as three real files. Naming them made this test a
    second, invisible reason those files had to keep their fragile bounds: rewriting
    `test_llama_cpp_wait_for_vram_settle.py` to assert on the naps the helper asks for
    instead of on how long they took -- which is the outcome the scan exists to push
    people toward -- failed HERE, in a file about CI topology, with a message about a
    sample. The scan's own coverage should not depend on the suite still containing the
    thing it is trying to remove.
    """
    shapes = {
        "assigned name": (
            "import time\n"
            "def test_x():\n"
            "    started = time.monotonic()\n"
            "    work()\n"
            "    elapsed = time.monotonic() - started\n"
            "    assert elapsed < 0.05\n"
        ),
        "inline difference": (
            "import time\n"
            "def test_x():\n"
            "    started = time.monotonic()\n"
            "    work()\n"
            "    assert time.monotonic() - started < 0.05\n"
        ),
        "helper, relative": (
            "import time\n"
            "def _elapsed(fn):\n"
            "    started = time.perf_counter()\n"
            "    fn()\n"
            "    return time.perf_counter() - started\n"
            "def test_x():\n"
            "    assert _elapsed(big) < 8 * _elapsed(small)\n"
        ),
    }
    for label, source in shapes.items():
        sample = tmp_path / f"test_{label.replace(' ', '_').replace(',', '')}.py"
        sample.write_text(source, encoding = "utf-8")
        assert _fragile_timing_asserts(sample), (
            f"the scan does not recognise the {label} shape, so a test written that way "
            "could carry a 50ms bound into the -n 4 run unnoticed"
        )

    # And quiet on a bound with headroom, or every anti-hang ceiling goes serial for nothing.
    roomy = tmp_path / "test_roomy.py"
    roomy.write_text(
        "import time\n"
        "def test_x():\n"
        "    started = time.monotonic()\n"
        "    work()\n"
        "    elapsed = time.monotonic() - started\n"
        "    assert elapsed < 30.0\n",
        encoding = "utf-8",
    )
    assert not _fragile_timing_asserts(roomy), _fragile_timing_asserts(roomy)

    # Deliberately nothing about the live suite: the synthetic files cover every shape, and
    # asserting the suite still has some would turn cleaning the last one into a failure.


def test_an_isolated_file_never_shadows_an_installed_library_with_a_stub():
    """A stub may stand in for a MISSING library, never for an installed one.

    `sys.modules.setdefault("httpx", stub)` reads as deferring to the real library and does
    not: sys.modules holds what has been IMPORTED, not what is installed, so where nothing
    has touched httpx yet the stub wins for the rest of the session. These stubs carry no
    Response, starlette.testclient reads httpx.Response at import, and every module after it
    reaching fastapi.testclient or routes.inference dies on it. In a 26,000-test run
    something always imports httpx first, so this stayed invisible while the suite was one
    process; the serial step collects ten files, and the 3.10 leg failed collection on two.

    Scoped to the isolated files on purpose: ~fifty other backend modules stub structlog the
    same way and are load-bearing in a run that also imports the real one. What has to hold
    here is that anything moved OUT of the parallel run stands on its own.
    """
    offenders = {}
    for name, _reason in BACKEND_ISOLATED:
        path = BACKEND_TESTS / Path(name).name
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        stubbed = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _installs_into_sys_modules(node)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        stubbed |= _assigned_into_sys_modules(tree)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        for stub in sorted(stubbed - imported):
            if _is_installed(stub):
                offenders.setdefault(path.name, []).append(stub)
    assert not offenders, (
        f"these files run in the serial step and install a stub over a library that IS "
        f"installed, without first trying to import it: {offenders}.\n"
        f"\n"
        f"Wrap the install in `try: import <name>` / `except ImportError:` the way "
        f"test_llama_cpp_placement.py does. setdefault is not that guard: sys.modules is "
        f"what has been imported, not what is available, so the stub wins whenever this "
        f"module is collected first and shadows the real library for the whole session. "
        f"That is decisive here precisely because the step collects ten files, so there "
        f"is no longer an unrelated module importing the real one first."
    )


def _installs_into_sys_modules(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "setdefault"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "modules"
    )


def _assigned_into_sys_modules(tree: ast.AST) -> set:
    """`sys.modules["name"] = stub`, the other spelling."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "modules"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                names.add(target.slice.value)
    return names


def _is_repo_module(name: str) -> bool:
    """Whether studio/backend itself provides this name.

    `loggers`, `utils`, `routes` and friends are the backend's OWN modules. A test that
    stands one of them up as a stub is not shadowing a third-party library, which is what
    the check below is about; it is substituting for repo code on purpose.
    """
    return (BACKEND_TESTS.parent / name).is_dir() or (BACKEND_TESTS.parent / f"{name}.py").is_file()


def _is_installed(name: str) -> bool:
    """Whether a stub for this name would shadow a real third-party library.

    Asked of the REPO first, and that ordering is the whole fix. The previous version
    asked importlib alone and reasoned that an in-repo name resolves only with
    studio/backend on sys.path, "which this test does not have and should not add". That
    was simply untrue in the job that runs it: under `pytest tests/ -n 4` from the repo
    root, studio/backend does end up on sys.path, `loggers` resolved, and the guard
    failed on main for a stub that shadows nothing. It passed locally, where the path
    happens to differ, which is the worst shape a CI-only assertion can have.

    So the question is answered from the tree, which is the same everywhere, and
    importlib is consulted only for names the repo does not define.
    """
    if _is_repo_module(name):
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
