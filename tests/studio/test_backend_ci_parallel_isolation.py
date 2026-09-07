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
import importlib.util
import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "studio-backend-ci.yml"

# (ignored path, why it cannot share a worker)
ISOLATED = [
    ("tests/studio/load_freeze", "wall-clock latency bounds"),
    ("tests/studio/test_hardware_dispatch_matrix.py", "mutates hardware.py globals"),
    ("tests/studio/test_is_mlx_dispatch_gate.py", "mutates hardware.py globals"),
    ("tests/studio/test_xpu_spoof_pipeline.py", "mutates hardware.py globals"),
    ("tests/studio/test_mlx_context_platform_matrix.py", "mutates hardware.py globals"),
]


def _pytest_commands(text: str) -> list[str]:
    """Every `python -m pytest ...` invocation in the workflow, line joins resolved.

    Read off the raw text rather than the parsed YAML: a `run:` block is one
    scalar and the interesting structure is inside it, so parsing buys nothing and
    would make this depend on the job/step layout instead of the commands.
    """
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    return [
        line.strip()
        for line in joined.splitlines()
        if "python -m pytest" in line and not line.lstrip().startswith("#")
    ]


# Two different trees are run in parallel now, and the isolation below belongs to exactly one of them. The repo-root
# job runs `tests/` from the checkout; the matrix job runs the backend's own suite with
# `working-directory: studio/backend`, so `tests/studio/...` is not a path that exists for it.
# Told apart by what they ignore, because that is the thing both the ignores and this scan are about: only the
# repo-root run excludes directories that live at the repo root.
REPO_ROOT_MARKER = "--ignore=tests/qlora"


def _over_the_repo_root(command: str) -> bool:
    return REPO_ROOT_MARKER in command


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
    # Found by staging rather than by the scan, and the scan cannot find it: see below.
    ("tests/test_tunnel_safe_long_post.py", "work sleeps 0.2s past a 0.05s keepalive timer"),
    ("tests/test_scan_loras_off_event_loop.py", "counts heartbeats during a 0.3s sleep"),
    ("tests/test_anthropic_messages.py", "counts SSE keepalives emitted during a 0.24s stall"),
    ("tests/test_profile_stats.py", "counts event-loop ticks during a 0.5s blocking call"),
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
BACKEND_MARKER = "--ignore=tests/test_studio_api.py"


def _over_the_backend(command: str) -> bool:
    return BACKEND_MARKER in command


TIGHT_BOUND_S = 0.1

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
    """
    try:
        tree = ast.parse(path.read_text(encoding = "utf-8", errors = "replace"))
    except SyntaxError:
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
    return found


@pytest.mark.parametrize("path, reason", ISOLATED, ids = [p for p, _ in ISOLATED])
def test_an_isolated_path_is_ignored_by_every_parallel_pytest_run(path, reason):
    for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8")):
        if " -n " not in f" {command} " or not _over_the_repo_root(command):
            continue
        assert f"--ignore={path}" in command, (
            f"{path} ({reason}) is not ignored by a parallel pytest run in "
            f"{WORKFLOW.name}, so it shares four workers on the runner's four vCPUs: {command}"
        )


@pytest.mark.parametrize("path, reason", ISOLATED, ids = [p for p, _ in ISOLATED])
def test_an_isolated_path_still_runs_in_a_serial_step(path, reason):
    """Ignoring it is half the change. Without this, the tests silently stop running."""
    serial = [
        command
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if " -n " not in f" {command} "
        and re.search(rf"(?<![\w/]){re.escape(path)}(?![\w/])", command)
    ]
    assert serial, (
        f"{path} is ignored from the parallel run ({reason}) and no serial pytest step runs "
        f"it, so it runs nowhere in {WORKFLOW.name} while the job stays green."
    )


def test_the_command_scan_sees_the_parallel_run_and_the_serial_steps():
    """Pin the parser: a scan that matched nothing would pass both tests above."""
    commands = _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
    parallel = [command for command in commands if " -n " in f" {command} "]
    assert len(parallel) == 2, (
        f"expected two parallel pytest runs, the backend matrix and repo-cpu-tests, got "
        f"{parallel}. If a job stopped running in parallel, say so here rather than "
        f"letting this scan quietly cover one run."
    )
    root = [command for command in parallel if _over_the_repo_root(command)]
    assert len(root) == 1, (
        f"expected exactly one parallel run over the repo root, got {root}. The isolation "
        f"checks above apply to that one, and a scan that matched none of them would pass "
        f"on nothing."
    )
    # The line joins have to be resolved, or the parallel command reads as `pytest tests/ -q` with none of its --ignore
    # flags and the first test above passes on nothing.
    assert "--ignore=" in root[0]
    assert len(commands) > 1, "no serial pytest steps found; the ignore checks cannot fail"


def test_the_backend_matrix_still_runs_in_parallel():
    """The matrix leg was 23.3 minutes serial and is the longest job in the repo.

    Measured over the same tree before it was turned on: 1322.6s serial against 343.0s at
    -n 4, with the two failure sets equal name for name, so nothing in the backend suite
    depends on the order it runs in. Asserted here because dropping the flag would show up
    only as CI slowly getting slower again, which nothing reports.
    """
    backend = [
        command
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if "--ignore=tests/test_studio_api.py" in command
    ]
    assert backend, "the backend matrix pytest step is gone or was renamed past this scan"
    assert " -n " in f" {backend[0]} ", (
        f"the backend matrix leg is running serially again, which costs about 17 minutes "
        f"per leg on every pull request and every push to main: {backend[0]}"
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
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if " -n " in f" {command} " and _over_the_backend(command)
    ]
    assert parallel, "the backend parallel run is gone or was renamed past this scan"
    assert f"--ignore={path}" in parallel[0], (
        f"{path} ({reason}) is back in the backend parallel run, where its measurements "
        f"compare a descheduled worker against an undescheduled one: {parallel[0]}"
    )


@pytest.mark.parametrize("path, reason", BACKEND_ISOLATED, ids = [p for p, _ in BACKEND_ISOLATED])
def test_a_backend_isolated_path_still_runs_serially(path, reason):
    """Ignoring it is half the change; without this it runs nowhere and the job is green."""
    serial = [
        command
        for command in _pytest_commands(WORKFLOW.read_text(encoding = "utf-8"))
        if " -n " not in f" {command} "
        and re.search(rf"(?<![\w/]){re.escape(path)}(?![\w/])", command)
    ]
    assert serial, (
        f"{path} is ignored from the backend parallel run ({reason}) and no serial step "
        f"runs it, so it runs nowhere in {WORKFLOW.name} while the job stays green."
    )


def test_every_tight_elapsed_bound_is_isolated():
    """The rule, applied by scanning rather than by memory.

    Two of the entries above were found by review rather than by CI: they passed on
    staging and would have flaked later. A new test asserting a 20ms bound would do the
    same. This finds them, so adding one forces the isolation instead of buying a flake.
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


def test_the_scan_finds_all_three_shapes():
    """A scan that matched nothing would pass the test above on an empty set.

    One of each form the suite actually uses, because each needed its own handling and
    the first version of this scan only understood the first:
      elapsed < 0.05                        a name assigned from a difference
      time.monotonic() - started < 0.2      the difference written inline
      _elapsed(big) < 8 * _elapsed(small)   a helper that returns a difference
    """
    found = {
        path.name: _fragile_timing_asserts(path)
        for path in sorted(BACKEND_TESTS.glob("*.py"))
        if _fragile_timing_asserts(path)
    }
    assert "test_llama_cpp_wait_for_vram_settle.py" in found, found  # named
    assert "test_tool_xml_strip.py" in found, found  # named
    assert "test_diffusion_checkpoint_resume.py" in found, found  # helper, relative

    # The inline form, which the suite currently uses only at 0.2s, above the threshold.
    inline = ast.parse(
        "import time\n"
        "def t():\n"
        "    started = 0\n"
        "    assert time.monotonic() - started < 0.05\n"
    )
    names, helpers = _timed_names(inline), _timing_helpers(inline)
    node = [n for n in ast.walk(inline) if isinstance(n, ast.Assert)][0]
    compare = node.test
    assert _is_timed(compare.left, names, helpers), (
        "an inline clock difference is not recognised as a duration, so a test written "
        "that way could assert a 20ms bound and run under -n 4 unnoticed"
    )


def test_an_isolated_file_never_shadows_an_installed_library_with_a_stub():
    """A stub may stand in for a MISSING library, never for an installed one.

    `sys.modules.setdefault("httpx", stub)` reads as deferring to the real library and
    does not: sys.modules holds what has been IMPORTED, not what is installed, so in a
    process where nothing has touched httpx yet the stub wins and shadows it for the rest
    of the session. These stubs carry no Response, starlette.testclient reads
    httpx.Response at import, and every module collected afterwards that reaches
    fastapi.testclient or routes.inference dies on it.

    In a 26,000-test run something always imports httpx first, so this was invisible for
    as long as the suite ran as one process. The serial step collects ten files and
    nothing else, and the 3.10 leg failed collection on two of them the first time it
    ran.

    Scoped to the isolated files on purpose. Roughly fifty other backend modules stub
    structlog the same way, and they are load-bearing in a run that also imports the real
    one; rewriting them is a separate change with its own risk, and the full parallel run
    is not the process where a small file list makes the shadowing decisive. What has to
    hold here is that anything moved OUT of that run stands on its own.
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
