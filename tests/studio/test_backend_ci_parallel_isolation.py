# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Files ignored by the parallel run must also run in a serial step, and the guard checks both."""

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
    """Entries without a `selection` are skipped, since the floor spot-check leg names its files
    directly."""
    job = _jobs()[job_name]
    include = job.get("strategy", {}).get("matrix", {}).get("include", [])
    return [" ".join(entry["selection"].split()) for entry in include if "selection" in entry]


def _commands_in(job_name: str) -> list[str]:
    """Expands `${{ matrix.selection }}` from this job's own matrix, not the other parallel job's."""
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
    """Reads `--ignore-glob` as well as `--ignore`; backend shards differ only by the glob they exclude."""
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
    """Matches as pytest's `--ignore-glob` does: the pattern, or `*/` plus the pattern, matches the path."""
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"*/{pattern}")


# The repo-root and backend matrix jobs are told apart by membership of each job's own matrix
# selection; the shape test below keeps the two sets disjoint.


def _over_the_repo_tests(command: str) -> bool:
    return any(selection in command for selection in _selections("repo-cpu-tests"))


# Ignored in the parallel run and rerun serially; dropping the rerun would be silent.
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
    # Found by staging; the scan cannot find it (see below).
    (
        "tests/test_tunnel_safe_long_post.py",
        ":101 requires len(chunks) > 2, which one 100ms stall falsifies",
    ),
    ("tests/test_scan_loras_off_event_loop.py", "counts heartbeats during a 0.3s sleep"),
    ("tests/test_anthropic_messages.py", "counts SSE keepalives emitted during a 0.24s stall"),
    # Tight at :400: `assert elapsed < 0.2` around a 0.03s join.
    ("tests/test_profile_stats.py", ":400 asserts elapsed < 0.2 around a 0.03s join"),
    # Timing-tight on a result at :1101 (patched _SWITCH_BUDGET_S = 0.3 vs a 1.98s cold path), and
    # it mutates keepwarm globals at :2628-2630.
    (
        "tests/test_media_auto_switch.py",
        ":1101 asserts started.is_set() under a patched 0.3s budget on a 1.98s cold path, "
        "and writes keepwarm globals at :2628-2630",
    ),
]

# Only clock-derived comparisons are scanned; timer races that assert results are not flagged.

# Below this an elapsed bound is within one scheduler quantum under -n 4 on four vCPUs.
TIGHT_BOUND_S = 0.1


def _over_the_backend(command: str) -> bool:
    return any(selection in command for selection in _selections("pytest"))


BACKEND_TESTS = Path(__file__).resolve().parents[2] / "studio" / "backend" / "tests"
_CLOCKS = ("monotonic", "perf_counter", "process_time", "time")


# Scan hits a human has read and found benign: sandwiches (`before <= t <= after`), poll
# deadlines, and sentinels. Keyed on the enclosing function, not a line number.
BENIGN_TIMING = {
    ("test_media_auto_switch.py", "_until"),
    # A 600 s poll deadline while llama-server loads the model: descheduling only delays the poll.
    ("test_decision_native_gpu.py", "_bare_llama_server"),
    # A 10 s poll deadline: descheduling only delays the poll, it cannot make the condition false.
    ("test_npu_chat_route.py", "_wait_for"),
    ("test_openai_auto_switch.py", "test_any_finished_download_drops_the_resolver_cache"),
    # A 600-second expiry vs the wall clock; late reads cannot make it false.
    (
        "test_openai_codex_subscription.py",
        "test_account_claim_and_token_response_are_validated_without_returning_raw_body",
    ),
    # A precondition: the snapshot is back-dated past the TTL, so descheduling only ages it more.
    (
        "test_account_local_model_resolver.py",
        "test_a_warm_scan_queues_behind_another_accounts_scan",
    ),
    # A poll deadline with a 30s budget.
    ("test_gpu_query_cache.py", "wait_for_call"),
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
    """A function counts if it returns any value containing its own timed names; run to a fixpoint."""
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    helpers: set = set()
    while True:
        grown = False
        for node in functions:
            if node.name in helpers:
                continue
            # A wrapper only counts as timed once `base` is known from an earlier pass.
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
    """Clock-holding names count as timed even when they hold a bare instant, not only a difference."""
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
    """A duration however spelled: a timed name, an inline clock difference, or a helper returning one."""
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
    """Scheduler-sensitive: absolute bounds at or below TIGHT_BOUND_S, or one duration vs another."""
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
            # Every adjacent pair: a chained `0.3 <= elapsed < 2.0` starts with a literal.
            operands = [cmp_node.left, *cmp_node.comparators]
            for index, op in enumerate(cmp_node.ops):
                lower, upper = operands[index], operands[index + 1]
                if isinstance(op, (ast.Gt, ast.GtE)):
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
    # The two jobs' sets must be disjoint or each job's isolation rules apply to the other.
    assert not set(root) & set(
        backend
    ), f"a command reads as belonging to both jobs: {set(root) & set(backend)}"
    assert len(parallel) == len(root) + len(backend), (
        f"a parallel run belongs to neither job's matrix, so nothing below checks it: "
        f"{[command for command in parallel if command not in root + backend]}"
    )
    # Line joins and matrix substitution must be resolved, or a command has no paths.
    assert all("${{" not in command for command in root + backend)
    assert any("--ignore=" in command for command in root)
    assert {command for command in root} == set(root), "a shard selection appears twice"
    assert len(set(backend)) == 3, "a backend shard selection appears twice"
    # Non-vacuous for the backend side too.
    assert any(_collects(command, "tests/test_account_contract.py") for command in backend)
    # Non-vacuous for the repo root too.
    assert any(_collects(command, "tests/test_model_registry.py") for command in root)
    assert len(commands) > 1, "no serial pytest steps found; the ignore checks cannot fail"


def test_the_backend_matrix_still_runs_in_parallel():
    """Asserts every backend pytest step keeps -n, since dropping it only makes CI slower with no signal."""
    backend = [command for command in _pytest_commands() if _over_the_backend(command)]
    assert backend, "the backend matrix pytest step is gone or was renamed past this scan"
    for command in backend:
        assert " -n " in f" {command} ", (
            f"a backend matrix shard is running serially again, which costs about 17 "
            f"minutes on every pull request and every push to main: {command}"
        )


@pytest.mark.parametrize("path, reason", BACKEND_ISOLATED, ids = [p for p, _ in BACKEND_ISOLATED])
def test_a_backend_isolated_path_is_ignored_by_the_parallel_run(path, reason):
    """Relative timing breaks under four workers on four vCPUs, so isolated files stay out of -n runs."""
    parallel = [
        command
        for command in _pytest_commands()
        if " -n " in f" {command} " and _over_the_backend(command)
    ]
    assert parallel, "the backend parallel run is gone or was renamed past this scan"
    # Asked of every shard as "does it reach the path", not "does it spell this --ignore".
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
    """Every file with a tight elapsed bound must be in BACKEND_ISOLATED, found by scanning, not review."""
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
    """Each timing shape is tested on a synthetic file, since an empty scan would pass vacuously."""
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

    # Quiet on a bound with headroom, or every anti-hang ceiling goes serial.
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

    # Nothing about the live suite: cleaning the last offender must not fail this.


def test_an_isolated_file_never_shadows_an_installed_library_with_a_stub():
    """Stubs stand in only for MISSING libraries: setdefault shadows an installed one not yet imported."""
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
    """Whether studio/backend itself provides this name, which a stub may deliberately replace."""
    return (BACKEND_TESTS.parent / name).is_dir() or (BACKEND_TESTS.parent / f"{name}.py").is_file()


def _is_installed(name: str) -> bool:
    """Repo names win over importlib: pytest can put studio/backend on sys.path, making them resolve."""
    if _is_repo_module(name):
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
