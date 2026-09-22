#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse dangerous GitHub Actions trigger patterns at PR time.

Bans patterns behind the TanStack GHSA-g7cv-rxg3-hmpx compromise:

1.  `pull_request_target` -- runs a fork's workflow against the base
    repo's secrets/permissions; use `pull_request` instead.
2.  `workflow_run` chained to a PR-triggered workflow -- same trust
    boundary problem one hop later (poisoned artifacts/caches run with
    elevated permissions).
3.  Cache keys shared between PR-triggered and publish/release/push
    workflows -- a fork PR could poison a cache the publish workflow
    restores. Partition the key namespaces.

Exit codes: 0 = no findings, 1 = findings (listed on stderr).
Run from repo root: python3 scripts/lint_workflow_triggers.py
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with 'pip install pyyaml'", file = sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Kept in one place because every reader below opens files the same way.
ENC = "utf-8"

DEFAULT_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

BANNED_TRIGGERS: tuple[str, ...] = ("pull_request_target",)
RESTRICTED_TRIGGERS: tuple[str, ...] = ("workflow_run",)
PUBLISH_WORKFLOW_STEMS: tuple[str, ...] = ("release-desktop",)

# The host must run on every PR and be able to fail.
LINT_SCRIPT_NAME = "lint_workflow_triggers.py"


def _normalise_on(on_field):
    if isinstance(on_field, str):
        return {on_field}
    if isinstance(on_field, list):
        return set(on_field)
    if isinstance(on_field, dict):
        return set(on_field.keys())
    return set()


def _load_workflow(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding = "utf-8"))
    except Exception as exc:
        print(f"ERROR: failed to parse {path}: {exc}", file = sys.stderr)
        sys.exit(2)


def _mappings(node):
    """Every mapping anywhere in a parsed document, at any depth."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _mappings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _mappings(item)


def _parse(path: Path):
    try:
        return yaml.safe_load(path.read_text(ENC))
    except Exception:
        return None


def _extract_cache_keys(path: Path) -> list[str]:
    """Every `key:` declared anywhere in the document.

    Read from the parsed structure rather than by scanning text for `key:`. The lexical
    version had to be taught one spelling at a time and never finished: `"restore-keys":`
    is valid YAML, so is `uses :`, so is flow style `- {uses: ...}`, and each spelling it
    did not know was a declaration the lint could not see at all. On a security check that
    is a bypass, not a rough edge.
    """
    doc = _parse(path)
    keys: list[str] = []
    for mapping in _mappings(doc):
        value = mapping.get("key")
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                keys.append(text)
    return keys


def _extract_restore_key_prefixes(path: Path) -> list[str]:
    """Every prefix a `restore-keys:` field offers as a fallback.

    The exact-key comparison is blind to these: `restore-keys` restores the newest entry
    whose key merely STARTS WITH the prefix, so a publish workflow can adopt an entry a
    pull request wrote without the two keys ever being equal.

    Taken from the parsed value, which is the string `actions/cache` itself receives, so
    the YAML details that decide the answer are decided by the parser and not
    re-implemented here. Both of them used to be wrong. A blank line inside a literal
    block does not end it, and breaking there dropped every prefix after it. A FOLDED
    block (`>`) is a single space-joined scalar, so `safe-only-` then `shared-` arrives
    as `safe-only- shared-` and offers no `shared-` fallback at all, while reading each
    physical line invented one -- a false rejection of a correct configuration. PyYAML
    gets both right by construction, and the sequence form (`restore-keys: [a-, b-]`),
    which the line reader never handled, comes free.
    """
    doc = _parse(path)
    prefixes: list[str] = []
    for mapping in _mappings(doc):
        value = mapping.get("restore-keys")
        if isinstance(value, str):
            prefixes.extend(line.strip() for line in value.splitlines())
        elif isinstance(value, list):
            prefixes.extend(str(item).strip() for item in value)
    return [p for p in prefixes if p]


def _local_uses(path: Path) -> list[str]:
    """Every `uses:` value in the document that points inside this repository."""
    doc = _parse(path)
    out: list[str] = []
    for mapping in _mappings(doc):
        value = mapping.get("uses")
        if isinstance(value, str) and value.strip().startswith("./"):
            out.append(value.strip())
    return out


def _call_sites(path: Path, target: str) -> list[dict]:
    """The `with:` mappings of every call to a local action or workflow named `target`.

    `target` is the last path component of the `uses: ./...` reference, which is the
    action's directory or the reusable workflow's filename.
    """
    doc = _parse(path)
    sites: list[dict] = []
    for mapping in _mappings(doc):
        value = mapping.get("uses")
        if not isinstance(value, str):
            continue
        ref = value.strip()
        if not ref.startswith("./"):
            continue
        stem = ref.rstrip("/").split("/")[-1]
        if stem != target and PurePosixPath(stem).stem != PurePosixPath(target).stem:
            continue
        with_ = mapping.get("with")
        sites.append(with_ if isinstance(with_, dict) else {})
    return sites


def _resolved_inputs(caller_paths: list, target: str) -> dict:
    """{input: (literal values, every call site resolved)} over all callers of `target`.

    Callers are read from every PR-reachable document, workflows AND composites, not
    just the top-level workflow files. A cache action reached through a wrapper action
    gets its inputs from that wrapper, and collecting only from workflows meant such a
    call site was invisible: if the workflow ALSO called the action directly with a
    literal, every input looked resolved and the narrowing below dropped the namespace
    the wrapper passes.

    The second element of each pair is what keeps the narrowing honest. A call site
    passing `name: ${{ matrix.cache_name }}`, or omitting the input so the action default
    applies, has no literal value here. Substituting only the literals its siblings pass
    would DISCARD that caller's namespace, turning a fix for a false rejection into a
    false acceptance, which is the worse of the two.

    A value that is itself a `steps.*` or `needs.*` reference counts as neither literal
    nor unresolved: it is genuine delegation. Every live caller of pip-cache-save and
    uv-cache-save passes `key: ${{ steps.pip-cache.outputs.key }}`, whose real namespace
    was already collected from the restoring action's shell, so reporting it as
    undecidable was a false failure on this tree -- and a check that fails on a correct
    configuration is one that gets switched off.
    """
    values: dict = {}
    seen_any = False
    for pth in caller_paths:
        for site in _call_sites(pth, target):
            seen_any = True
            for name, raw in site.items():
                literal = str(raw).strip().strip("'\"")
                bucket = values.setdefault(str(name), [set(), 0])
                if re.fullmatch(r"[A-Za-z0-9][\w.-]*", literal):
                    bucket[0].add(literal)
                elif _DELEGATED_KEY.fullmatch(literal):
                    pass
                else:
                    bucket[1] += 1
            for name, bucket in values.items():
                if name not in site:
                    # Omitted, so the action's default applies and is not visible here.
                    bucket[1] += 1
    if not seen_any:
        return {}
    return {name: (vals, unresolved == 0) for name, (vals, unresolved) in values.items()}


def _literal_prefix(key: str) -> str:
    """The fixed-text head of a key: everything before the first expression.

    Keys are mostly `literal-${{ something }}`, so comparing whole strings compares the
    expressions too and almost never matches.
    """
    return re.split(r"\$\{\{", key, maxsplit = 1)[0].strip().strip("'\"")


# `runner.os` is the only expression that routinely LEADS a cache key, and it takes
# exactly three values, so expanding it turns the common undecidable case into three
# decidable ones. Confirmed against GitHub's docs: the values are Linux, Windows and
# macOS, exact and case-sensitive.
_RUNNER_OS_VALUES = ("Linux", "Windows", "macOS")
_RUNNER_OS_EXPR = re.compile(r"\$\{\{\s*runner\.os\s*\}\}")

# A key that is nothing but one expression referring to a step output or an action input
# delegates its namespace rather than declaring one.
_DELEGATED_KEY = re.compile(r"\$\{\{\s*(steps|needs)\.[^}]*\}\}")

# `inputs.X` is NOT delegation: the value arrives from the caller, so it has to be
# resolved against the call sites rather than dismissed.
_INPUT_KEY = re.compile(r"\$\{\{\s*inputs\.([A-Za-z_][\w-]*)\s*\}\}")


def _prefix_candidates(key: str) -> list[str]:
    """Every literal head this key could have at runtime.

    A key beginning with an expression has no literal head at all, and dropping it was a
    hole: a PR writing `${{ runner.os }}-shared-abc` and a publish job restoring
    `Linux-shared-` would never be compared, because the PR side reduced to the empty
    string and was filtered out. Expanding `runner.os` first gives `Linux-shared-abc`,
    which is comparable. Anything still expression-led afterwards is genuinely
    undecidable and is reported rather than dropped.
    """
    keys = (
        [_RUNNER_OS_EXPR.sub(v, key) for v in _RUNNER_OS_VALUES]
        if _RUNNER_OS_EXPR.search(key)
        else [key]
    )
    return [h for h in (_literal_prefix(k) for k in keys) if h]


def _prefix_compatible(pr_head: str, publish_prefix: str) -> bool:
    """Can a key with this literal head be restored by this prefix?

    `restore-keys` matching is left-anchored and exact, with no globbing, so the two are
    compatible when either is a prefix of the other. The second direction is the one that
    was missing: a PR key `pip-v2-${{ runner.os }}-abc` reduces to the head `pip-v2-`,
    and a publish prefix `pip-v2-Linux-` is LONGER than that head, so a one-directional
    `head.startswith(prefix)` test says no while the runtime key `pip-v2-Linux-abc` does
    start with the prefix and would be restored.
    """
    return pr_head.startswith(publish_prefix) or publish_prefix.startswith(pr_head)


def _shell_built_key_prefixes(
    text: str,
    inputs: set | None = None,
    all_literal: bool = True,
) -> list[str]:
    """Literal key heads assembled in a composite action's shell, not in its YAML.

    The pip and uv caches build their key in a `run:` step and expose it as an output, so
    the YAML `key:` is only `${{ steps.probe.outputs.key }}` and carries no namespace at
    all. Reading YAML alone therefore learned nothing about the very composites this
    check exists to cover: pip-cache-restore's real namespace is the `pip-v2-` in
    `prefix="pip-v2-${name}-..."`, several lines away from any `key:`.

    `inputs` are the values callers actually pass for the first shell variable in such a
    prefix, which keeps the recorded namespace as narrow as the real one. See
    `_local_action_inputs` for why a broader one is not the safe direction.

    Only `key`-ish and `prefix`-ish variables are read. Taking every shell assignment
    would invent namespaces that no cache uses, and each invented one is a potential
    false rejection of a publish prefix.
    """
    heads: list[str] = []
    pattern = re.compile(
        r"""(?:^|[\s;(])(?:[A-Za-z_]*_)?(?:key|prefix|KEY|PREFIX)\s*=\s*["']?"""
        r"""([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""",
        re.M,
    )
    for m in pattern.finditer(text):
        heads.append(m.group(1))
    # `echo "key=pip-v2-${hash}" >> "$GITHUB_OUTPUT"` is the same thing written inline.
    for m in re.finditer(
        r"""echo\s+["']?(?:key|prefix)=([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""", text
    ):
        heads.append(m.group(1))
    if inputs and all_literal:
        return [f"{h}{v}-" for h in heads for v in sorted(inputs)]
    # An unresolved call site means the broad head still has to be carried, or narrowing
    # would silently drop the namespace that caller writes.
    return heads + [f"{h}{v}-" for h in heads for v in sorted(inputs or ())]


def _pr_reachable_action_dirs(workflows_dir: Path, pr_paths: list) -> set:
    """Composite actions a PR-triggered workflow actually uses.

    Scanning every action under .github/actions treated a publish-only composite's keys
    as a namespace pull requests write, so a publish workflow restoring its OWN action's
    prefix was rejected as PR-poisonable. That is a false failure on a safe
    configuration, and a security lint that cries wolf gets switched off.

    Local reusable WORKFLOWS are followed too, not just composite actions. A job-level
    `uses: ./.github/workflows/shared.yml` names the workflow file itself, so probing
    only for an `action.yml` beneath the reference found nothing and the keys that
    workflow declares stayed outside the comparison entirely -- reachable from a pull
    request in fact, invisible to the check.
    """
    root = workflows_dir.parent
    dirs: set = set()
    seen: set = set()
    queue = [pth for pth in pr_paths]
    while queue:
        pth = queue.pop()
        if pth in seen:
            continue
        seen.add(pth)
        for ref in _local_uses(pth):
            cand = root.parent / ref[2:]
            # A local reusable workflow reference names the .yml file itself rather than a
            # directory containing an action.yml, so it has to be followed on its own.
            if cand.is_file() and cand.suffix in (".yml", ".yaml"):
                dirs.add(cand)
                queue.append(cand)
                continue
            for action in (cand / "action.yml", cand / "action.yaml"):
                if action.is_file():
                    dirs.add(action)
                    queue.append(action)
    return dirs


def _on_field(yaml_doc):
    # PyYAML parses a bare `on:` key as True.
    on = yaml_doc.get(True) if isinstance(yaml_doc, dict) else None
    if on is None and isinstance(yaml_doc, dict):
        on = yaml_doc.get("on")
    return on


def _trigger_set(yaml_doc) -> set[str]:
    return _normalise_on(_on_field(yaml_doc))


# Accept only a plain invocation of this script; fail closed on wrappers.
_PYTHON_BASENAME = re.compile(r"python(3(\.\d+)?)?")
# Allow only flags that preserve script execution.
_SAFE_OPTS = ("-u", "-E", "-s", "-S", "-B", "-O", "-OO", "-q")
LINT_SCRIPT_PATH = f"scripts/{LINT_SCRIPT_NAME}"
# These options consume the next token.
_OPTS_WITH_VALUE = ("-X", "-W", "--check-hash-based-pycs")
_SHELL_OPERATORS = ("|", "&", ";", ">", "<", "`", "$(")


def _is_trusted_python(token: str) -> bool:
    """A bare `python3`, or an absolute system path to one.

    A relative `./python3` would resolve inside the checkout, where a PR can
    add an executable of that name.
    """
    if any(op in token for op in _SHELL_OPERATORS):
        return False  # a substitution runs before the path is used
    path = PurePosixPath(token)
    if not _PYTHON_BASENAME.fullmatch(path.name):
        return False
    return token == path.name or token.startswith(("/usr/", "/bin/", "/opt/"))


def _classify_lint_line(line: str) -> tuple[bool, str | None]:
    """(is an enforcing invocation, problem) for one line naming the script."""
    try:
        tokens = shlex.split(line.strip())
    except ValueError:
        return False, None
    if not tokens or not _is_trusted_python(tokens[0]):
        return False, None  # `echo <script>`, a decoy interpreter, or not python

    args, i = tokens[1:], 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] in _OPTS_WITH_VALUE:
            value = args[i + 1] if i + 1 < len(args) else ""
            if any(op in value for op in _SHELL_OPERATORS):
                return False, None  # a substitution runs before python
            i += 2
        elif args[i] in _SAFE_OPTS:
            i += 1
        else:
            return False, None

    rest = args[i:]
    # Require the repository-relative path, not a suffix match.
    if not rest or rest[0] not in (LINT_SCRIPT_PATH, f"./{LINT_SCRIPT_PATH}"):
        return False, None
    if len(rest) == 1:
        return True, None

    trailing = " ".join(rest[1:])
    if any(op in tok for tok in rest[1:] for op in _SHELL_OPERATORS):
        return False, (
            f"its lint command is chained, piped or backgrounded ({trailing}), "
            "so the step's exit status need not be the lint's"
        )
    return False, (
        f"its lint command passes {trailing}, so it does not gate the live "
        "workflows with its own checks on"
    )


def _lint_step_report(run: str) -> tuple[bool, list[str]]:
    """Return whether the step enforces the lint and any problems found."""
    lines = [line for line in run.splitlines() if line.strip() and not line.strip().startswith("#")]
    mentions = [line for line in lines if LINT_SCRIPT_NAME in line]
    if not mentions:
        return False, []

    problems = [p for _, p in map(_classify_lint_line, mentions) if p]
    enforcing = any(ok for ok, _ in map(_classify_lint_line, mentions))
    if enforcing and len(lines) > 1:
        return False, problems + [
            "its lint step runs other shell besides the lint command, so the "
            "lint need not execute (a function body or here-document is not a "
            "call)"
        ]
    return enforcing, problems


# Shell templates can wrap the command and hide its status.
SAFE_SHELLS: tuple[str, ...] = ("bash", "sh")

# These variables can redirect execution before the script runs.
UNSAFE_ENV_KEYS: tuple[str, ...] = (
    "BASH_ENV",
    "ENV",
    "PATH",
    # `sitecustomize.py` on PYTHONPATH is imported before the script runs.
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
)


def _effective_run_setting(yaml_doc, job: dict, step: dict, key: str) -> str | None:
    """A step's `run` setting, falling back to job then workflow defaults."""
    if step.get(key):
        return str(step[key])
    for scope in (job, yaml_doc):
        defaults = scope.get("defaults") if isinstance(scope, dict) else None
        run = defaults.get("run") if isinstance(defaults, dict) else None
        if isinstance(run, dict) and run.get(key):
            return str(run[key])
    return None


def _effective_env(yaml_doc, job: dict, step: dict) -> dict:
    """Workflow, job and step `env`, merged in precedence order."""
    merged = {}
    for scope in (yaml_doc, job, step):
        env = scope.get("env") if isinstance(scope, dict) else None
        if isinstance(env, dict):
            merged.update(env)
    return merged


def _pull_request_config_problem(yaml_doc) -> str | None:
    """`pull_request:` must be bare or a mapping; GitHub rejects anything else."""
    on = _on_field(yaml_doc)
    if not isinstance(on, dict) or "pull_request" not in on:
        return None
    pr = on["pull_request"]
    if pr is None or isinstance(pr, dict):
        return None
    return (
        f"its 'pull_request' value is {pr!r}, which is not a valid event "
        "configuration, so GitHub will not load the workflow at all"
    )


def _lint_steps(yaml_doc) -> list[tuple[dict, dict, bool, list[str]]]:
    """Return every step that runs the lint and its enforcement status."""
    jobs = yaml_doc.get("jobs") if isinstance(yaml_doc, dict) else None
    if not isinstance(jobs, dict):
        return []
    found = []
    for job in jobs.values():
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            enforcing, problems = _lint_step_report(str(step.get("run") or ""))
            if not (enforcing or problems):
                continue
            if job.get("container") is not None:
                enforcing = False
                problems.append(
                    "its lint job runs in a 'container:', a PR-selected image "
                    "that controls the shell and environment"
                )
            shell = _effective_run_setting(yaml_doc, job, step, "shell")
            if shell is not None and shell not in SAFE_SHELLS:
                enforcing = False
                problems.append(
                    f"its lint step runs under shell {shell!r}, which can wrap "
                    "the command and drop its exit status"
                )
            unsafe_env = sorted(
                k for k in _effective_env(yaml_doc, job, step) if k in UNSAFE_ENV_KEYS
            )
            if unsafe_env:
                enforcing = False
                problems.append(
                    f"its lint step sets {' + '.join(unsafe_env)}, which can "
                    "redirect the step before the lint runs"
                )
            workdir = _effective_run_setting(yaml_doc, job, step, "working-directory")
            if workdir is not None:
                enforcing = False
                problems.append(
                    f"its lint step runs in working-directory {workdir!r}, so "
                    "the command resolves to a different file than this "
                    "repository's script"
                )
            found.append((job, step, enforcing, problems))
    return found


def _pull_request_restrictions(yaml_doc) -> list[str]:
    """Keys narrowing the `pull_request` trigger. A gate wants none of them."""
    on = _on_field(yaml_doc)
    pr = on.get("pull_request") if isinstance(on, dict) else None
    return sorted(pr) if isinstance(pr, dict) else []


def _is_truthy(value) -> bool:
    """YAML truthiness, treating any `${{ ... }}` expression as possibly true."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() not in ("", "false")


def main() -> int:
    # Do not let abbreviated options bypass the gate.
    parser = argparse.ArgumentParser(description = __doc__, allow_abbrev = False)
    parser.add_argument(
        "--workflows-dir",
        type = Path,
        default = DEFAULT_WORKFLOWS_DIR,
        help = "Override the workflows directory (used by tests).",
    )
    parser.add_argument(
        "--require-host",
        action = "store_true",
        default = None,
        help = "Require a workflow that runs this script on unfiltered "
        "`pull_request`. Defaults on for the live tree, off for a "
        "fixture directory.",
    )
    parser.add_argument(
        "--no-require-host",
        dest = "require_host",
        action = "store_false",
        help = "Skip the host-wiring check.",
    )
    args = parser.parse_args()
    workflows_dir = args.workflows_dir

    require_host = args.require_host
    if require_host is None:
        require_host = workflows_dir.resolve() == DEFAULT_WORKFLOWS_DIR.resolve()

    findings: list[str] = []
    workflows = sorted(list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")))
    pr_triggered: list[tuple[Path, list[str]]] = []
    publish_triggered: list[tuple[Path, list[str]]] = []
    publish_restore_prefixes: list[tuple[Path, list[str]]] = []
    unfiltered_hosts: list[Path] = []

    for path in workflows:
        doc = _load_workflow(path)
        triggers = _trigger_set(doc)

        for t in BANNED_TRIGGERS:
            if t in triggers:
                findings.append(
                    f"{path.name}: BANNED trigger '{t}' (GHSA-g7cv-rxg3-hmpx "
                    "pattern: fork PRs run in base-repo context). Switch to "
                    "'pull_request' and use a deploy-on-merge workflow for "
                    "any privileged step."
                )

        for t in RESTRICTED_TRIGGERS:
            if t in triggers:
                text = path.read_text(encoding = "utf-8")
                if "lint:workflow_triggers-allow-workflow_run" not in text:
                    findings.append(
                        f"{path.name}: RESTRICTED trigger '{t}' requires an "
                        "explicit `# lint:workflow_triggers-allow-workflow_run` "
                        "comment somewhere in the file, with a justification."
                    )

        lint_steps = _lint_steps(doc)
        if lint_steps:
            problems = []
            restrictions = _pull_request_restrictions(doc)
            if restrictions:
                problems.append(
                    f"its 'pull_request' trigger is narrowed by "
                    f"{' + '.join(restrictions)}, so a PR adding that skips "
                    "this workflow for its own PR"
                )
            config_problem = _pull_request_config_problem(doc)
            if config_problem:
                problems.append(config_problem)
            if any(
                _is_truthy(job.get("continue-on-error"))
                or _is_truthy(step.get("continue-on-error"))
                for job, step, _, _ in lint_steps
            ):
                problems.append(
                    "its lint step is continue-on-error, so findings cannot fail the run"
                )
            if any(
                job.get("if") is not None or step.get("if") is not None
                for job, step, _, _ in lint_steps
            ):
                problems.append(
                    "its lint step is gated by an 'if:' condition, so the gate "
                    "can be skipped while the run still succeeds"
                )
            if any(job.get("needs") is not None for job, _, _, _ in lint_steps):
                problems.append(
                    "its lint job declares 'needs:', so a skipped prerequisite "
                    "skips the gate without failing the run"
                )
            for _, _, _, step_problems in lint_steps:
                problems.extend(step_problems)
            if problems:
                findings.append(
                    f"{path.name}: runs {LINT_SCRIPT_NAME} but "
                    + ", and ".join(problems)
                    + ". The gate must run, and be able to fail, on every PR."
                )
            elif "pull_request" in triggers and any(enforcing for _, _, enforcing, _ in lint_steps):
                unfiltered_hosts.append(path)

        if "pull_request" in triggers:
            pr_triggered.append((path, _extract_cache_keys(path)))
        is_dispatch_only = "workflow_dispatch" in triggers and not (
            "push" in triggers or "pull_request" in triggers
        )
        if path.stem in PUBLISH_WORKFLOW_STEMS or is_dispatch_only:
            publish_triggered.append((path, _extract_cache_keys(path)))
            publish_restore_prefixes.append((path, _extract_restore_key_prefixes(path)))

    if require_host and not unfiltered_hosts:
        findings.append(
            f"no workflow runs {LINT_SCRIPT_NAME} on an unfiltered "
            "'pull_request' trigger, so this gate does not cover every PR. "
            "Restore the workflow-trigger-lint workflow."
        )

    # A PR-triggered workflow usually delegates its key to a composite action, so the
    # literal key lives in .github/actions/*/action.yml and the workflow only carries
    # `${{ steps.x.outputs.key }}`. Those count as PR-reachable: the workflow that uses
    # them runs on pull requests. Without this the prefix rule below would compare
    # against opaque expressions and match nothing.
    pr_workflow_paths = [pth for pth, _ in pr_triggered]
    pr_reachable = sorted(_pr_reachable_action_dirs(workflows_dir, pr_workflow_paths))
    # Call sites come from every reachable document, workflows and composites alike, so a
    # cache action reached through a wrapper action is narrowed by what the WRAPPER passes
    # and not only by what a workflow passes directly.
    pr_callers = pr_workflow_paths + pr_reachable
    composite_keys: list[str] = []
    for action_path in pr_reachable:
        composite_keys.extend(_extract_cache_keys(action_path))
        resolved = _resolved_inputs(pr_callers, action_path.parent.name)
        names, all_literal = resolved.get("name", (set(), False))
        composite_keys.extend(
            _shell_built_key_prefixes(action_path.read_text(ENC), names, all_literal)
        )

    # The publish side delegates to local actions exactly as the pull-request side does,
    # and reading only the top-level workflow file left that half unexamined. A publish
    # workflow whose composite holds the `actions/cache/restore` declares its keys and its
    # `restore-keys` in the action, so a PR writing `shared-*` against a publish-only
    # composite restoring `shared-` passed: the prefix was never collected, and a
    # comparison that collects nothing on one side reports success.
    for action_path in sorted(
        _pr_reachable_action_dirs(workflows_dir, [pth for pth, _ in publish_triggered])
    ):
        publish_triggered.append((action_path, _extract_cache_keys(action_path)))
        publish_restore_prefixes.append((action_path, _extract_restore_key_prefixes(action_path)))

    # Composite keys belong in the exact comparison as well, not only the prefix one. A
    # PR-reachable action declaring `key: shared-key`, against a publish workflow using
    # that same key and no restore-keys at all, is the original cache-poisoning shape, and
    # it was invisible while this set held workflow-declared keys only.
    pr_keys = {key for _, keys in pr_triggered for key in keys} | set(composite_keys)
    for pub_path, pub_keys in publish_triggered:
        for k in pub_keys:
            if k in pr_keys:
                findings.append(
                    f"{pub_path.name}: cache key {k!r} is also declared in a "
                    "PR-triggered workflow. A fork PR could poison this cache "
                    "and the publish workflow would restore it on next run. "
                    "Add a unique suffix (e.g. '-publish-only') to partition "
                    "the namespaces."
                )

    # Same trust boundary, reached by prefix instead of by an equal key. `restore-keys`
    # restores the newest entry whose key merely STARTS WITH the prefix, so a publish
    # workflow can adopt an entry a pull request wrote without the two keys ever being
    # equal, which is the only thing the check above compares.
    pr_heads: set = set()
    undecidable_pr_keys: list[str] = []
    # Inputs any PR-reachable target is called with, so a `key: ${{ inputs.X }}` can be
    # resolved to the namespace its callers actually produce.
    input_namespaces: dict = {}
    for target in pr_reachable:
        for field, pair in _resolved_inputs(
            pr_callers, target.parent.name if target.name.startswith("action.") else target.name
        ).items():
            vals, ok = input_namespaces.get(field, (set(), True))
            input_namespaces[field] = (vals | pair[0], ok and pair[1])
    for k in list(pr_keys) + composite_keys:
        cands = _prefix_candidates(k)
        if cands:
            pr_heads.update(cands)
        elif _INPUT_KEY.fullmatch(k.strip()):
            # `key: ${{ inputs.cache_key }}` in a reusable workflow or composite names no
            # namespace here, but unlike a `steps.*` reference it is not delegation
            # either: the value comes from the CALLER, and a pull request caller passing
            # `cache_key: shared-abc` writes the `shared-` namespace. Treating it as
            # delegation dropped it silently and a publish `shared-` fallback passed.
            field = _INPUT_KEY.fullmatch(k.strip()).group(1)
            vals, resolved = input_namespaces.get(field, (set(), False))
            heads = [h for v in sorted(vals) for h in _prefix_candidates(v)]
            pr_heads.update(heads)
            if not resolved:
                # Some call site passes a value this check cannot expand, such as
                # `${{ matrix.cache_name }}`, or omits the input so the action default
                # applies. Say so rather than assuming it is harmless. Resolved with no
                # literals is the DELEGATION case and is fine: every live caller of
                # pip-cache-save passes `key: ${{ steps.pip-cache.outputs.key }}`, whose
                # real namespace was collected from the restoring action's shell.
                undecidable_pr_keys.append(k)
        elif _DELEGATED_KEY.fullmatch(k.strip()):
            # `key: ${{ steps.pip-cache.outputs.key }}` names no namespace of its own; it
            # hands the decision to a composite action, whose real prefix was collected
            # above from that action's own YAML and shell. Reporting it as undecidable
            # would flag every workflow that factors its cache out into an action.
            continue
        else:
            undecidable_pr_keys.append(k)

    for pub_path, prefixes in publish_restore_prefixes:
        for prefix in prefixes:
            pub_heads = _prefix_candidates(prefix)
            if not pub_heads:
                findings.append(
                    f"{pub_path.name}: restore-keys entry {prefix!r} begins with an "
                    "expression, so what it can restore is not decidable here. Give it a "
                    "literal prefix."
                )
                continue
            for pub_head in pub_heads:
                hit = next((h for h in sorted(pr_heads) if _prefix_compatible(h, pub_head)), None)
                if hit is not None:
                    findings.append(
                        f"{pub_path.name}: restore-keys prefix {pub_head!r} matches "
                        f"{hit!r}, a cache key namespace a PR-triggered workflow writes. "
                        "A prefix restore takes the newest matching entry, so this "
                        "publish workflow could adopt a cache a pull request produced "
                        "even though no key is equal. Partition the namespaces, or drop "
                        "the restore-keys fallback on the publish side."
                    )
                    break
            else:
                # Only reachable when nothing matched: an undecidable PR key could still
                # expand into this prefix, so say so rather than passing silently.
                for k in undecidable_pr_keys:
                    findings.append(
                        f"{pub_path.name}: restore-keys prefix {prefix!r} cannot be "
                        f"compared against PR cache key {k!r}, which begins with an "
                        "expression this check cannot expand, so whether the prefix "
                        "reaches that namespace is undecidable. Give the PR key a "
                        "literal prefix."
                    )
                    break

    if findings:
        print("Workflow trigger lint failed with the following issues:", file = sys.stderr)
        for f in findings:
            print(f"  - {f}", file = sys.stderr)
        return 1

    print(
        f"OK: scanned {len(workflows)} workflow file(s); "
        f"no pull_request_target, no unjustified workflow_run, "
        f"no PR/publish cache-key collision."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
