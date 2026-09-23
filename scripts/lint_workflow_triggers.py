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
    sites: list = []
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    scoped = (
        [(str(jid), job) for jid, job in jobs.items()]
        if isinstance(jobs, dict)
        else [("", doc)]
    )
    for job_id, container in scoped:
      for mapping in _mappings(container):
        value = mapping.get("uses")
        if not isinstance(value, str):
            continue
        ref = value.strip()
        if not ref.startswith("./"):
            continue
        # Compared as the FULL local reference when one is given, because two actions
        # may share a directory name: `./.github/actions/a/cache` and
        # `./.github/actions/b/cache` both end in `cache`, so matching on the last
        # component pooled their call sites and invented one action's value for the
        # other. A bare name is still accepted, which is what the predicate self-tests
        # and the shell-head narrowing pass.
        full = ref[2:].rstrip("/") if ref.startswith("./") else ref.rstrip("/")
        if "/" in target:
            if full != target and PurePosixPath(full).parent.as_posix() != target:
                continue
        else:
            stem = full.split("/")[-1]
            if stem != target and PurePosixPath(stem).stem != PurePosixPath(target).stem:
                continue
        with_ = mapping.get("with")
        # The CALLER's scope travels with the call site, because a value it delegates
        # names a step of the caller's job, not of the target. Resolving
        # `${{ steps.pip-cache.outputs.key }}` in the target's scope looked for a step
        # the target does not have, so this repository's own pip-cache-save read as
        # unresolvable and failed the live tree.
        sites.append((
            with_ if isinstance(with_, dict) else {},
            (path.as_posix(), job_id),
        ))
    return sites



_STEP_OUTPUT = re.compile(
    r"\$\{\{\s*steps\.([A-Za-z_][\w-]*)\.outputs\.[A-Za-z_][\w-]*\s*\}\}"
)


_CACHE_KEY_OUTPUTS = ("cache-primary-key", "cache-matched-key", "key")



def _scoped_cache_keys(path: Path) -> list:
    """[(job id, key)] so a key carries the scope its step ids are resolved in.

    Step ids are unique only within a job, so a key referring to
    `${{ steps.probe.outputs.key }}` means THIS job's `probe`. Without the scope the
    only safe reading was "every step of that id, anywhere, must be readable", and this
    repository has three unrelated `id: probe` steps -- two of which emit no cache key
    at all -- so the safe reading failed a correct tree. The scope makes the question
    answerable instead of merely conservative.
    """
    doc = _parse(path)
    out = []
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    if isinstance(jobs, dict):
        for job_id, job in jobs.items():
            for mapping in _mappings(job):
                key = mapping.get("key")
                if key is not None and not isinstance(key, (dict, list)):
                    # Stringified, as `_extract_cache_keys` already does. Accepting only
                    # `str` dropped an unquoted scalar such as `key: 123`, which YAML
                    # parses as an int, so two workflows sharing that key were reported
                    # as collision-free.
                    out.append((str(job_id), str(key)))
        return out
    for mapping in _mappings(doc):
        key = mapping.get("key")
        if key is not None and not isinstance(key, (dict, list)):
            out.append(("", str(key)))
    return out


def _producer_steps(path: Path) -> dict:
    """{(document, job, step id): was its key output actually recovered}.

    Keyed by the whole scope, because a step id is unique only within its job. Merging
    bare ids let a readable `id: probe` in one job -- or in a later-sorted file -- stand
    in for an unreadable `id: probe` in another, and the unread key was then dismissed
    on the strength of a step that has nothing to do with it. The dictionary is built
    globally, so a bare id was exactly the wrong identity.

    Per producer, because "some producer somewhere was readable" is not evidence about
    THIS one. A side-wide flag let a workflow with one ordinary
    `echo 'key=safe-key' >> "$GITHUB_OUTPUT"` vouch for a second step emitting
    `printf 'key=%s\n' "shared-$GITHUB_SHA"`, whose namespace was never recovered: the
    flag was true, the delegated key was dismissed, and a publish `restore-keys:
    shared-` passed. One readable step is not a warrant for an unreadable one.
    """
    out: dict = {}
    doc = _parse(path)
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    scopes = []
    if isinstance(jobs, dict):
        scopes = [(str(jid), job) for jid, job in jobs.items()]
    else:
        # An action definition has no jobs; its steps share one scope.
        scopes = [("", doc)]
    for job_id, scope in scopes:
      for mapping in _mappings(scope):
        step_id = mapping.get("id")
        if not isinstance(step_id, str):
            continue
        ident = (path.as_posix(), job_id, step_id)
        # A step that declares its own `key:` publishes it as an output
        # (`cache-primary-key`), so the namespace is right there in the YAML. This is
        # how `actions/cache/restore` steps hand a key onward, and reading only `run:`
        # bodies and local composites declared them unreadable.
        with_ = mapping.get("with")
        uses_value = str(mapping.get("uses") or "")
        if (
            isinstance(with_, dict)
            and with_.get("key") is not None
            and uses_value.strip().split("@")[0].casefold().startswith("actions/cache")
        ):
            # Only for an action whose published key output IS this input. Marking every
            # id-bearing step with a `with.key` readable was too generous: a local action
            # may take an unrelated `key` input while emitting its own `outputs.key` from
            # a command this check cannot read, and the delegated key was then dismissed
            # with nothing recovered.
            out[ident] = True
            continue
        body = mapping.get("run")
        if isinstance(body, str):
            out[ident] = bool(
                _shell_built_key_prefixes(body) or _shell_output_keys(body)
            )
            continue
        # A step may produce its output by CALLING a local action rather than by running
        # a shell body -- which is how this repository's own pip-cache-restore works, and
        # reading `run:` steps alone declared that producer unreadable and failed the
        # live tree. The action's own shell is what has to be readable.
        uses = mapping.get("uses")
        if not isinstance(uses, str) or not uses.strip().startswith("./"):
            continue
        # The repository root, found by walking up to `.github` rather than counting
        # levels. A workflow sits at `.github/workflows/x.yml` and an action at
        # `.github/actions/<name>/action.yml`, one level deeper, so a fixed three-up
        # landed inside `.github` for every action.
        root = path.parent
        for parent in path.parents:
            if parent.name == ".github":
                root = parent.parent
                break
        candidates = []
        for target in _local_ref_candidates(uses, root):
            candidates += [target / "action.yml", target / "action.yaml", target]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            text = candidate.read_text(ENC)
            # YAML-declared keys count as recovered too, and are in fact the usual case:
            # this repository's frontend-dist-restore hands out
            # `steps.restore.outputs.cache-primary-key`, whose value is the `key:` the
            # action declares. Requiring a SHELL-built key declared that producer
            # unreadable and failed the live tree on a correct configuration.
            out[ident] = _local_action_publishes_a_key(candidate, text)
            break
    return out



def _local_action_publishes_a_key(action: Path, text: str) -> bool:
    """Does this local action hand out a key whose value this check can see?

    Its declared `outputs.<name>.value` is what a caller receives. When that value is a
    step output, the step behind it has to be readable; when the action has no declared
    outputs at all, the question does not arise and the shell is the only evidence.
    Accepting any action that merely mentions a cache key let one emitting its
    `outputs.key` through an unreadable command vouch for itself.
    """
    doc = _parse(action)
    outputs = doc.get("outputs") if isinstance(doc, dict) else None
    inner = _producer_steps(action)
    inner_by_id = {ident[2]: ok for ident, ok in inner.items()}
    if isinstance(outputs, dict):
        for name, spec in outputs.items():
            if str(name) not in _CACHE_KEY_OUTPUTS:
                continue
            value = spec.get("value") if isinstance(spec, dict) else None
            match = _STEP_OUTPUT.search(str(value or ""))
            if match is None:
                # Not delegated to a step: the value is literal or an expression this
                # check reads no further into, so the shell is the evidence.
                return bool(
                    _shell_built_key_prefixes(text) or _shell_output_keys(text)
                )
            return bool(inner_by_id.get(match.group(1)))
    return bool(
        _extract_cache_keys(action)
        or _shell_built_key_prefixes(text)
        or _shell_output_keys(text)
    )


def _delegation_is_read(expression: str, producers: dict, scope = None) -> bool:
    """Was the step this expression names one whose key output we could read?

    An expression naming no step at all -- `${{ needs.build.outputs.key }}` -- is not
    something this check can follow, so it is not evidence either.
    """
    match = _STEP_OUTPUT.search(expression or "")
    if match is None:
        return False
    step_id = match.group(1)
    if scope is not None:
        return bool(producers.get((scope[0], scope[1], step_id)))
    # Without a scope, EVERY step of that id has to be readable. A single unreadable
    # namesake anywhere is enough to make the answer no, which is the conservative
    # reading and the one that stops a readable namesake vouching for it.
    seen = [ok for ident, ok in producers.items() if ident[2] == step_id]
    return bool(seen) and all(seen)


def _resolved_inputs(caller_paths: list, target: str, _depth: int = 0) -> dict:
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
    sites: list = []
    for pth in caller_paths:
        sites.extend(_call_sites(pth, target))
    if not sites:
        return {}
    # Every input named anywhere, collected BEFORE counting omissions. Doing both in one
    # pass made the answer depend on call-site order: a site that omitted an input had no
    # bucket yet, so the omission went unrecorded, and a later site supplying a literal
    # made the input look fully resolved. Verified before fixing -- a composite called
    # first with no `name` and then with `name: safe` reported ({'safe'}, True), so the
    # namespace narrowed to `safe` and the real default namespace was left undefended.
    names = {str(name) for site, _scope in sites for name in site}
    # {input name: {(wrapper's own input, wrapper scope)}} -- values a wrapper forwards
    # from its own inputs, resolvable against that wrapper's call sites.
    forwarded_inputs: dict = {}
    for name in names:
        literals: set = set()
        delegated: set = set()
        omitted = 0
        dynamic = 0
        for site, site_scope in sites:
            if name not in site:
                # Omitted, so the action's default applies. A declared default settles
                # THIS, and only this.
                omitted += 1
                continue
            literal = str(site[name]).strip().strip("'\"")
            if re.fullmatch(r"[A-Za-z0-9][\w.-]*", literal):
                literals.add(literal)
            elif _DELEGATED_KEY.fullmatch(literal):
                # Kept, not discarded. Which producer supplies the value decides
                # whether the delegation settles anything, and that is only knowable
                # from the expression the call site wrote.
                delegated.add((literal, site_scope))
            else:
                forwarded = _INPUT_KEY.fullmatch(literal)
                if forwarded is not None and site_scope is not None:
                    # A WRAPPER forwarding its own input: `with: {name: ${{ inputs.name }}}`
                    # inside another composite. The value is whatever that wrapper's own
                    # callers pass, so it is resolvable one level up rather than unknown,
                    # and counting it as dynamic left a nested composite undecidable and
                    # rejected an unrelated publish key -- a false failure on a valid
                    # arrangement.
                    #
                    # Recorded for the caller to resolve rather than followed here, which
                    # would mean recursing while already inside the resolution it needs.
                    forwarded_inputs.setdefault(name, set()).add(
                        (forwarded.group(1), site_scope)
                    )
                    continue
                # An explicit value this check cannot expand, such as
                # `${{ matrix.cache_key }}`. No default can settle it, because the caller
                # overrode the default with something unknown.
                dynamic += 1
        for wrapper_input, wrapper_scope in forwarded_inputs.get(name, ()):
            # Resolved against the WRAPPER'S OWN callers, which is the whole point: the
            # forwarded value is whatever they pass it. Resolving the wrapper file
            # against itself looked for call sites inside the wrapper and found none.
            #
            # One level up, and one level only. A wrapper forwarding a wrapper is rare
            # enough that the extra depth is not worth the recursion, and `_depth` makes
            # the fallback the conservative one -- the value stays unresolved.
            outer = (
                {}
                if _depth
                else _resolved_inputs(
                    caller_paths, _target_name(Path(wrapper_scope[0])), _depth + 1
                )
            ).get(wrapper_input)
            if outer and outer[1] and outer[0]:
                literals |= outer[0]
            else:
                dynamic += 1
        values[name] = (
            literals, dynamic == 0 and omitted == 0, omitted, dynamic, delegated,
        )
    return values


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
    keys = [_RUNNER_OS_EXPR.sub(v, key) for v in _RUNNER_OS_VALUES] if _RUNNER_OS_EXPR.search(key) else [key]
    return [h for h in (_literal_prefix(k) for k in keys) if h]


def _is_truncated(key: str) -> bool:
    """Did this key continue into an expression past its literal head?

    Only then can a publish prefix LONGER than the head still reach the runtime key.
    """
    keys = [_RUNNER_OS_EXPR.sub(v, key) for v in _RUNNER_OS_VALUES] if _RUNNER_OS_EXPR.search(key) else [key]
    return any("${{" in k for k in keys)

def _prefix_compatible(pr_head: str, publish_prefix: str, truncated: bool = True) -> bool:
    """Can a key with this literal head be restored by this prefix?

    `restore-keys` matching is left-anchored and exact, with no globbing, so a publish
    prefix restores a PR-written entry when the PR's runtime key starts with it.

    A head is TRUNCATED when the key continued into an expression this check cannot
    expand, and that is the only case where the reverse direction holds. A PR key
    `pip-v2-${{ runner.os }}-abc` reduces to the head `pip-v2-`, and the publish prefix
    `pip-v2-Linux-` is longer than that head, yet the runtime key `pip-v2-Linux-abc` does
    start with it, so the pairing has to be treated as compatible.

    For a head that is the WHOLE key, the reverse direction is simply wrong. A PR key
    that is exactly `shared` is saved as `shared`, and `shared`.startswith(`shared-long`)
    is false, so a publish fallback `shared-long` cannot reach it. Allowing the reverse
    unconditionally rejected every longer fallback that merely shared an opening with a
    complete key, which is a false failure on a correct configuration.
    """
    if pr_head.startswith(publish_prefix):
        return True
    return truncated and publish_prefix.startswith(pr_head)


def _shell_output_keys(text: str) -> list:
    """Fully literal `key=<value>` values written to `$GITHUB_OUTPUT`.

    The companion to `_shell_built_key_prefixes`, which recovers the literal HEAD of a
    key whose tail is assembled at runtime and therefore finds nothing in a key that is
    literal all the way through. Both are evidence that the producer was read; a step
    emitting `key=own-v1-abc` is completely resolved, and treating "no dynamic head" as
    "producer not understood" failed a correct configuration.

    These are exact keys, so they join the comparison rather than only vouching for it.
    """
    out = []
    for line in text.splitlines():
        # A commented-out line executes nothing, and reading one as a recovered output
        # let an otherwise unreadable producer certify itself: a `# echo 'key=safe-key'`
        # above a real `printf 'key=%s\n' "shared-$GITHUB_SHA"` marked the step readable,
        # its delegated key was dismissed, and a publish `restore-keys: shared-` passed.
        stripped = line.split("#", 1)[0]
        if "GITHUB_OUTPUT" not in stripped:
            continue
        for value in re.findall(r"\bkey=([A-Za-z0-9][A-Za-z0-9._-]*)", stripped):
            if "$" not in value and value not in out:
                out.append(value)
    return out


def _shell_built_key_prefixes(
    text: str, inputs: set | None = None, all_literal: bool = True,
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
    # A shell value only becomes a cache key by leaving the step, and `$GITHUB_OUTPUT` is
    # how it leaves. Every assignment named `key` or `prefix` was being treated as a cache
    # namespace regardless, so an unrelated `key="shared-${RANDOM}"` in a workflow that
    # caches nothing made a publish `restore-keys: shared-` fail the hard gate.
    #
    # Requiring the output channel rather than a cache step in the same file is
    # deliberate: the composite that BUILDS the key and the workflow that CACHES with it
    # are routinely different documents, which is the whole arrangement this function
    # exists to read.
    #
    # Residual, stated rather than implied: a document that writes an unrelated `key=`
    # value AND uses $GITHUB_OUTPUT elsewhere still over-collects. That direction is a
    # false rejection rather than a bypass, and narrowing it further would mean tracing
    # shell dataflow, which is past what this check should attempt.
    if "GITHUB_OUTPUT" not in text:
        return []
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
    # A prefix whose literal head does not come FIRST yields nothing above, because the
    # pattern requires the head to begin with an alphanumeric, and
    # `prefix="${name}-pip-..."` puts its literal part after the variable. Substituting
    # the values callers actually pass makes it readable. Without this the composite
    # contributed no head at all, while the `steps.*` key reading its output was
    # dismissed as delegation, so a publish `restore-keys: shared-pip-` had nothing to be
    # compared against.
    for value in sorted(inputs or ()):
        substituted = re.sub(
            r"\$\{\s*[\w-]+\s*\}|\$\{\{\s*inputs\.[\w-]+\s*\}\}", value, text
        )
        for found in pattern.findall(substituted) + re.findall(
            r"""echo\s+["']?(?:key|prefix)=([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""",
            substituted,
        ):
            if found not in heads:
                heads.append(found)
    if inputs and all_literal:
        # Narrowed heads, plus any head that already CARRIES a caller value because it
        # was recovered by substitution above and needs no further narrowing.
        return [f"{h}{v}-" for h in heads for v in sorted(inputs)] + [
            h for h in heads if any(h.startswith(v) for v in inputs)
        ]
    # An unresolved call site means the broad head still has to be carried, or narrowing
    # would silently drop the namespace that caller writes.
    return heads + [f"{h}{v}-" for h in heads for v in sorted(inputs or ())]






def _target_name(path: Path) -> str:
    """What a `uses:` line writes to reach this definition.

    An action is named by its DIRECTORY (`./.github/actions/pip-cache`), a reusable
    workflow by its FILE (`./.github/workflows/reuse.yml`). Using the directory for both
    made every reusable workflow look like a target called `workflows`, so no call site
    ever matched and the values its callers pass were never recovered -- and a key built
    from one of those values then had no namespace at all.
    """
    directory = path.parent if path.name.startswith("action.") else path
    for parent in path.parents:
        if parent.name == ".github":
            try:
                return directory.relative_to(parent.parent).as_posix()
            except ValueError:
                break
    return directory.name


def _namespaces_by_target(callers: list, targets: list) -> dict:
    """{target: {input name: (literal values, every call site resolved)}}.

    Per target, NOT merged across all of them. Two unrelated definitions may each declare
    an input called `cache_key`; pooling them by field name assigned one target's values
    to the other, so a publish key matching a value that only the NON-caching composite
    ever receives was rejected. The input name alone does not identify a namespace, and a
    guard that rejects a correct configuration is one someone eventually deletes.
    """
    out: dict = {}
    for target in targets:
        resolved = _resolved_inputs(callers, _target_name(target))
        # A declared default is a value the target really can be called with, so it
        # belongs in the namespace even when no call site mentions the input, and
        # recording it settles the OMISSION that made the input unresolved. It settles
        # nothing else: a caller that explicitly passes `${{ matrix.cache_key }}` has
        # overridden the default with a value this check cannot expand.
        for field, value in _declared_defaults(target).items():
            entry = resolved.get(field, (set(), True, 0, 0, set()))
            vals, _ok, _omitted, dynamic, delegated = entry
            resolved[field] = (vals | {value}, dynamic == 0, 0, dynamic, delegated)
        out[target] = {f: (p[0], p[1], p[4]) for f, p in resolved.items()}
    return out


def _expand_key(
    key: str, namespaces: dict, producers: dict | None = None, scope = None,
) -> tuple:
    """(every literal key this can take, whether that list is complete).

    Substitutes each `${{ inputs.X }}` OCCURRENCE, rather than only a key that is
    nothing but one expression. `key: prefix-${{ inputs.name }}` called with
    `name: shared` runs as `prefix-shared`, and matching only whole-key expressions left
    that exact collision uncompared -- the commonest way a composite names a key, missed
    because it had a prefix in front of it.

    `runner.os` is expanded too, so the exact comparison sees the three keys a job really
    writes instead of one expression that equals nothing.
    """
    out = {key.strip()}
    complete = True
    # Inputs whose every call site DELEGATES: the value is assembled elsewhere and its
    # namespace is recorded by `_shell_built_key_prefixes`, so the expression left
    # standing for one of these is not an unanswered question.
    delegated_inputs: set = set()
    for _ in range(6):
        nxt: set = set()
        changed = False
        for k in out:
            match = _INPUT_KEY.search(k)
            if match is None:
                nxt.add(k)
                continue
            entry = namespaces.get(match.group(1), (set(), False, set()))
            values, resolved = entry[0], entry[1]
            delegated = entry[2] if len(entry) > 2 else set()
            if not values:
                nxt.add(k)
                # Resolved with NO literals means every call site delegates. Only
                # resolved-false is genuine doubt. Treating the two alike reported this
                # repository's own pip-cache-save, whose every caller passes
                # `${{ steps.pip-cache.outputs.key }}`, against every publish key in the
                # tree -- a false failure on a correct configuration.
                readable = bool(delegated) and all(
                    _delegation_is_read(expr, producers or {}, expr_scope)
                    for expr, expr_scope in delegated
                )
                if resolved and readable:
                    delegated_inputs.add(match.group(1))
                else:
                    # Delegation only settles the key when the producer's namespace was
                    # actually recovered. `_shell_built_key_prefixes` reads two narrow
                    # spellings; a workflow that emits its key with `printf 'key=%s\n'`
                    # matches neither, so nothing was recorded and dismissing the key as
                    # "delegated" turned an unread producer into a clean bill of health.
                    complete = False
                continue
            changed = True
            if not resolved:
                # An unresolved call site keeps the raw expression alongside the
                # literals, or the caller this check cannot expand is silently dropped.
                complete = False
                nxt.add(k)
            for value in sorted(values):
                nxt.add(k[: match.start()] + value + k[match.end() :])
        out = nxt
        if not changed:
            break
    expanded: set = set()
    for k in out:
        if _RUNNER_OS_EXPR.search(k):
            expanded.update(_RUNNER_OS_EXPR.sub(v, k) for v in _RUNNER_OS_VALUES)
        else:
            expanded.add(k)
    # Residue that a delegated input accounts for does not make the key undecided; any
    # other surviving expression does. Checked after the delegated ones are removed, so a
    # key mixing the two is still reported.
    for k in expanded:
        residue = k
        for name in delegated_inputs:
            residue = re.sub(
                r"\$\{\{\s*inputs\." + re.escape(name) + r"\s*\}\}", "", residue
            )
        if "${{" in residue:
            complete = False
            break
    return sorted(expanded), complete


def _input_namespaces(callers: list, targets: list) -> dict:
    """The merged view of `_namespaces_by_target`, for the shell-head narrowing only.

    Merging is wrong for deciding a key's value, which is what `_namespaces_by_target`
    exists for. It stays here because narrowing a shell-built HEAD only ever adds
    candidate heads, so a value borrowed from a neighbouring target widens the recorded
    namespace rather than moving it, and widening is the safe direction.
    """
    merged: dict = {}
    for target in targets:
        resolved = _resolved_inputs(callers, _target_name(target))
        # A declared default is a value the target really can be called with, so it
        # belongs in the namespace even when no call site mentions the input, and
        # recording it settles the OMISSION that made the input unresolved.
        #
        # It settles nothing else. A caller that explicitly passes
        # `${{ matrix.cache_key }}` has overridden the default with a value this check
        # cannot expand, so the namespace stays undecided however many defaults exist.
        # Treating a default as blanket resolution dropped that caller's namespace
        # entirely, which turned the fix for one false failure into a silent bypass.
        defaults = _declared_defaults(target)
        for field, value in defaults.items():
            entry = resolved.get(field, (set(), True, 0, 0, set()))
            vals, _ok, _omitted, dynamic = entry[0], entry[1], entry[2], entry[3]
            resolved[field] = (vals | {value}, dynamic == 0, 0, dynamic, entry[4])
        for field, pair in resolved.items():
            vals, ok = merged.get(field, (set(), True))
            merged[field] = (vals | pair[0], ok and pair[1])
    return merged


def _expand_input_key(key: str, namespaces: dict) -> list[str]:
    """The literal keys a `key: ${{ inputs.X }}` can take, given its call sites.

    Needed on BOTH sides of the exact comparison, not just the prefix one. A pull request
    writing `shared-key` directly, against a dispatch workflow that calls a reusable
    workflow whose key is `${{ inputs.cache_key }}` with `cache_key: shared-key`, is an
    exact collision with no `restore-keys` anywhere, and comparing a literal against an
    unexpanded expression never matches. Resolution reached the prefix comparison through
    `pr_heads` and stopped there, so this plainest of all the shapes stayed open.
    """
    match = _INPUT_KEY.fullmatch(key.strip())
    if match is None:
        return [key]
    values, resolved = namespaces.get(match.group(1), (set(), False))
    if not values:
        return [key]
    # An unresolved call site keeps the raw expression alongside the literals. Returning
    # the literals alone dropped that caller: with one site passing `cache_key: safe-key`
    # and another `${{ matrix.cache_key }}`, the matrix value could be anything, and
    # discarding it meant neither comparison saw the namespace it writes.
    return sorted(values) if resolved else sorted(values) + [key]


def _declared_defaults(path: Path) -> dict:
    """`inputs.<name>.default` from an action or reusable workflow definition.

    Actions applies a declared default when a caller omits the input, so a composite
    declaring `cache_key` with default `shared-key` writes the `shared-key` namespace on
    a bare invocation. Reading only the call sites left that key as an unexpanded
    expression, the exact comparison matched nothing, and with no `restore-keys` in play
    the undecidable-prefix path never reported it either: a silent pass.
    """
    doc = _parse(path)
    out: dict = {}
    if not isinstance(doc, dict):
        return out
    declared = doc.get("inputs")
    if not isinstance(declared, dict):
        on = doc.get(True) if True in doc else doc.get("on")
        call = on.get("workflow_call") if isinstance(on, dict) else None
        declared = call.get("inputs") if isinstance(call, dict) else None
    if isinstance(declared, dict):
        for name, spec in declared.items():
            if isinstance(spec, dict) and spec.get("default") is not None:
                out[str(name)] = str(spec["default"])
    return out


def _local_ref_candidates(ref: str, root: Path) -> list:
    """Every source-tree path a `./...` reference could name.

    A job that checks this repository out into a subdirectory writes
    `./unsloth/.github/actions/x`, which is the same action reached through a layout
    that exists only at run time. Probing the reference as written found nothing in the
    source tree, so those actions were never added to the reachable set and never
    flattened: the keys they declare stayed outside both comparisons, and a login inside
    one was invisible. `notebooks-ci.yml` and `version-compat-ci.yml` both use this form.

    Tried as written first, then from the embedded `.github` component.
    """
    ref = ref.strip()
    if ref.startswith("./"):
        ref = ref[2:]
    ref = ref.rstrip("/")
    out = [root / ref]
    parts = ref.split("/")
    if ".github" in parts[1:]:
        out.append(root / "/".join(parts[parts.index(".github"):]))
    return out


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
            for cand in _local_ref_candidates(ref, root.parent):
                # A local reusable workflow reference names the .yml file itself rather
                # than a directory containing an action.yml, so it has to be followed on
                # its own.
                if cand.is_file() and cand.suffix in (".yml", ".yaml"):
                    dirs.add(cand)
                    queue.append(cand)
                    break
                found = False
                for action in (cand / "action.yml", cand / "action.yaml"):
                    if action.is_file():
                        dirs.add(action)
                        queue.append(action)
                        found = True
                if found:
                    break
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
    shell_built: set = set()
    # A workflow can build its key in one of its OWN `run:` steps and consume it as
    # `${{ steps.probe.outputs.key }}`, with no composite involved at all. Only actions
    # were read for shell-built heads, so that key hit the delegation branch with nothing
    # recovered and a publish `restore-keys: shared-` had nothing to compare against.
    shell_literals: set = set()
    for pth in pr_workflow_paths:
        text = pth.read_text(ENC)
        built = _shell_built_key_prefixes(text)
        composite_keys.extend(built)
        shell_built.update(built)
        # Literal keys the producer writes outright. Exact keys, so they belong in the
        # comparison, and evidence the step was understood.
        shell_literals.update(_shell_output_keys(text))
    for action_path in pr_reachable:
        composite_keys.extend(_extract_cache_keys(action_path))
        resolved = _resolved_inputs(pr_callers, _target_name(action_path))
        names, all_literal = resolved.get("name", (set(), False))[:2]
        built = _shell_built_key_prefixes(action_path.read_text(ENC), names, all_literal)
        composite_keys.extend(built)
        shell_literals.update(_shell_output_keys(action_path.read_text(ENC)))
        # A shell-built value is the literal HEAD of a key whose remainder is assembled
        # at runtime, so it is truncated by construction even though no `${{` survives
        # in the recorded string. Without this the reverse-direction rule below stopped
        # comparing them, and a publish prefix longer than the head -- which is the usual
        # shape, `pip-v2-shared-` against the head `pip-v2-` -- went unexamined.
        shell_built.update(built)

    # The publish side delegates to local actions exactly as the pull-request side does,
    # and reading only the top-level workflow file left that half unexamined. A publish
    # workflow whose composite holds the `actions/cache/restore` declares its keys and
    # its `restore-keys` in the action, so a PR writing `shared-*` against a publish-only
    # composite restoring `shared-` passed: the prefix was never collected, and a
    # comparison that collects nothing on one side reports success.
    for action_path in sorted(
        _pr_reachable_action_dirs(workflows_dir, [pth for pth, _ in publish_triggered])
    ):
        publish_triggered.append((action_path, _extract_cache_keys(action_path)))
        publish_restore_prefixes.append(
            (action_path, _extract_restore_key_prefixes(action_path))
        )

    # Composite keys belong in the exact comparison as well, not only the prefix one. A
    # PR-reachable action declaring `key: shared-key`, against a publish workflow using
    # that same key and no restore-keys at all, is the original cache-poisoning shape,
    # and it was invisible while this set held workflow-declared keys only.
    # Inputs each side's targets are called with, so an input-backed key resolves to the
    # namespace its callers actually produce before either comparison runs.
    input_namespaces = _input_namespaces(pr_callers, pr_reachable)
    publish_callers = [pth for pth, _ in publish_triggered]
    publish_reachable = sorted(
        _pr_reachable_action_dirs(workflows_dir, publish_callers)
    )
    publish_namespaces = _input_namespaces(publish_callers, publish_reachable)
    # The publish side builds keys in shell too, and its heads were never collected:
    # `_shell_built_key_prefixes` ran over PR-reachable documents only. A publish
    # workflow that emits `key=shared-key` and restores `${{ steps.probe.outputs.key }}`
    # therefore had that key dismissed as delegated with nothing recovered to compare,
    # while a pull request writing the literal `shared-key` passed.
    publish_shell: set = set()
    pub_scopes: dict = {}
    for pth in publish_callers + publish_reachable:
        for jid, key in _scoped_cache_keys(pth):
            pub_scopes.setdefault((pth.as_posix(), key), (pth.as_posix(), jid))
        text = pth.read_text(ENC)
        publish_shell.update(_shell_built_key_prefixes(text))
        publish_shell.update(_shell_output_keys(text))
    publish_producers: dict = {}
    for pth in publish_callers + publish_reachable:
        publish_producers.update(_producer_steps(pth))
    # Each definition keeps its OWN inputs. See `_namespaces_by_target`.
    pr_by_target = _namespaces_by_target(pr_callers, pr_reachable)
    publish_by_target = _namespaces_by_target(publish_callers, publish_reachable)

    # (namespace, key) rather than a bare key, so every key is expanded against the
    # inputs of the definition that declares it and no other.
    # (namespace, key, scope) -- the scope is the document and job whose step ids a
    # delegated key refers to.
    pr_sites = [
        ({}, k, (pth.as_posix(), jid))
        for pth, _keys in pr_triggered
        for jid, k in _scoped_cache_keys(pth)
    ]
    # Shell-built heads only. Re-adding every composite's YAML keys here gave each one a
    # SECOND entry carrying an empty namespace: it expanded to nothing, counted as
    # unresolved, and a composite whose callers all pass literals then rejected an
    # unrelated publish key on the strength of its own duplicate. The target loop below
    # is where those keys belong, with the inputs that decide them.
    pr_sites += [({}, k, None) for k in sorted(shell_built | shell_literals)]
    for target, namespace in pr_by_target.items():
        pr_sites += [
            (namespace, k, (target.as_posix(), jid))
            for jid, k in _scoped_cache_keys(target)
        ]

    pr_keys: set = set()
    # PR keys whose value this check could not settle, with the literal head they are
    # known to start with. Only these can collide with a publish key unseen.
    pr_undecided: list = []
    # Per producing STEP, not per side. One readable producer is not a warrant for an
    # unreadable one, and a side-wide flag let an ordinary `echo key=...` vouch for a
    # `printf` producer whose namespace was never recovered.
    pr_producers: dict = {}
    for pth in pr_workflow_paths + list(pr_reachable):
        pr_producers.update(_producer_steps(pth))
    for namespace, key, scope in pr_sites:
        literals, complete = _expand_key(key, namespace, pr_producers, scope)
        pr_keys.update(k for k in literals if "${{" not in k)
        # Delegation is not indecision. A key that is nothing but
        # `${{ steps.probe.outputs.key }}` names a value assembled in a shell step, and
        # `_shell_built_key_prefixes` recovers its head and records that namespace
        # separately; counting it here as well reported every publish key in the tree
        # against it, which failed the live tree on a correct configuration.
        if not complete and not (
            _DELEGATED_KEY.fullmatch(key.strip())
            and _delegation_is_read(key.strip(), pr_producers, scope)
        ):
            pr_undecided.append((key, _prefix_candidates(key)))
    pr_raw_unresolved = {raw.strip() for raw, _heads in pr_undecided}

    for pub_path, pub_keys in publish_triggered:
        # This target's own inputs, not the merged view. Two publish-reachable actions
        # sharing an input name had every key expanded with both their values, so a cache
        # composite given `publish-key` was also expanded to an unrelated action's
        # `safe-key` and reported as colliding with a PR cache of that name.
        # A top-level workflow is not a target, and its `${{ inputs.X }}` is a
        # workflow_dispatch input chosen by whoever dispatches it. Falling back to the
        # merged CHILD namespace expanded that user-controlled value using an unrelated
        # action's literal and then called it settled, so dispatching with
        # `cache_key=shared-key` restored a pull request's cache with the lint green.
        # An empty namespace leaves it undecidable, which is what it is.
        pub_ns = publish_by_target.get(pub_path, {})
        for raw in pub_keys:
            pub_scope = pub_scopes.get((pub_path.as_posix(), raw))
            literals, complete = _expand_key(
                raw, pub_ns, publish_producers, pub_scope
            )
            if _DELEGATED_KEY.fullmatch(raw.strip()):
                # Compared as the heads its own shell produced, rather than skipped.
                read = _delegation_is_read(
                    raw.strip(), publish_producers, pub_scope
                )
                literals = sorted(publish_shell) if read else [raw.strip()]
                complete = read
            # Identical spellings first. Two keys written the same way around an
            # expression this check cannot expand -- `shared-${{ hashFiles('lock') }}`
            # on both sides -- resolve identically at run time, so a PR that leaves the
            # lockfile alone writes the very entry the publish run restores. Both sides
            # were being dropped for still containing `${{`, which made the most direct
            # collision of all invisible.
            if not complete and raw.strip() in pr_raw_unresolved:
                findings.append(
                    f"{pub_path.name}: cache key {raw.strip()!r} is spelled identically "
                    "in a PR-triggered workflow. Neither value can be resolved here, "
                    "but two identical keys resolve identically at run time, so a fork "
                    "PR could write the entry this workflow restores. Add a unique "
                    "suffix (e.g. '-publish-only')."
                )
                continue
            for k in literals:
                if k in pr_keys:
                    findings.append(
                        f"{pub_path.name}: cache key {k!r} is also declared in a "
                        "PR-triggered workflow. A fork PR could poison this cache "
                        "and the publish workflow would restore it on next run. "
                        "Add a unique suffix (e.g. '-publish-only') to partition "
                        "the namespaces."
                    )
                    continue
                if "${{" in k:
                    if _DELEGATED_KEY.fullmatch(k.strip()) and _delegation_is_read(
                        k.strip(), publish_producers, pub_scope
                    ):
                        # Delegated AND read. See the PR side above.
                        continue
                    # An unresolved PUBLISH key, which needs the same treatment as an
                    # unresolved PR key and was simply skipped. A publish composite
                    # called with `cache_key: ${{ matrix.cache_key }}` may well produce a
                    # literal a pull request also writes, and failing closed only on the
                    # PR side left exactly half of this check's own rule unenforced.
                    # No literal head at all means the key is expression-led and
                    # could be ANY value, so it is not narrowable and every PR key is a
                    # candidate. Requiring a head silently exempted exactly the least
                    # decidable keys, which is the wrong way round.
                    heads = [h for h in _prefix_candidates(k) if h]
                    reported = False
                    for literal in sorted(pr_keys):
                        if not heads or any(literal.startswith(h) for h in heads):
                            findings.append(
                                f"{pub_path.name}: cache key {k!r} cannot be shown not "
                                f"to collide with the PR-written key {literal!r}, "
                                "because this check cannot resolve the publish key's "
                                "value. Give it a unique suffix (e.g. "
                                "'-publish-only'), or make it literal."
                            )
                            reported = True
                            break
                    if reported:
                        continue
                    # BOTH sides unresolved, and spelled differently. The identical-text
                    # rule above does not reach `shared-${{ matrix.pr_part }}` against
                    # `shared-${{ matrix.pub_part }}`, and comparing an unresolved
                    # publish key only against RESOLVED pr keys skipped the pairing
                    # entirely -- so two keys that can both become `shared-x` passed.
                    # Compatible when either head could lead to the other, which for two
                    # truncated keys is the prefix test in both directions.
                    for raw_pr, pr_heads in pr_undecided:
                        fixed = [h for h in pr_heads if h]
                        if (
                            not heads
                            or not fixed
                            or any(
                                h.startswith(f) or f.startswith(h)
                                for h in heads
                                for f in fixed
                            )
                        ):
                            findings.append(
                                f"{pub_path.name}: cache key {k!r} and the PR-reachable "
                                f"key {raw_pr!r} can both resolve to the same value, "
                                "and neither can be resolved here. Give the publish key "
                                "a unique suffix (e.g. '-publish-only'), or make one of "
                                "them literal."
                            )
                            break
                    continue
                # Fail closed. An unsettled PR key can equal this publish key at run
                # time, and reporting it only from the restore-prefix pass meant a
                # publish workflow with no `restore-keys` at all -- the plainest shape
                # there is -- never had the question asked. Narrowed to PR keys whose
                # fixed head this publish key actually begins with, because a PR key
                # headed `pip-v2-` cannot become `wheels-x` however its tail resolves.
                for raw_pr, heads in pr_undecided:
                    # Same rule in the other direction: a PR key with no literal head is
                    # unnarrowable rather than harmless.
                    fixed = [h for h in heads if h]
                    if not fixed or any(k.startswith(h) for h in fixed):
                        findings.append(
                            f"{pub_path.name}: cache key {k!r} cannot be shown not to "
                            f"collide with the PR-reachable key {raw_pr!r}, whose value "
                            "this check cannot resolve. Give the publish key a unique "
                            "suffix (e.g. '-publish-only'), or make the PR key literal."
                        )
                        break

    # Same trust boundary, reached by prefix instead of by an equal key. `restore-keys`
    # restores the newest entry whose key merely STARTS WITH the prefix, so a publish
    # workflow can adopt an entry a pull request wrote without the two keys ever being
    # equal, which is the only thing the check above compares.
    pr_heads: set = set()
    # Heads whose key continued into an unexpandable expression. Only these may be
    # reached by a publish prefix LONGER than the head itself.
    truncated_heads: set = set()
    undecidable_pr_keys: list[str] = []
    for k in list(pr_keys) + composite_keys:
        cands = _prefix_candidates(k)
        if cands:
            pr_heads.update(cands)
            if _is_truncated(k) or k in shell_built:
                truncated_heads.update(cands)
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
            truncated_heads.update(h for v in sorted(vals) if _is_truncated(v)
                                   for h in _prefix_candidates(v))
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
                hit = next(
                    (
                        h
                        for h in sorted(pr_heads)
                        if _prefix_compatible(h, pub_head, h in truncated_heads)
                    ),
                    None,
                )
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
