# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every third-party action must be referenced by commit SHA, never by a tag.

A tag is a pointer the upstream owner can move whenever they like. `uses: foo/bar@v3` is
therefore not a dependency on reviewed code, it is an agreement to run whatever that
repository contains on the day CI happens to run, decided by someone outside this
project. The decision is theirs, the credentials are ours: the action runs inside our
job, with whatever secrets and token permissions that job holds. `tj-actions/changed-files`
(GHSA-mrrh-fwg8-l2gq, March 2025) is the case everybody cites, where retagging one action
exfiltrated secrets out of tens of thousands of repositories, and it needed nothing more
exotic than a moved tag.

This repository already got this right nearly everywhere, which is exactly why a guard is
worth having: 53 of ~400 `uses:` references had drifted onto mutable tags, all of them in
the `docker-*` and `woa-wheelhouse` family, while every other workflow pinned properly.
That is the shape a convention with no test develops. Nothing asserted it before this
module -- `scripts/lint_workflow_triggers.py` checks triggers and cache keys, and no test
under `tests/` mentioned SHA pinning at all.

The 53 were pinned at the SHAs their tags already resolved to, which changed no versions.
Worth recording how that was confirmed, because it is the reassuring part: the resolved
SHAs were byte-identical to what the same actions were already pinned to elsewhere in this
repo, e.g. `actions/checkout` at 3d3c42e5 and `actions/upload-artifact` at 043fb46d. So
the docker family had simply been written in a different style, not held at a different
version.

Scope, and why it is drawn here. First-party `uses: ./...` references are exempt: they
resolve inside this checkout at the commit under test, so there is no third party and no
mutable pointer. Everything else is in, including `actions/*` and `docker/*`. Those are
reputable publishers, but reputable is not the property that matters. The property that
matters is whether the bytes can change without a commit here, and for a tag they can. A
compromised upstream account moves the tag either way.

Docker image references in `container:` and `services:` are deliberately NOT covered.
They are a real instance of the same problem, and pinning them by digest is worth doing,
but a `:tag` image reference and a `uses:` action reference have different syntax and
different failure modes, and a guard that tried to cover both would assert neither
clearly. Recorded here so the gap is known rather than mistaken for coverage.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
GITHUB = REPO / ".github"

_SHA = re.compile(r"^[0-9a-f]{40}$")

# A pinned-by-SHA reference whose upstream repository is gone or renamed cannot be
# reviewed, but it also cannot change, so it is not this module's problem.
DELIBERATELY_UNPINNED = {
    # Each entry needs the reason written here. Empty is the goal, and it growing needs
    # an argument rather than a deadline: there is no version of "we will pin it later"
    # that is safer than pinning it now, because the pin is a one-line change.
}


# Actions deliberately held at two different commits, keyed on `owner/repo`, with the
# reason. The failure below used to tell people to "say so in a comment at each site",
# which nothing read, so the documented remedy could not make CI pass and the only way
# out was editing this file. This is the mechanism that advice implied.
DELIBERATELY_SPLIT: dict[str, str] = {
    # Empty, and a split is nearly always a half-finished upgrade rather than a decision.
    # An entry here needs the reason a single commit will not do, not a note that two
    # exist.
}


def _sources():
    """Every workflow and action definition under .github, as (path, text)."""
    for path in sorted(GITHUB.rglob("*.y*ml")):
        yield path, path.read_text(encoding = "utf-8", errors = "ignore")


_REF = re.compile(
    r"""^(?P<repo>[A-Za-z0-9][\w.-]*/[\w.-]+(?:/[\w.\-/]+)?)@(?P<rev>[\w.\-/]+)$"""
)


def _uses_values(node):
    """Every `uses` value anywhere in a parsed document, at any depth.

    Walking the parsed structure rather than the source text, because enumerating
    spellings of the key does not converge. The lexical scan started on `uses:`, then
    needed `"uses":` and `uses :`, and flow style `- {uses: actions/checkout@v4}` is
    another valid step mapping again -- each one a reference the guard simply could not
    see, in a module whose entire job is to refuse mutable tags. PyYAML resolves all of
    them to the same mapping key, so asking it ends the sequence instead of extending it.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).strip() == "uses" and isinstance(value, str):
                yield value.strip()
            else:
                yield from _uses_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _uses_values(item)


def _references():
    """(path, lineno, ref, repo, rev) for every third-party `uses:` in the tree."""
    for path, text in _sources():
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        for ref in _uses_values(doc):
            # `uses: ./.github/actions/x` resolves inside this checkout, and
            # `docker://` is an image rather than an action; neither is a mutable
            # third-party pointer.
            if ref.startswith(".") or ref.startswith("docker://"):
                continue
            match = _REF.match(ref)
            if match is None:
                continue
            # The parser discards positions, so recover the line from the source for the
            # failure message. Only ever cosmetic: a reference that cannot be located is
            # still reported, at line 0.
            index = text.find(ref)
            lineno = text.count("\n", 0, index) + 1 if index >= 0 else 0
            yield path, lineno, ref, match.group("repo"), match.group("rev")


def test_the_scan_finds_the_references_it_claims_to():
    """A regex that matched nothing would pass every check below on an empty set."""
    refs = list(_references())
    assert len(refs) >= 200, (
        f"only found {len(refs)} third-party `uses:` references; the scan is wrong. This "
        f"tree has hundreds across .github/workflows and .github/actions."
    )
    repos = {repo for _, _, _, repo, _ in refs}
    for expected in ("actions/checkout", "step-security/harden-runner"):
        assert expected in repos, f"{expected} is used here but the scan missed it"


def test_the_sha_predicate_reads_the_revision():
    """The guard is only as good as this predicate, so the predicate is tested too."""
    forty = "3d3c42e5aac5ba805825da76410c181273ba90b1"
    cases = [
        (forty, True),
        (forty.upper(), False),  # git object names are lowercase hex
        (forty[:39], False),  # short SHAs are ambiguous and not pins
        (forty + "a", False),
        ("v7", False),
        ("v7.0.1", False),
        ("main", False),
        ("master", False),
        ("", False),
    ]
    for rev, expected in cases:
        assert bool(_SHA.match(rev)) is expected, f"_SHA.match({rev!r})"


def test_the_scan_reads_every_spelling_of_a_step_mapping():
    """A guard that recognises one spelling of its own key is a guard with a keyhole.

    The scan began as a regex on `uses:`, which missed `"uses":` and `uses :`, and then
    missed flow style `- {uses: actions/checkout@v4}` after those were added. Each miss
    was a reference absent from the scan entirely, so a mutable tag written that way
    passed a test whose whole purpose is to refuse mutable tags, and the cost of the
    bypass was a pair of quotes or a pair of braces. Enumerating spellings does not
    converge; the parser resolves all of them to the same mapping key.
    """
    spellings = [
        "steps:\n  - uses: actions/checkout@v4\n",
        'steps:\n  - "uses": actions/checkout@v4\n',
        "steps:\n  - 'uses': actions/checkout@v4\n",
        "steps:\n  - uses : actions/checkout@v4\n",
        "steps:\n  - {uses: actions/checkout@v4}\n",
        "steps: [{uses: actions/checkout@v4}]\n",
        'steps:\n  - uses: "actions/checkout@v4"\n',
        # Nested inside a composite action, which is the other place `uses` appears.
        "runs:\n  using: composite\n  steps:\n    - uses: actions/checkout@v4\n",
    ]
    for source in spellings:
        found = list(_uses_values(yaml.safe_load(source)))
        assert found == ["actions/checkout@v4"], f"{source!r} produced {found!r}"

    # A `uses` value that is not a reference, and a key that merely contains the word,
    # must not be picked up as third-party references.
    assert list(_uses_values(yaml.safe_load("steps:\n  - uses: ./.github/actions/x\n"))) == [
        "./.github/actions/x"
    ]
    assert list(_uses_values(yaml.safe_load("steps:\n  - name: uses a cache\n"))) == []


def test_the_reference_predicate_reads_owner_repo_at_ref():
    """`_REF` decides what counts as a third-party reference at all."""
    cases = [
        ("actions/checkout@v4", True),
        ("actions/cache/restore@v4", True),
        ("owner/repo@0123456789abcdef0123456789abcdef01234567", True),
        ("./.github/actions/x", False),
        ("docker://alpine:3", False),
        ("actions/checkout", False),      # no ref at all
        ("notapath@v4", False),           # no owner
        ("", False),
    ]
    for ref, expected in cases:
        assert bool(_REF.match(ref)) is expected, f"_REF.match({ref!r})"


def test_every_third_party_action_is_pinned_to_a_commit_sha():
    offenders = []
    for path, lineno, ref, _repo, rev in _references():
        if _SHA.match(rev):
            continue
        if ref in DELIBERATELY_UNPINNED:
            continue
        offenders.append(f"{path.relative_to(GITHUB).as_posix()}:{lineno}: {ref}")

    assert not offenders, (
        f"these `uses:` references name a mutable tag or branch instead of a commit:\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nA tag is a pointer the upstream owner can move, so each of these runs "
        "whatever that repository contains on the day CI runs, inside a job holding this "
        "repo's secrets and token permissions. That is the tj-actions/changed-files "
        "compromise (GHSA-mrrh-fwg8-l2gq), which needed nothing but a moved tag.\n\n"
        "Pin to the 40-character commit the tag currently points at and keep the version "
        "in a trailing comment, which is what the rest of this tree does:\n"
        "  uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n\n"
        "`gh api repos/<owner>/<repo>/commits/<tag> --jq .sha` resolves it. If a reference "
        "genuinely cannot be pinned, add it to DELIBERATELY_UNPINNED with the reason."
    )


def test_every_exemption_still_exists_and_still_needs_one():
    """A stale exemption is a hole nobody is asking for any more."""
    live = {ref for _, _, ref, _, _ in _references()}
    for ref in DELIBERATELY_UNPINNED:
        assert ref in live, (
            f"{ref} is exempted here but no longer referenced anywhere under .github, so "
            f"the exemption describes nothing. Remove it from DELIBERATELY_UNPINNED."
        )


@pytest.mark.parametrize(
    "repo",
    sorted({r for _, _, _, r, _ in _references()}),
)
def test_an_action_is_pinned_to_one_sha_everywhere_it_is_used(repo):
    """Two SHAs for one action means two versions of it run, which is nearly always a slip.

    Not a security property on its own, but it is how a half-finished upgrade shows up,
    and a stale copy is the one that keeps an already-fixed bug alive.
    """
    revs = {rev for _, _, _, r, rev in _references() if r == repo and _SHA.match(rev)}
    if len(revs) <= 1:
        return
    if repo in DELIBERATELY_SPLIT:
        return
    sites = sorted(
        f"{p.relative_to(GITHUB).as_posix()}:{ln}: {rev[:12]}"
        for p, ln, _, r, rev in _references()
        if r == repo and _SHA.match(rev)
    )
    pytest.fail(
        f"{repo} is pinned to {len(revs)} different commits, so different jobs run "
        f"different versions of it:\n  " + "\n  ".join(sites) + "\n\n"
        f"Settle on one commit, usually the newest reviewed one. If the split really is "
        f"deliberate, add {repo!r} to DELIBERATELY_SPLIT in this file with the reason a "
        f"single commit will not do."
    )


def test_every_split_exemption_still_exists_and_still_needs_one():
    """A stale exemption hides a split that has since been resolved."""
    live = {r for _, _, _, r, _ in _references()}
    for repo, reason in DELIBERATELY_SPLIT.items():
        assert reason.strip(), f"{repo} is exempted with no reason written; add one"
        assert repo in live, (
            f"{repo} is exempted from the one-commit rule but is no longer used anywhere "
            f"under .github, so the exemption describes nothing. Remove it."
        )
        revs = {rev for _, _, _, r, rev in _references() if r == repo and _SHA.match(rev)}
        assert len(revs) > 1, (
            f"{repo} is exempted from the one-commit rule but is now pinned to a single "
            f"commit, so the exemption is stale. Remove it from DELIBERATELY_SPLIT."
        )
