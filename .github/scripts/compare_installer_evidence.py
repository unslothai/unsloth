#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decide whether two installer runs behaved the same, given evidence collected on Windows.

This is the half of the base-vs-head functional lane that does not need Windows, so it lives in a
file with unit tests rather than inline in YAML. The Windows jobs run an installer and write
evidence; this compares two sets of it and produces a verdict.

The comparison is deliberately narrow: it answers "did the user-visible behaviour change", not "are
these runs identical". Two installs of the same commit are never byte-identical -- they differ in
timings, in temp directory names, in which mirror answered, in a `uv` patch release that shipped
between the two jobs. Every one of those is normalised away, and the normalisation rules are the
interesting part of this file: each one is a specific observed source of noise, and widening one to
silence a failure is how a lane like this stops being able to fail.

Three things are compared, because they fail independently:

- **transcript**: every line the installer printed, normalised. A hardening change must not alter
  what a user reads.
- **shortcuts**: the `.lnk` properties read back through the shell. This is where a change to the
  launch transport shows up, and it is invisible in the transcript.
- **artifacts**: a manifest of the files the installer wrote, with hashes for the ones whose content
  is a contract (`launch-studio.ps1`, `unsloth.cmd`).

`VOID` is a first-class outcome and not a pass. If the two sides are the same commit, or the
evidence is missing, or an installer did not finish, there is nothing to compare and saying "no
differences" would be a lie of exactly the kind this lane exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Each entry is (pattern, replacement, why). The `why` is not decoration: the next person to widen
# one of these needs to know what it was for, and a rule with no recorded cause is a rule nobody can
# argue with.
_NORMALISERS: tuple[tuple[re.Pattern[str], str, str, bool], ...] = (
    (re.compile(r"\b\d+\.\d+s\b"), "<duration>", "elapsed times, printed by every step", False),
    (re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%"), "<percent>", "download progress", False),
    (
        re.compile(r"\b\d+(?:\.\d+)?\s?(?:[KMGT]i?B|bytes)\b", re.I),
        "<size>",
        "download sizes, which differ with a CDN or a patch release",
        False,
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"), "<timestamp>", "timestamps", True),
    (
        re.compile(r"\b[0-9a-f]{40}\b|\b[0-9A-F]{64}\b|\b[0-9a-f]{64}\b"),
        "<hash>",
        "commit SHAs and file digests: the two sides are different commits by construction",
        True,
    ),
    (
        re.compile(r"(\.?unsloth-[A-Za-z][A-Za-z-]*[.-])[0-9a-fA-F]{8}[0-9a-fA-F-]*"),
        r"\1<temp>",
        "the random tail of an unsloth-* scratch name: unsloth-probe-<hex8>.tmp, the "
        "unsloth-uv-<hex8> work directory, .unsloth-write-probe.<guid>, "
        "unsloth-torch-overrides-<guid>.txt. Anchored on the random part on purpose. The rule "
        "this replaces matched any six characters after 'unsloth-', which also erased "
        "unsloth-studio-managed-launcher becoming unsloth-desktop-managed-launcher inside "
        "unsloth.cmd, a file whose text this lane treats as a contract",
        True,
    ),
    (
        re.compile(r"\\Temp\\[A-Za-z0-9._-]{6,}"),
        r"\\Temp\\<temp>",
        "Windows temp directory names",
        True,
    ),
    (re.compile(r"\b(pid|PID)[= ]\d+"), r"\1=<pid>", "process ids", False),
    (
        re.compile(r"127\.0\.0\.1:\d+|localhost:\d+"),
        "127.0.0.1:<port>",
        "the port Studio bound, which is chosen from what is free",
        False,
    ),
    (
        re.compile(r"\x1b\[[0-9;?]*[A-Za-z]"),
        "",
        "ANSI sequences, in case a run was not redirected after all",
        False,
    ),
    (re.compile(r"[\r\x08]"), "", "carriage returns and backspaces from progress redraws", False),
)

# Volatile only because the two jobs ran minutes apart. A version drift is not a behaviour change,
# but it IS worth printing, so these are normalised and separately reported.
_VERSION_PATTERN = re.compile(
    r"\b(uv|python|Python|CPython|git|cmake|torch|node|npm)[\s/-]+v?(\d+\.\d+(?:\.\d+)?)",
)


def collect_versions(text: str) -> dict[str, set[str]]:
    """What each side reported installing, reported rather than compared.

    A `uv` patch release that shipped between the two jobs is not a behaviour change and must not
    fail the lane. But a *deliberate* pin bump looks identical after normalisation, so the versions
    are printed side by side: silently normalising something away and never mentioning it is how a
    lane loses the ability to tell you anything.
    """
    found: dict[str, set[str]] = {}
    for match in _VERSION_PATTERN.finditer(text):
        found.setdefault(match.group(1).lower(), set()).add(match.group(2))
    return found


def report_version_drift(base: str, head: str, where: str, verdict: "Verdict") -> None:
    """Print what the version normaliser erased, wherever it was applied.

    `normalise_line` runs on shortcut fields and on generated-file content too, not only on the
    transcript, so a launcher retargeted from a python-3.11 directory to a python-3.13 one is
    erased in exactly the same way. That is the right call for a patch release that shipped between
    the two jobs, and the wrong one to make silently: this is the only rule in the set that can
    reach a deliberate change, so everywhere it reaches, the drift is said out loud.
    """
    base_versions = collect_versions(base)
    head_versions = collect_versions(head)
    for tool in sorted(set(base_versions) | set(head_versions)):
        before = ",".join(sorted(base_versions.get(tool, {"-"})))
        after = ",".join(sorted(head_versions.get(tool, {"-"})))
        if before != after:
            verdict.notes.append(
                f"version drift in the {where} (normalised away, not a failure): "
                f"{tool} base={before} head={after}"
            )


def normalise_line(line: str) -> str:
    out = line
    for pattern, replacement, _why, _in_scripts in _NORMALISERS:
        out = pattern.sub(replacement, out)
    out = _VERSION_PATTERN.sub(lambda m: f"{m.group(1)}/<version>", out)
    # Trailing whitespace only. Leading whitespace is load-bearing: `step` pads its label to exactly
    # 15 columns and REQUIRED_OUTPUT pins the indent, so stripping the left side would hide the one
    # regression class most likely to slip through a prose review.
    return out.rstrip()


def normalise_contract_value(value: str) -> str:
    """One persisted value, with only the per-run VALUES rewritten.

    Deliberately not `normalise_line`. That one is built for console output, where `10MB`, `0.5s` and
    `127.0.0.1:8123` are noise. In a shortcut's `arguments` or `targetPath` the same text is the
    launch contract: a `--limit 10MB` that becomes `20MB`, a timeout that changes, or a port written
    into the command line are exactly the changes this lane exists to see, and rewriting both sides
    to the same token reported every field equal.

    No rstrip either. Trailing whitespace in an argument string is part of the value, and the
    transcript rule that strips it exists because `step` pads its labels.
    """
    return _VERSION_PATTERN.sub(
        lambda m: f"{m.group(1)}/<version>", _apply_value_normalisers(value)
    )


def normalise_script(text: str) -> list[str]:
    """A generated script, with only the volatile VALUES rewritten.

    Deliberately not `normalise_transcript`. That one is built for captured console output: it drops
    blank lines, rstrips every line, and discards lines starting with runner noise like `Run `,
    `shell: ` or `env:`. Every one of those is destructive applied to a script. Trailing whitespace
    in a CMD `set` value is part of the value, a dropped blank line changes a here-string, and an
    echoed line that happens to begin with `Run ` is content. A candidate could change any of them
    and both sides would still compare equal.
    """
    return [
        # The scratch names, the embedded install ID, temp directories, timestamps and version
        # strings still have to go: they differ between the two sides for reasons that are not
        # behaviour. Nothing else is touched -- see `_apply_value_normalisers` -- and the line is
        # kept exactly as it is otherwise, trailing spaces and all.
        _VERSION_PATTERN.sub(
            lambda m: f"{m.group(1)}/<version>",
            _apply_value_normalisers(raw),
        )
        for raw in text.splitlines()
    ]


def _apply_value_normalisers(line: str) -> str:
    """Only the rules whose fourth field says the value is per-run volatile in a FILE.

    The other rules exist for captured console output and are content in a generated script. A
    launcher whose health probe moved from port 8888 to 9999, or a `unsloth.cmd` whose upload limit
    went from 10MB to 20MB, is a behaviour change, and the transcript rules rewrite both sides to
    the same token and report PASS. What genuinely differs between two installs of two commits is
    the embedded studio_root_id, the scratch names, the temp directory and any timestamp, so those
    four are all that a script is normalised for.
    """
    out = line
    for pattern, replacement, _why, in_scripts in _NORMALISERS:
        if not in_scripts:
            continue
        out = pattern.sub(replacement, out)
    return out


def normalise_transcript(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = normalise_line(raw)
        if not line.strip():
            continue
        # Runner-injected noise, and ONLY what the runner injects. These carry the workflow's own
        # group names and the side's SHA, so they differ between sides for nothing to do with the
        # installer.
        #
        # `Run `, `shell: ` and `env:` were in this list and are now gone, because they were never in
        # the file. The workflow tees the child powershell.exe stream into transcript.txt
        # (windows-installer-differential-ci.yml:273-274), so GitHub's step headers never reach it,
        # while the installers print at least six lines that begin with `Run ` once indentation is
        # stripped: install.ps1:1518 and studio/setup.ps1:2245, :2526, :3073, :3526, :4506. Every one
        # of those is user-visible guidance on an exercised path, and this rule deleted them from
        # both sides, so changing or dropping one of them compared equal. Matching a prose prefix is
        # the wrong shape for this job; if the capture ever widens to include the step's own output,
        # the honest fix is to narrow the capture, not to delete lines that might be ours.
        #
        # Left anchored at column 0: nothing the installers print starts in column 0 with these.
        if line.startswith(("##[group]", "##[endgroup]", "::group::", "::endgroup::", "##[debug]")):
            continue
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class Verdict:
    """The outcome, with every difference kept rather than only the first."""

    def __init__(self) -> None:
        self.void: list[str] = []
        self.differences: list[str] = []
        self.notes: list[str] = []

    @property
    def is_void(self) -> bool:
        return bool(self.void)

    @property
    def passed(self) -> bool:
        return not self.void and not self.differences

    def exit_code(self) -> int:
        # VOID and DIFFERENT are both non-zero, and deliberately distinct: 2 means "measured, and it
        # changed"; 3 means "could not measure", which must never read as a pass.
        if self.void:
            return 3
        if self.differences:
            return 2
        return 0


def _unified(
    base: list[str],
    head: list[str],
    label: str,
    limit: int = 60,
) -> list[str]:
    import difflib

    diff = list(
        difflib.unified_diff(
            base, head, fromfile = f"base/{label}", tofile = f"head/{label}", lineterm = "", n = 2
        )
    )
    if len(diff) > limit:
        omitted = len(diff) - limit
        diff = diff[:limit] + [f"... {omitted} more diff lines omitted; the artifact has all of it"]
    return diff


def compare_transcripts(
    base: str,
    head: str,
    verdict: Verdict,
    label: str = "transcript",
) -> None:
    base_lines = normalise_transcript(base)
    head_lines = normalise_transcript(head)
    if not base_lines or not head_lines:
        verdict.void.append(
            f"one side's {label} is empty after normalisation, so there is nothing to compare. "
            f"An installer that printed nothing did not run."
        )
        return
    report_version_drift(base, head, label, verdict)

    if base_lines == head_lines:
        verdict.notes.append(f"{label}: identical over {len(base_lines)} normalised lines")
        return
    verdict.differences.append(
        f"the installer's user-visible output changed in the {label}:\n"
        + "\n".join(_unified(base_lines, head_lines, label))
    )


def _shortcut_key(entry: dict) -> str:
    """Location AND file name.

    A normal install writes the same file name to the Desktop and to the Start Menu, so a key of
    just the name collapses the pair into one entry. If one of the two stopped being created and
    the survivor kept its fields, both maps still held one identical key and the comparison
    reported equality -- the disappearance of a shortcut being exactly what this lane is for. The
    collector already records `root` as a leaf name, so it costs nothing and does not reintroduce
    the workspace path that deliberately is not part of the key.
    """
    name = str(entry.get("name") or entry.get("path") or "<unnamed>")
    root = entry.get("root")
    return f"{root}/{name}" if root else name


# Read back from the shell, and every one of them is a contract. Arguments especially: it carries
# -WindowStyle and -ExecutionPolicy, which is the pair this whole effort is about, and a change
# there is completely invisible in the transcript.
_SHORTCUT_FIELDS = (
    "targetPath",
    "arguments",
    "workingDirectory",
    "windowStyle",
    "iconLocation",
    # The tooltip. It is user-visible, the collector records it, and install.ps1 reads it back at
    # :3455 as part of deciding whether a shortcut is already correct, so a change to it is both a
    # behaviour change and invisible in the transcript. Leaving it out made those compare equal.
    "description",
)


# The two files this lane treats as contracts by their TEXT, kept in step with the collector's
# $contentFiles (.github/scripts/Collect-InstallerEvidence.ps1). A name here without captured content
# is VOID rather than skipped, and the list is explicit so adding a third place to the collector
# without adding it here is visible rather than silent.
_CONTENT_CONTRACTS = ("launch-studio.ps1", "unsloth.cmd")


def _as_list(value) -> list[dict]:
    """ConvertTo-Json unwraps a one-element collection into a bare object.

    The collector forces an array, but this side must not depend on that: a schema that changes with
    the number of shortcuts found would make the single-shortcut case iterate dictionary *keys* and
    compare strings, which reports agreement for entirely the wrong reason.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return [item for item in value if isinstance(item, dict)]


def _shape_problem(value) -> str | None:
    """Anything that is not a list of objects, or one unwrapped object, is not a manifest.

    `_as_list` used to absorb the difference silently: handed a bare string it iterated characters,
    kept none of them, and produced an empty list, which then compared against a populated side as
    "every shortcut disappeared". That reads as a behaviour change and is nothing of the kind, so
    the shape is checked rather than coerced.
    """
    if value is None or isinstance(value, dict):
        return None
    if isinstance(value, list):
        bad = sum(1 for item in value if not isinstance(item, dict))
        if bad:
            return f"{bad} of {len(value)} entries are not objects"
        return None
    return f"the manifest is a {type(value).__name__}, not a list of shortcut objects"


def compare_shortcuts(base, head, verdict: Verdict) -> None:
    for side, value in (("base", base), ("head", head)):
        problem = _shape_problem(value)
        if problem:
            verdict.void.append(
                f"{side}'s shortcut manifest is not the shape this lane writes: {problem}. "
                f"Evidence that cannot be parsed was not measured."
            )
    if verdict.is_void:
        return

    before_differences = len(verdict.differences)
    base, head = _as_list(base), _as_list(head)
    if not base and not head:
        verdict.void.append(
            "neither side reported any shortcut. The installer creates a desktop and a Start Menu "
            "entry, so zero on both sides means the evidence was not collected, not that they agree."
        )
        return
    # A collection error on both sides compares equal to itself. Observed while wiring this up: two
    # runs that both failed to read any shortcut reported "1 compared, every field equal" and exited
    # zero. Failures are symmetric far more often than behaviour changes are -- they usually come
    # from the host, which both sides share -- so the symmetry is no comfort at all.
    for side, entries in (("base", base), ("head", head)):
        for entry in entries:
            if entry.get("error"):
                verdict.void.append(
                    f"{side} could not read shortcut {_shortcut_key(entry)!r}: {entry['error']}. "
                    f"Two sides that both failed to collect evidence agree with each other and "
                    f"prove nothing."
                )
                continue
            # An entry with no identity and no launch contract is not a shortcut. `{}` survives
            # `_shape_problem` (it IS an object), `_as_list` counts it as one, and `_shortcut_key`
            # names it `<unnamed>`, so two empty objects compared equal and the run reported "1
            # compared, every field equal". The same collector writes both sides, so a schema
            # regression is symmetric and this is the shape it takes.
            if not (entry.get("name") or entry.get("path")):
                verdict.void.append(
                    f"{side} reported a shortcut with no name and no path, so there is nothing to "
                    f"identify it by and nothing was measured: {entry!r}"
                )
                continue
            if not any(entry.get(field) for field in _SHORTCUT_FIELDS):
                verdict.void.append(
                    f"{side}'s shortcut {_shortcut_key(entry)!r} carries none of the launch contract "
                    f"fields {list(_SHORTCUT_FIELDS)}, so the contract this lane exists to compare "
                    f"was never collected"
                )
    if verdict.is_void:
        return

    base_map = {_shortcut_key(e): e for e in base}
    head_map = {_shortcut_key(e): e for e in head}

    def _fields(entries: list[dict]) -> str:
        return "\n".join(str(e.get(f, "")) for e in entries for f in _SHORTCUT_FIELDS)

    report_version_drift(_fields(base), _fields(head), "shortcut fields", verdict)

    for missing in sorted(set(base_map) - set(head_map)):
        verdict.differences.append(f"shortcut {missing!r} exists on base and not on head")
    for added in sorted(set(head_map) - set(base_map)):
        verdict.differences.append(f"shortcut {added!r} exists on head and not on base")

    for name in sorted(set(base_map) & set(head_map)):
        for field in _SHORTCUT_FIELDS:
            before = normalise_contract_value(str(base_map[name].get(field, "")))
            after = normalise_contract_value(str(head_map[name].get(field, "")))
            if before != after:
                verdict.differences.append(
                    f"shortcut {name!r} field {field!r} changed:\n  base: {before}\n  head: {after}"
                )
    # Scoped to this comparison. Reading the whole verdict here meant a transcript difference
    # suppressed the shortcut note, so a run that reported a changed line also stopped saying
    # whether the launch contract had been looked at.
    if len(verdict.differences) == before_differences:
        verdict.notes.append(f"shortcuts: {len(base_map)} compared, every field equal")


def compare_artifacts(base: dict, head: dict, verdict: Verdict) -> None:
    """The installed files. Paths on both sides, content only where content is a contract."""
    # The same reasoning as for shortcuts, which this did not have. The collector records a
    # collection failure as an `error` field rather than a red step, and an error read as data is a
    # green run: two sides that both failed to enumerate the install root list no files, compare
    # equal, and report agreement. A per-file error is worse, because the entry still exists with
    # the same key and only the `content` is gone, so the content check below skips it silently and
    # the file that was never compared is the one whose text is the contract.
    for side, data in (("base", base), ("head", head)):
        if not isinstance(data, dict):
            verdict.void.append(
                f"{side}'s artifact manifest is a {type(data).__name__}, not an object"
            )
            continue
        if data.get("error"):
            verdict.void.append(
                f"{side} could not collect its artifact manifest: {data['error']}. A collection "
                f"failure is not evidence, and it is symmetric far more often than a behaviour "
                f"change is."
            )
    if verdict.is_void:
        return

    base_files = base.get("files") or {}
    head_files = head.get("files") or {}
    for side, files in (("base", base_files), ("head", head_files)):
        if not isinstance(files, dict):
            verdict.void.append(
                f"{side}'s artifact manifest lists files as a {type(files).__name__}, not an object"
            )
            continue
        for name in sorted(files):
            entry = files[name]
            if isinstance(entry, dict) and entry.get("error"):
                verdict.void.append(
                    f"{side} could not read {name!r}: {entry['error']}. The file is still listed, "
                    f"so without this the content check below would skip it and the run would "
                    f"report agreement about a file it never read."
                )
    if verdict.is_void:
        return

    if not base_files and not head_files:
        verdict.void.append("neither side listed any installed file, so nothing was measured")
        return

    for missing in sorted(set(base_files) - set(head_files)):
        verdict.differences.append(f"base installed {missing!r} and head did not")
    for added in sorted(set(head_files) - set(base_files)):
        verdict.differences.append(f"head installed {added!r} and base did not")

    for name in sorted(set(base_files) & set(head_files)):
        before, after = base_files[name], head_files[name]
        if not isinstance(before, dict) or not isinstance(after, dict):
            # VOID, not skipped. The SAME collector runs on both legs, so a malformed entry is
            # malformed identically on both and skipping it left the maps non-empty, the key sets
            # matching and nothing compared, which the run then reported as agreement. The shortcut
            # manifest and the top-level manifests are already validated this way.
            sides = [
                side
                for side, value in (("base", before), ("head", after))
                if not isinstance(value, dict)
            ]
            verdict.void.append(
                f"{name!r} is a {type(before).__name__ if 'base' in sides else type(after).__name__}"
                f" and not an object on {' and '.join(sides)}, so its evidence could not be read"
            )
            continue
        # The two entries whose CONTENT is the contract. An empty object on both sides made the
        # asymmetry check below false and the comparison below that false too, so the loop compared
        # nothing and the run passed. Symmetric malformed evidence is the likely failure mode here,
        # because the candidate collector writes both manifests.
        if name in _CONTENT_CONTRACTS and not before.get("error") and not after.get("error"):
            absent = [
                side
                for side, value in (("base", before), ("head", after))
                if "content" in value and value.get("content") is not None
            ]
            if len(absent) != 2:
                verdict.void.append(
                    f"{name!r} is one of the files whose text is the contract, and its content was "
                    f"not captured on "
                    f"{' or '.join(s for s in ('base', 'head') if s not in absent)}. Nothing was "
                    f"compared, and two sides that both captured nothing agree with each other."
                )
                continue
        if ("content" in before) != ("content" in after):
            # One side captured the text and the other did not. Skipping quietly, which is what
            # happened before, means the file whose content is the whole reason it is in the
            # manifest goes uncompared while the run still reports agreement.
            side = "head" if "content" in before else "base"
            verdict.void.append(
                f"{name!r} has captured content on one side only, so {side} never contributed the "
                f"text this lane compares"
            )
            continue
        if "content" in before and "content" in after:
            # The drift note first, on the RAW text, exactly as the transcript and shortcut
            # comparisons do it. Without this a launcher retargeted from python-3.11.9 to
            # python-3.13.0 was normalised away and the lane returned PASS with no note at all,
            # which is the one thing normalisation is supposed to buy back.
            report_version_drift(before["content"], after["content"], f"generated {name}", verdict)
            b = normalise_script(before["content"])
            a = normalise_script(after["content"])
            if b != a:
                verdict.differences.append(
                    f"the generated {name} changed:\n" + "\n".join(_unified(b, a, name))
                )
        # Encoding is part of the contract and is invisible in the decoded text. Windows PowerShell
        # 5.1 reads a BOM-less file as ANSI, so a launcher that silently stops carrying its UTF-8
        # BOM breaks every install whose paths contain non-ASCII characters while comparing equal.
        bom_before, bom_after = before.get("bom"), after.get("bom")
        if name in _CONTENT_CONTRACTS and not (bom_before and bom_after):
            # VOID, not skipped. The same collector writes both manifests, so a regression that drops
            # the field drops it on both sides, and skipping quietly reported a pass for an encoding
            # contract that was never measured. This is the field whose absence in the FILE breaks
            # every install with a non-ASCII path, so unmeasured is not a pass.
            missing = [
                side for side, value in (("base", bom_before), ("head", bom_after)) if not value
            ]
            verdict.void.append(
                f"{name!r} carries no bom metadata on {' and '.join(missing)}, so the encoding "
                f"contract was not measured. Windows PowerShell 5.1 reads a file with no BOM as ANSI."
            )
        elif bom_before and bom_after and bom_before != bom_after:
            verdict.differences.append(
                f"{name!r} changed encoding: base wrote {bom_before} and head wrote {bom_after}. "
                f"Windows PowerShell 5.1 reads a file with no BOM as ANSI."
            )
        # Where it landed, not only what is in it. The collector probes each contract at several
        # supported locations and records the one it found, so a candidate that moves studio.conf
        # between `share\studio.conf` and `studio.conf` without touching a byte keeps the same key
        # and the same content. Comparing content alone reports that as agreement, while everything
        # that has to open the file now looks in the wrong place.
        found_before, found_after = before.get("foundAt"), after.get("foundAt")
        if found_before and found_after and found_before != found_after:
            verdict.differences.append(
                f"{name!r} moved: base wrote it to {found_before} and head wrote it to "
                f"{found_after}. The bytes may match, but consumers must now look elsewhere."
            )

    # The install ID, checked WITHIN each side rather than across them. The launcher embeds the ID
    # it will accept from the backend, and the backend reads the persisted one, so if those two
    # disagree Studio refuses its own server and never starts. They are expected to differ between
    # base and head, which is precisely why a cross-side comparison cannot see this and why the
    # transcript normaliser rewriting every 64-hex token hides it completely.
    for side, data in (("base", base), ("head", head)):
        persisted, embedded = data.get("installId"), data.get("embeddedId")
        if persisted and embedded and persisted != embedded:
            verdict.differences.append(
                f"{side}: the launcher expects studio_root_id {embedded!r} but the install "
                f"persisted {persisted!r}. Studio would refuse its own backend."
            )
        elif embedded and not persisted:
            verdict.void.append(
                f"{side}: the launcher embeds an expected studio_root_id but no persisted "
                f"studio_install_id was found, so the pair could not be checked"
            )

    # Idempotency is reported by the Windows side, which is the only place it can be observed: it
    # runs the installer twice and records whether the second run rewrote anything.
    for side, data in (("base", base), ("head", head)):
        rewritten = data.get("rewrittenOnSecondRun")
        if rewritten is None:
            # VOID, not a note. `None` and `[]` mean different things here and the collector is
            # careful to keep them apart: `[]` is "measured, nothing was rewritten" and `None` is
            # "not measured". Treating the second as optional evidence let the lane report equality
            # while one of the four contracts it advertises had never been checked, which happens
            # whenever the first collector or the non-terminating Copy-Item ahead of the second
            # install fails while everything after it succeeds.
            verdict.void.append(
                f"{side}: idempotency was never measured, so there is no evidence that a reinstall "
                f"writes nothing. That is one of this lane's four contracts, and an unmeasured "
                f"contract is not a passing one."
            )
        elif rewritten:
            verdict.differences.append(
                f"{side}: running the installer a second time rewrote {sorted(rewritten)}. "
                f"A reinstall that changed nothing must write nothing."
            )
        else:
            verdict.notes.append(f"{side}: the second run rewrote nothing")


# The installer's own exit status, which the transcript does not carry. Recorded by the workflow,
# which is the only thing that sees it.
_RUN_CODES = (
    ("installExit", "the installer"),
    ("secondInstallExit", "the second, idempotency install"),
)


def compare_run_status(base, head, verdict: Verdict) -> None:
    """An installer that failed measured nothing, however tidy its transcript looks.

    This is the most dangerous symmetry the lane has. A host problem, a mirror outage, a Defender
    definition push: any of them fails both installs the same way, at the same point, printing the
    same lines. The transcripts then match, the shortcut manifests are both empty in the same way,
    and the lane reports "no behaviour difference" about two installs that never happened. The exit
    status is the only thing that distinguishes that from a real pass, so a missing one is VOID too:
    the comparer is taken from the candidate on both sides, so evidence without it is evidence from
    a run that was never in a position to say the installer finished.
    """
    for side, data in (("base", base), ("head", head)):
        if not isinstance(data, dict):
            verdict.void.append(
                f"{side} recorded no usable installer exit status, so nothing establishes that its "
                f"installer finished"
            )
            continue
        for key, what in _RUN_CODES:
            code = data.get(key)
            if not isinstance(code, int) or isinstance(code, bool):
                verdict.void.append(
                    f"{side} recorded no exit status for {what} ({key!r} is {code!r}). Two runs "
                    f"that both failed print matching transcripts, so a comparison that cannot see "
                    f"the exit status cannot tell a pass from a shared failure."
                )
            elif code != 0:
                verdict.void.append(
                    f"{side}: {what} exited {code}. Nothing it left behind is evidence of what a "
                    f"successful install does."
                )


def _load(path: Path, verdict: Verdict, what: str):
    if not path.is_file():
        verdict.void.append(f"{what} is missing at {path}, so this side produced no evidence")
        return None
    try:
        # utf-8-sig, not utf-8: Windows PowerShell 5.1 writes a BOM for `Set-Content -Encoding
        # utf8`, and a BOM makes json.loads fail on a file that is otherwise perfectly good.
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding = "utf-8-sig", errors = "replace"))
        return path.read_text(encoding = "utf-8-sig", errors = "replace")
    except (OSError, ValueError) as exc:
        verdict.void.append(f"{what} at {path} could not be read: {exc}")
        return None


def compare_directories(
    base_dir: Path,
    head_dir: Path,
    base_sha: str = "",
    head_sha: str = "",
) -> Verdict:
    verdict = Verdict()

    if base_sha and head_sha and base_sha == head_sha:
        verdict.void.append(
            f"both sides are {base_sha[:12]}. Comparing a commit with itself cannot show that a "
            f"change preserved behaviour; it shows only that the lane is deterministic."
        )
        return verdict

    base_transcript = _load(base_dir / "transcript.txt", verdict, "the base transcript")
    head_transcript = _load(head_dir / "transcript.txt", verdict, "the head transcript")
    # The reinstall output, which the workflow has always captured and this comparer never read. A
    # candidate that changes what the installer prints only when an installation already exists --
    # a reinstall warning added or dropped, a "nothing to do" line reworded -- leaves the first-run
    # transcripts identical and the artifacts untouched, so without this the lane reported PASS on
    # a user-visible change. Required rather than optional: the step that writes it runs under
    # `if: always()`, so a side that does not have one did not produce the evidence.
    base_second = _load(
        base_dir / "transcript-second-run.txt",
        verdict,
        "the base second-run transcript",
    )
    head_second = _load(
        head_dir / "transcript-second-run.txt",
        verdict,
        "the head second-run transcript",
    )
    base_shortcuts = _load(base_dir / "shortcuts.json", verdict, "the base shortcut manifest")
    head_shortcuts = _load(head_dir / "shortcuts.json", verdict, "the head shortcut manifest")
    base_artifacts = _load(base_dir / "artifacts.json", verdict, "the base artifact manifest")
    head_artifacts = _load(head_dir / "artifacts.json", verdict, "the head artifact manifest")
    base_run = _load(base_dir / "run.json", verdict, "the base run status")
    head_run = _load(head_dir / "run.json", verdict, "the head run status")

    if verdict.is_void:
        return verdict

    compare_run_status(base_run, head_run, verdict)
    if verdict.is_void:
        return verdict

    compare_transcripts(base_transcript, head_transcript, verdict, "first-run transcript")
    compare_transcripts(base_second, head_second, verdict, "second-run transcript")
    # Passed through as loaded, not coerced with `or []` / `or {}`. The coercion turned a manifest
    # of the wrong shape into an empty one of the right shape, and an empty manifest against a
    # populated one reads as "every file disappeared" -- a behaviour difference, reported about
    # evidence that was never parsed.
    compare_shortcuts(base_shortcuts, head_shortcuts, verdict)
    compare_artifacts(base_artifacts, head_artifacts, verdict)
    return verdict


# ---------------------------------------------------------------------------
# The positive control
# ---------------------------------------------------------------------------

# A differ that reports no differences looks exactly the same whether it is working or broken. So
# before the real comparison is trusted, it is handed a pair it MUST call different, and a pair it
# MUST call equal. Both directions matter: a differ that flags everything is as useless as one that
# flags nothing, it just fails more loudly.
_CONTROL_TRANSCRIPT = "\n".join(
    [
        "  python         3.13.14 ready",
        "  studio         installed in 12.4s",
        "  shortcut       desktop and Start Menu",
    ]
)


def self_test() -> list[str]:
    failures: list[str] = []

    noisy = "\n".join(
        [
            "  python         3.13.9 ready",
            "  studio         installed in 41.9s",
            "  shortcut       desktop and Start Menu",
        ]
    )
    v = Verdict()
    compare_transcripts(_CONTROL_TRANSCRIPT, noisy, v)
    if v.differences:
        failures.append(
            "the normaliser is too strict: a version drift and a timing difference were reported "
            "as a behaviour change, which would make this lane fail on every run and get disabled. "
            f"Reported: {v.differences}"
        )

    changed = _CONTROL_TRANSCRIPT.replace("desktop and Start Menu", "desktop only")
    v = Verdict()
    compare_transcripts(_CONTROL_TRANSCRIPT, changed, v)
    if not v.differences:
        failures.append(
            "the normaliser is too loose: a changed output line was NOT reported. Every 'no "
            "differences' verdict this lane has ever produced would be worthless."
        )

    indented = _CONTROL_TRANSCRIPT.replace("  python", "   python")
    v = Verdict()
    compare_transcripts(_CONTROL_TRANSCRIPT, indented, v)
    if not v.differences:
        failures.append(
            "an indentation change was not reported. `step` pads its label to exactly 15 columns "
            "and the output lock pins the indent, so a lost space is a real regression."
        )

    v = Verdict()
    compare_shortcuts(
        [
            {
                "name": "Unsloth.lnk",
                "arguments": "-NoProfile -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File x",
            }
        ],
        [
            {
                "name": "Unsloth.lnk",
                "arguments": "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File x",
            }
        ],
        v,
    )
    if not v.differences:
        failures.append(
            "a shortcut whose execution policy changed from RemoteSigned to Bypass was NOT "
            "reported. That is the single substitution this entire effort is about."
        )

    v = Verdict()
    compare_shortcuts([], [], v)
    if not v.is_void:
        failures.append(
            "two empty shortcut manifests were treated as agreement rather than as VOID"
        )

    v = compare_directories(Path("/nonexistent/base"), Path("/nonexistent/head"), "aaa", "bbb")
    if not v.is_void or v.exit_code() != 3:
        failures.append("missing evidence did not produce VOID with exit code 3")

    # The normaliser that erased a renamed launcher marker. `unsloth-studio-managed-launcher` is
    # written into unsloth.cmd and is how the installer recognises its own shim, so a rename is a
    # behaviour change; the rule meant for `unsloth-uv-<hex8>` swallowed it.
    v = Verdict()
    compare_transcripts(
        "  cmd            rem unsloth-studio-managed-launcher",
        "  cmd            rem unsloth-desktop-managed-launcher",
        v,
    )
    if not v.differences:
        failures.append(
            "a renamed unsloth-* marker was normalised away. The temp-name rule is anchored on a "
            "random hex tail precisely so that it cannot reach a name that means something."
        )
    v = Verdict()
    compare_transcripts(
        r"  work           C:\Temp\unsloth-uv-1a2b3c4d\bin",
        r"  work           C:\Temp\unsloth-uv-99ffee00\bin",
        v,
    )
    if v.differences:
        failures.append(
            f"a random temp directory name was reported as a behaviour change: {v.differences}"
        )

    v = Verdict()
    compare_artifacts(
        {"files": {"launch-studio.ps1": {"content": "x\n"}}},
        {"files": {"launch-studio.ps1": {"error": "access denied"}}},
        v,
    )
    if not v.is_void:
        failures.append(
            "an artifact entry carrying an error was treated as data. The entry still has its key, "
            "so the content check skips it and the run agrees about a file it never read."
        )

    v = Verdict()
    compare_run_status(
        {"installExit": 1, "secondInstallExit": 1}, {"installExit": 1, "secondInstallExit": 1}, v
    )
    if not v.is_void:
        failures.append(
            "two installers that both exited non-zero were not VOID. A shared failure produces "
            "matching transcripts, which is the one symmetry that looks exactly like a pass."
        )
    v = Verdict()
    compare_run_status({}, {}, v)
    if not v.is_void:
        failures.append("evidence with no recorded installer exit status was not VOID")
    v = Verdict()
    compare_run_status(
        {"installExit": 0, "secondInstallExit": 0}, {"installExit": 0, "secondInstallExit": 0}, v
    )
    if v.is_void or v.differences:
        failures.append(f"two successful installs were not accepted: {v.void} {v.differences}")

    return failures


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--base", type = Path, help = "directory holding the base side's evidence")
    parser.add_argument("--head", type = Path, help = "directory holding the head side's evidence")
    parser.add_argument("--base-sha", default = "")
    parser.add_argument("--head-sha", default = "")
    parser.add_argument(
        "--self-test", action = "store_true", help = "run the positive controls and exit"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        failures = self_test()
        for failure in failures:
            print(f"::error::self-test: {failure}")
        if failures:
            print(
                "::error::the comparer's own controls failed, so no verdict it produces can be "
                "trusted. Refusing to compare."
            )
            return 1
        print("self-test: the comparer reports real changes and ignores known noise")
        return 0

    if not args.base or not args.head:
        parser.error("--base and --head are required unless --self-test is given")

    verdict = compare_directories(args.base, args.head, args.base_sha, args.head_sha)

    for note in verdict.notes:
        print(f"  {note}")

    if verdict.is_void:
        print()
        for reason in verdict.void:
            print(f"::error::VOID: {reason}")
        print(
            "::error::VOID is not a pass. Nothing was compared, so nothing was shown to be "
            "unchanged."
        )
        return verdict.exit_code()

    if verdict.differences:
        print()
        for difference in verdict.differences:
            print(f"::error::{difference}")
        print(
            f"::error::{len(verdict.differences)} behaviour difference(s) between "
            f"{args.base_sha[:12] or 'base'} and {args.head_sha[:12] or 'head'}. A hardening "
            f"change must not alter what the installer does or what a user sees."
        )
        return verdict.exit_code()

    print()
    print(
        f"PASS: {args.base_sha[:12] or 'base'} and {args.head_sha[:12] or 'head'} produced the "
        f"same user-visible output, the same shortcuts and the same installed files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
