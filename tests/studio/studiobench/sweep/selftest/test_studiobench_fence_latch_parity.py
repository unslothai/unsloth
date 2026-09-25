# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A code fence the reader scrolled past on one arm is not a UI change.

THE FALSE ALARM THIS HOLDS. `code-fence-defer.tsx` renders a code fence nobody has come near as a
plain shell and upgrades it to token spans, one way only, the first time it comes within a viewport.
On the r100K fast film `reasoning_toggle` leaves the viewport either near the tail or ~20,000px
higher, at random per cell on ONE build, and the higher landing latches every fence in msg11/13/15
for the rest of the cell. Their serialisations then differ by up to 1.7 million characters between
two runs of the same build. A null control whose four cells all happened to land low called every
later action stable, and two backend-only pull requests failed the gate on 2026-09-23:

    #11727  run 35913031644  model_change / reasoning_toggle / select_all_copy / settings,
                             msg15(assistant):1847269->2150610c, null control quiet
    #11724  run 35913021155  image_upload, msg11(assistant):174956->283962c, the null control
                             could not decide image_upload (one arm missed its slot)

`testdata/` holds both runs' payloads as recorded by CI, trimmed to the fields the verdict reads
(the verdict text is byte-identical on the trimmed and the full files). They predate the per-fence
readings, so the first tests replay the failure as it happened. The rest add the readings that
`scene/parity.js` now takes, modelled on the mechanism: in the messages whose size moved with the
census's highlight-span count (msg11/13/15/17), each distinct digest is a distinct set of latched
fences over identical fence TEXT and an identical message outside the fences. That model is the
claim under test; the live evidence for it is a null run recorded with the new capture, and the
DOM-level half is `test_the_capture_reads_a_shell_and_its_highlighted_fence_as_one_text` below.

Every "clears" test is paired with one that injects a genuine difference into the same recorded
payload and requires it to still fail, because a fix for a false alarm that also silences real
changes is the worse bug.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.sweep import ui_parity as U  # noqa: E402

TESTDATA = Path(__file__).resolve().parent / "testdata"
PARITY_JS = Path(__file__).resolve().parents[2] / "scene" / "parity.js"

#: The messages whose serialised size tracked `census.highlight_spans` in the recorded cells.
LATCHING = (11, 13, 15, 17)
#: Fences per latching message in the model. Fence 0 is the one every cell latches at mount.
FENCES = 12


def _load(name: str) -> list[dict]:
    with gzip.open(TESTDATA / name, "rt", encoding = "utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(rows: list[dict], path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    with open(path / "payload.jsonl", "w", encoding = "utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _captures(rows: list[dict]):
    for row in rows:
        if row.get("row_type") == "action" and isinstance(row.get("parity"), dict):
            yield row, row["parity"]


def annotate(*payloads: list[dict]) -> None:
    """Add the fence readings `scene/parity.js` now records, IN PLACE, across one run's payloads.

    Built over BOTH payloads of a run together, so a digest seen in the null control and in the
    result maps to the same latch set in each.
    """
    seen: dict[int, list[str]] = {i: [] for i in LATCHING}
    for rows in payloads:
        for _row, cap in _captures(rows):
            for m in cap.get("messages") or []:
                i = m.get("i")
                if i in seen and m.get("digest") not in seen[i]:
                    seen[i].append(m["digest"])
    for rows in payloads:
        for _row, cap in _captures(rows):
            for m in cap.get("messages") or []:
                i = m.get("i")
                if i not in seen:
                    continue
                j = seen[i].index(m["digest"])
                assert j < 2 ** (FENCES - 1), "more distinct digests than latch sets"
                m["fences"] = [
                    {
                        "latched": latched,
                        "digest": f"f{i}.{k}.{'hl' if latched else 'shell'}",
                        "text": f"t{i}.{k}",
                    }
                    for k in range(FENCES)
                    for latched in [k == 0 or bool((j >> (k - 1)) & 1)]
                ]
                m["digest_unfenced"] = f"u{i}"


def _run(tmp_path, capsys, result, null) -> tuple[int, str]:
    """The workflow's verdict step, flag for flag: (exit code, what it printed)."""
    rdir = _write(result, tmp_path / "parity-result")
    ndir = _write(null, tmp_path / "parity-null-control")
    rc = U.main(["--min-reps", "2", "--min-compared", "16", "--null", str(ndir), str(rdir)])
    return rc, capsys.readouterr().out


def _stable_section(out: str) -> str:
    """The lines under the heading the exit code is taken from."""
    head = "UI PARITY DIFFERENCES ON STABLE ACTIONS"
    if head not in out:
        return ""
    block = out.split(head, 1)[1]
    lines = []
    for line in block.splitlines()[1:]:
        if not line.startswith("    "):
            break
        lines.append(line)
    return "\n".join(lines)


def _runs():
    return [
        pytest.param(
            "pr11727",
            {"model_change", "reasoning_toggle", "select_all_copy", "settings"},
            id = "11727-quiet-null",
        ),
        pytest.param("pr11724", {"image_upload"}, id = "11724-undecided-null"),
    ]


def _recorded(tag: str) -> tuple[list[dict], list[dict]]:
    return _load(f"{tag}_result.payload.jsonl.gz"), _load(f"{tag}_null.payload.jsonl.gz")


# ── the failure, replayed from what CI recorded ──────────────────────────────────────────────


@pytest.mark.parametrize("tag,flagged", _runs())
def test_the_recorded_run_reproduces_the_false_alarm(tmp_path, capsys, tag, flagged):
    """As recorded, without fence readings, the verdict fails exactly as CI did."""
    result, null = _recorded(tag)
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    section = _stable_section(out)
    for action in flagged:
        assert action in section, (action, section)
    # And every one of them is an assistant message in the latching set, which is the whole finding.
    for line in section.splitlines():
        for claim in line.split(": ", 1)[1].split(", "):
            assert any(claim.startswith(f"msg{i}(assistant)") for i in LATCHING), line


@pytest.mark.parametrize("tag,flagged", _runs())
def test_fence_readings_clear_the_false_alarm(tmp_path, capsys, tag, flagged):
    """With the latch recorded, every one of those differences is a latch and the run passes."""
    result, null = _recorded(tag)
    annotate(result, null)
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 0, out
    assert _stable_section(out) == ""
    # Said, and not paid for in coverage: a latch-only pair is a comparison, not a refusal, so the
    # --min-compared floor (exit 3) is not what turned this green.
    assert "of which fence-latch only" in out


@pytest.mark.parametrize("tag,flagged", _runs())
def test_the_workflow_null_audit_still_passes_with_fence_readings(tmp_path, capsys, tag, flagged):
    result, null = _recorded(tag)
    annotate(result, null)
    rdir = _write(result, tmp_path / "parity-result")
    ndir = _write(null, tmp_path / "parity-null-control")
    rc = U.main(
        [
            "--audit-null",
            "--allow-undecided",
            "image_upload",
            "--compared-in",
            str(rdir),
            "--min-reps",
            "2",
            str(ndir),
        ]
    )
    assert rc == 0, capsys.readouterr().out


# ── the other direction: a real change on the same recorded payload still fails ────────────────


def _treatment_rows(rows: list[dict], action: str):
    for row, cap in _captures(rows):
        if row.get("action") == action and ".treatment." in row.get("cell_id", ""):
            yield row, cap


def _msg(cap: dict, i: int) -> dict:
    return next(m for m in cap["messages"] if m["i"] == i)


@pytest.mark.parametrize("tag,flagged", _runs())
def test_a_real_message_change_on_a_stable_action_still_fails(tmp_path, capsys, tag, flagged):
    """A message with no fences, rendered differently by head in both repetitions of `settings`."""
    result, null = _recorded(tag)
    annotate(result, null)
    hit = 0
    for _row, cap in _treatment_rows(result, "settings"):
        m = _msg(cap, 3)
        m["digest"] = "0badc0de"
        m["chars"] += 17
        hit += 1
    assert hit == 2
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    assert "settings" in _stable_section(out) and "msg3(assistant)" in _stable_section(out)


@pytest.mark.parametrize("tag,flagged", _runs())
def test_a_change_outside_the_fences_of_a_latched_message_still_fails(
    tmp_path, capsys, tag, flagged
):
    """msg15 differs by latch AND by something outside its fences: the latch cannot excuse it."""
    result, null = _recorded(tag)
    annotate(result, null)
    for _row, cap in _treatment_rows(result, "settings"):
        m = _msg(cap, 15)
        m["digest_unfenced"] = "u15-changed"
        m["digest"] = m["digest"] + "x"
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    assert "msg15(assistant)" in _stable_section(out)


@pytest.mark.parametrize("tag,flagged", _runs())
def test_a_highlighting_change_on_a_fence_both_arms_latched_still_fails(
    tmp_path, capsys, tag, flagged
):
    """Fence 0 is latched on every arm, so its token markup is compared in full."""
    result, null = _recorded(tag)
    annotate(result, null)
    for _row, cap in _treatment_rows(result, "settings"):
        m = _msg(cap, 17)
        m["fences"][0]["digest"] = "f17.0.hl-changed"
        m["digest"] = m["digest"] + "x"
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    assert "msg17(assistant)" in _stable_section(out)


@pytest.mark.parametrize("tag,flagged", _runs())
def test_a_build_whose_fences_never_upgrade_is_not_excused(tmp_path, capsys, tag, flagged):
    """Every fence a shell on head: the one regression the text comparison alone would swallow."""
    result, null = _recorded(tag)
    annotate(result, null)
    for row, cap in _captures(result):
        if ".treatment." not in row.get("cell_id", ""):
            continue
        for m in cap["messages"]:
            if "fences" in m:
                m["fences"] = [
                    {**f, "latched": False, "digest": f"f{m['i']}.{k}.shell"}
                    for k, f in enumerate(m["fences"])
                ]
                m["digest"] = f"never-{m['i']}"
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    assert "msg17(assistant)" in _stable_section(out)


# ── the rule itself ──────────────────────────────────────────────────────────────────────────


def _cap(
    fences: list[dict],
    unfenced: str = "u",
    digest: str = "d",
    role: str = "assistant",
):
    msg = {"i": 0, "role": role, "digest": digest, "chars": 10}
    if fences is not None:
        msg["fences"] = fences
        msg["digest_unfenced"] = unfenced
    return {
        "parity_attempted": True,
        "root_kind": "thread",
        "digest": digest,
        "digest_scaffold": "s",
        "chars_scaffold": 5,
        "messages": [msg],
        "overlays": [],
        "styles": {"digest": "st", "elements": 3, "capped": False},
    }


def _f(
    latched: bool,
    text: str = "t",
    digest: str | None = None,
    lang: str = "python",
) -> dict:
    return {
        "latched": latched,
        "lang": lang,
        "text": text,
        "digest": digest or ("hl" if latched else "sh"),
    }


def test_a_latch_only_difference_is_a_match_not_a_refusal():
    base = _cap([_f(True), _f(False)], digest = "a")
    treat = _cap([_f(True), _f(True)], digest = "b")
    got = P.compare(base, treat)
    assert got["verdict"] == P.MATCH, got
    assert got["fence_latch"] == [0]


@pytest.mark.parametrize(
    "treat_fences,unfenced,why",
    [
        ([_f(True), _f(True, text = "other")], "u", "the one-sided fence's text changed"),
        ([_f(True, digest = "hl2"), _f(True)], "u", "a fence latched on both arms changed"),
        ([_f(True)], "u", "a fence vanished"),
        ([_f(True), _f(True)], "u2", "something outside the fences changed"),
        ([_f(True), _f(False, digest = "sh2")], "u", "no fence changed latch; the shell did"),
        (
            [_f(True), _f(True, lang = "javascript")],
            "u",
            "the one-sided fence's language changed",
        ),
    ],
)
def test_anything_but_the_latch_still_differs(treat_fences, unfenced, why):
    base = _cap([_f(True), _f(False)], digest = "a")
    treat = _cap(treat_fences, unfenced = unfenced, digest = "b")
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, why
    assert got["fence_latch"] == [], why


def test_one_arm_latching_nothing_anywhere_is_never_excused():
    base = _cap([_f(True), _f(False)], digest = "a")
    treat = _cap([_f(False), _f(False)], digest = "b")
    assert P.fence_latch_residue(base, treat) == []
    assert P.compare(base, treat)["verdict"] == P.DIFFER


def test_a_payload_recorded_before_the_fence_readings_is_scored_as_before():
    base, treat = _cap(None, digest = "a"), _cap(None, digest = "b")
    assert P.fence_latch_residue(base, treat) == []
    assert P.compare(base, treat)["verdict"] == P.DIFFER
    # And a message whose digest agrees is never reported, whatever its fences say.
    same = _cap([_f(True)], digest = "a")
    assert P.fence_latch_residue(same, _cap([_f(False)], digest = "a")) == []


def test_the_derived_null_counts_a_latch_only_pair_as_stable():
    """`derive_unstable` reads `compare`, so a latch-only null pair is an observation of stability."""
    base = _cap([_f(True), _f(False)], digest = "a")
    treat = _cap([_f(True), _f(True)], digest = "b")
    row = P.derive_unstable([("settings", P.compare(base, treat))] * 2)["settings"]
    assert row["unstable"] is False and row["observations"] == 2


# ── the capture, the real parity.js under node ───────────────────────────────────────────────

HARNESS_JS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const window = {};
const document = { body: { tagName: "BODY", attributes: [], childNodes: [],
                           getAttribute: () => null },
                   querySelectorAll: () => [] };
window.getComputedStyle = () => ({ getPropertyValue: () => "" });
(new Function("window", "document", src))(window, document);
const build = (spec) => {
  if (typeof spec === "string") return { nodeType: 3, nodeValue: spec };
  const attrs = spec.attrs || {};
  return {
    nodeType: 1,
    tagName: (spec.tag || "div").toUpperCase(),
    attributes: Object.keys(attrs).map((name) => ({ name })),
    getAttribute: (name) => (name in attrs ? attrs[name] : null),
    childNodes: (spec.children || []).map(build),
  };
};
const P = window.__sb.parity;
const specs = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = specs.map((s) => {
  const el = build(s);
  const r = P.messageReading(el);
  return { fences: r.fences || null, digest_unfenced: r.digest_unfenced || null,
           digest: P.hash(r.sig), same_as_signature: r.sig === P.signature(build(s)) };
});
process.stdout.write(JSON.stringify(out));
"""

CODE = "def f(x):\n    return x + 1\n\nprint(f(2))"


def _shell(code: str) -> dict:
    """What `code-fence-defer.tsx`'s FenceShell renders for a fence nobody has reached."""
    return {
        "attrs": {
            "data-streamdown": "code-block",
            "data-unsloth-fence-deferred": "true",
            "data-language": "python",
            "class": "my-4 flex",
        },
        "children": [
            {
                "attrs": {"data-streamdown": "code-block-header", "data-language": "python"},
                "children": [{"tag": "span", "children": ["python"]}],
            },
            {
                "attrs": {"data-streamdown": "code-block-body", "data-language": "python"},
                "children": [{"tag": "pre", "children": [{"tag": "code", "children": [code]}]}],
            },
        ],
    }


def _highlighted(code: str, lang: str = "python") -> dict:
    """streamdown's highlighted fence: one span per line, NO newline between lines, a span per token."""
    lines = []
    for line in code.split("\n"):
        tokens = [t for t in line.replace(" ", "\0 \0").split("\0") if t] or ["\n"]
        lines.append(
            {
                "tag": "span",
                "attrs": {"class": "block"},
                "children": [
                    {"tag": "span", "attrs": {"style": "--sdm-c:#fff"}, "children": [t]}
                    for t in tokens
                ],
            }
        )
    return {
        "attrs": {
            "data-streamdown": "code-block",
            "data-language": lang,
            "class": "my-4 flex",
            "style": "content-visibility:auto",
        },
        "children": [
            {
                "attrs": {"data-streamdown": "code-block-header", "data-language": lang},
                "children": [{"tag": "span", "children": [lang]}],
            },
            {
                "attrs": {"data-streamdown": "code-block-actions"},
                "children": [{"tag": "button", "attrs": {"title": "Copy Code"}}],
            },
            {
                "attrs": {"data-streamdown": "code-block-body", "data-language": lang},
                "children": [{"tag": "pre", "children": [{"tag": "code", "children": lines}]}],
            },
        ],
    }


def _message(fence: dict, prose: str = "Here is the function.") -> dict:
    return {
        "attrs": {"data-role": "assistant"},
        "children": [{"tag": "p", "children": [prose]}, fence, {"tag": "p", "children": ["Done."]}],
    }


def _node_readings(tmp_path: Path, specs: list[dict]) -> list[dict]:
    exe = shutil.which("node") or shutil.which("nodejs")
    if not exe:
        pytest.skip("node is not installed, so the shipped parity.js could not be evaluated")
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS_JS, encoding = "utf-8")
    spec_file = tmp_path / "specs.json"
    spec_file.write_text(json.dumps(specs), encoding = "utf-8")
    got = subprocess.run(
        [exe, str(harness), str(PARITY_JS), str(spec_file)],
        capture_output = True,
        text = True,
        timeout = 60,
        check = True,
    )
    return json.loads(got.stdout)


def test_the_capture_reads_a_shell_and_its_highlighted_fence_as_one_text(tmp_path):
    shell, lit, edited, reworded, bare = _node_readings(
        tmp_path,
        [
            _message(_shell(CODE)),
            _message(_highlighted(CODE)),
            _message(_highlighted(CODE.replace("x + 1", "x + 2"))),
            _message(_shell(CODE), prose = "Here is a different sentence."),
            {
                "attrs": {"data-role": "assistant"},
                "children": [{"tag": "p", "children": ["no code"]}],
            },
        ],
    )
    # The false alarm, reproduced at the DOM: one fence, two serialisations.
    assert shell["digest"] != lit["digest"]
    assert shell["fences"][0]["latched"] is False and lit["fences"][0]["latched"] is True
    # ...and the reading that resolves it: same text, same message around it.
    assert shell["fences"][0]["text"] == lit["fences"][0]["text"]
    assert shell["digest_unfenced"] == lit["digest_unfenced"]
    # The other direction: a changed line of code and a changed sentence are both still seen.
    assert edited["fences"][0]["text"] != lit["fences"][0]["text"]
    assert reworded["digest_unfenced"] != shell["digest_unfenced"]
    # The digest every existing caller compares is unchanged, and a fence-free message carries no
    # new fields at all.
    assert all(r["same_as_signature"] for r in (shell, lit, edited, reworded, bare))
    assert bare["fences"] is None and bare["digest_unfenced"] is None


def test_the_capture_keeps_spacing_indentation_and_language_in_a_fence(tmp_path):
    """Only line breaks separate the two forms, so only line breaks are dropped from the text."""
    lit, spaced, dedented, quoted, other_lang = _node_readings(
        tmp_path,
        [
            _message(_highlighted(CODE)),
            _message(_shell(CODE.replace("x + 1", "x+1"))),
            _message(_shell(CODE.replace("    return", "  return"))),
            _message(_shell(CODE.replace("f(2)", "f( 2)"))),
            _message(_highlighted(CODE, lang = "javascript")),
        ],
    )
    for got, why in (
        (spaced, "spaces around an operator"),
        (dedented, "an indentation change"),
        (quoted, "a space inside the call"),
    ):
        assert got["fences"][0]["latched"] is False and lit["fences"][0]["latched"] is True
        assert got["fences"][0]["text"] != lit["fences"][0]["text"], why
    assert other_lang["fences"][0]["text"] == lit["fences"][0]["text"]
    assert other_lang["fences"][0]["lang"] == "javascript" and lit["fences"][0]["lang"] == "python"


def test_the_capture_keeps_where_a_fence_breaks_its_lines(tmp_path):
    """Lines, not a run of characters: a line break moved within the code is a different fence."""
    lit, moved, merged, trailing, blank_lit, blank_shell = _node_readings(
        tmp_path,
        [
            _message(_highlighted("a\nbc")),
            _message(_shell("ab\nc")),
            _message(_shell("abc")),
            _message(_shell("a\nbc\n\n")),
            _message(_highlighted("a\n\n\nbc")),
            _message(_shell("a\n\n\nbc")),
        ],
    )
    assert moved["fences"][0]["text"] != lit["fences"][0]["text"], "a line break moved"
    assert merged["fences"][0]["text"] != lit["fences"][0]["text"], "two lines joined"
    # The shell trims trailing newlines; that is not a difference.
    assert trailing["fences"][0]["text"] == lit["fences"][0]["text"]
    # Blank lines render as a line holding a lone newline, and count once, as in the shell.
    assert blank_lit["fences"][0]["text"] == blank_shell["fences"][0]["text"]
    assert blank_lit["fences"][0]["text"] != lit["fences"][0]["text"]


def test_the_capture_keeps_literals_that_look_volatile_in_prose(tmp_path):
    """The time and id placeholders are for UI prose; inside code a literal is the content."""
    lit, duration, clock = _node_readings(
        tmp_path,
        [
            _message(_highlighted('timeout = "295ms"\nat = "10:30"')),
            _message(_shell('timeout = "310ms"\nat = "10:30"')),
            _message(_shell('timeout = "295ms"\nat = "11:45"')),
        ],
    )
    assert duration["fences"][0]["text"] != lit["fences"][0]["text"], "a duration literal changed"
    assert clock["fences"][0]["text"] != lit["fences"][0]["text"], "a time literal changed"
