# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""UI parity across the arms of a studiobench A/B: does the PR change what is on screen?

The performance question and the parity question are the same experiment run once. studiobench
already drives both arms through the same eighteen scripted actions against a byte-identical
seeded thread inside one session, so at the close of every action the two DOMs are supposed to
agree. This reads the digests taken there and reports, per action, whether they did.

WHAT A MISMATCH MEANS, AND WHAT IT DOES NOT. A differing digest is not automatically a defect and
is never reported as one. Three things produce it:

  a real UI change      the PR renders something different. This is the finding.
  a volatile the
  normaliser missed     a generated id, a rendered duration, a backend-minted record id. The fix
                        is in parity.js, and until it is fixed the reading is noise. The first
                        null control this tool ever ran failed on all eighteen actions for exactly
                        one such volatile; run `parity_null_control.py --hunt` before believing a
                        wall of red.
  a legitimately
  divergent action      `stop_generation` stops a live stream, so how much text arrived before the
                        stop differs between two runs of ANY build.

So the output separates STABLE actions, where a mismatch is a genuine signal, from actions that
are expected to vary. A tool that reported all eighteen equally would be ignored within a day.

WHAT A PASS DOES NOT MEAN. The digest is a serialisation of the DOM. It cannot see CSS coming from
a stylesheet, computed layout, colour, typography, images or canvas pixels. `scene/parity.js`
states the full list at the top of the file, and the bounded computed-style probe reported here
covers three properties on a few dozen elements, no more. "18 actions, 0 differences" is not
"the UI is pixel-identical", and quoting it as that is the misreading this paragraph exists to
prevent.

THE THIRD OUTCOME. An action whose digest could not be captured on one side is NOT COMPARABLE, and
it is counted and printed as its own outcome rather than folded into the passes. An action that
was never measured and an action that matched produce the same silence otherwise, and telling them
apart is the entire difference between an instrument and a decoration.

    python -m tests.studio.studiobench.sweep.ui_parity outputs/mine
    python -m tests.studio.studiobench.sweep.ui_parity --null outputs/null outputs/mine
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):  # pragma: no cover
    # Running the file directly rather than as a module, which is the first thing anyone tries.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import behaviour as B  # noqa: E402
from tests.studio.studiobench.analysis import parity as P  # noqa: E402

# The DECLARED unstable set, each entry carrying its mechanism. It lives in the studiobench
# package rather than here so that a test can require a mechanism for every entry, and so that
# this script and the null control cannot drift into two different opinions about which actions
# are trusted.
#
# Declared is not the same as true. `--null` replaces it with the set MEASURED from a base-vs-base
# run and reports every disagreement in both directions, which is the only way an entry here gets
# audited rather than inherited.
UNSTABLE_ACTIONS = frozenset(P.UNSTABLE_ACTIONS)


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()
    ]


def arm_of(cell_id: str) -> str:
    return "treatment" if ".treatment." in cell_id else "base"


def collect(paths: list[Path]) -> dict:
    """{(shard, rep, action): {arm: action row}} plus a tally of what was captured at all.

    A row whose parity is missing or failed is KEPT, as the failed capture it is. Dropping it here
    would delete the evidence that the surface went unmeasured, and the comparison layer needs to
    see the failure in order to call the pair not comparable.

    The WHOLE ROW is carried, not just its digest, because `ran` is part of what the comparison
    means: a matching digest on an action that never ran is not coverage of that action.
    """
    out: dict[tuple, dict] = collections.defaultdict(dict)
    attempted = missing = 0
    for path in paths:
        shard = path.parent.name
        for r in rows(path):
            if r.get("row_type") != "action":
                continue
            parity = r.get("parity")
            if isinstance(parity, dict) and parity.get("parity_attempted"):
                attempted += 1
            else:
                missing += 1
            cid = r.get("cell_id") or ""
            rep = cid.rsplit(".", 1)[-1]
            out[(shard, rep, r.get("action"))][arm_of(cid)] = r
    return {"pairs": out, "attempted": attempted, "missing": missing}


def any_windowed(paths: list[Path]) -> Optional[str]:
    """Did either arm of this payload mount a WINDOW of the thread rather than all of it?

    DETECTED, not declared. The alternative is a flag the operator sets, and a flag that is
    forgotten produces a full page of red that reads as eighteen UI regressions. The capture
    carries `mounted_messages` and `thread_total`, so the payload answers for itself.
    """
    for path in paths:
        for r in rows(path):
            if r.get("row_type") != "action":
                continue
            parity = r.get("parity")
            if P.windowed_mount(parity):
                return (
                    f"{r.get('cell_id')} / {r.get('action')} mounted "
                    f"{parity.get('mounted_messages')} of {parity.get('thread_total')} messages"
                )
    return None


def behaviour_report(
    paths: list[Path],
    label: str,
    windowed: bool = True,
) -> int:
    """The windowed arm's report: behavioural invariants instead of a structural digest.

    Printed with the reason it is being printed, every time. The one way this could mislead is by
    quietly replacing a strict check with a looser one, so the banner says outright which question
    is no longer being asked.
    """
    got = collect(paths)
    results = []
    for (shard, rep, action), sides in sorted(got["pairs"].items()):
        if "base" not in sides or "treatment" not in sides:
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "reason": f"only the {next(iter(sides))} arm recorded this action",
                        "checks": [],
                    },
                )
            )
            continue
        results.append((action, shard, rep, B.compare_behaviour(sides["base"], sides["treatment"])))

    print(f"\n{label}  (BEHAVIOURAL MODE)")
    print(f"  CLAIM: {P.CLAIM_BEHAVIOURAL}.")
    # WHICH REASON, and only if it is true. Forced behavioural mode on a payload that mounts its
    # whole thread -- which is exactly how a NULL CONTROL is scored on the same scale as the
    # windowed arm it is the control for -- used to print "one arm of this payload mounts a window
    # of the thread" about a payload where neither arm does. A control that misdescribes itself in
    # its own heading is not a small thing when the heading is what gets read.
    if windowed:
        print(
            "  One arm of this payload mounts a window of the thread, so the structural DOM digest"
            "\n  is NOT APPLICABLE: it compares what is on screen, and this arm changes what is on"
            "\n  screen by design. It would report a difference on every action and prove nothing."
        )
    else:
        print(
            "  NEITHER arm of this payload mounts a window; behavioural mode was FORCED. The"
            "\n  structural digest would have been applicable here and was not run, so nothing"
            "\n  below says anything about whether the mounted messages render identically."
        )
    print(
        "  What is scored is the scroll extent plus the behaviours a windowed mount breaks first.\n"
        "  WHAT IS NOT BEING ASKED: whether the mounted messages render identically. An arm that\n"
        "  passes everything below can still have changed how a message looks."
    )
    if not results:
        print("  NO ACTION DATA in this payload.")
        return 2

    broken, unchecked, idle, blind = [], [], [], []
    matched = 0
    for action, shard, rep, r in results:
        verdict = r["verdict"]
        if verdict == B.BROKEN:
            broken.append((action, shard, rep, r))
        elif verdict == P.MATCH:
            matched += 1
        elif verdict == P.NOT_EXERCISED:
            idle.append((action, shard, rep, r))
        elif verdict == P.NOT_COMPARABLE:
            blind.append((action, shard, rep, r))
        else:
            unchecked.append((action, shard, rep, r))

    print(f"\n  {len(results)} action pairs across {len(paths)} shard(s)")
    print(f"  invariants held:            {matched}")
    print(f"  INVARIANTS BROKEN:          {len(broken)}")
    print(f"  UNCHECKED:                  {len(unchecked)}  (no invariant declared; not a pass)")
    print(f"  NOT COMPARABLE:             {len(blind)}")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")

    if broken:
        print("\n  BEHAVIOURAL INVARIANTS BROKEN -- these are user-visible, not measurement noise:")
        for action, shard, rep, r in broken:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
    else:
        print("\n  Every declared behavioural invariant held on both arms.")

    if unchecked:
        names = sorted({a for a, _s, _r, _v in unchecked})
        print(
            f"\n  UNCHECKED -- {len(unchecked)} pair(s) over {len(names)} action(s) with no "
            f"declared invariant. These surfaces carry NO verdict on this arm:"
        )
        for name in names:
            print(f"    {name}")
    if blind:
        print("\n  NOT COMPARABLE:")
        for action, shard, rep, r in blind[:8]:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
    if idle:
        names = sorted({a for a, _s, _r, _v in idle})
        print(f"\n  NOT EXERCISED: {', '.join(names)}")

    if broken:
        return 1
    if matched == 0:
        # NOTHING WAS VALIDATED, which is not the same as nothing being wrong. With every pair
        # unchecked, not comparable or never exercised, `broken` is empty and the block above has
        # just printed "Every declared behavioural invariant held on both arms" -- a sentence that
        # is technically true of an empty set and reads as a pass. This is the same false green
        # the digest path returns 2 for, and it is the more dangerous of the two here, because
        # behavioural mode is what REPLACES the digest on a windowed arm: if it silently validates
        # nothing then a windowed arm has no UI verdict at all while appearing to have passed one.
        print(
            "\n  NOTHING WAS COMPARED. Not one behavioural invariant was evaluated on any pair, "
            "so this run carries no UI verdict -- neither a pass nor a failure. Treat it as an "
            "absent result and find out why the actions did not run."
        )
        return 2
    return 0


def visible_unstable_set(null_paths: list[Path] | None) -> frozenset[str]:
    """Actions whose VISIBLE REGION differs between two runs of the SAME build.

    A floor, measured rather than assumed, and it has to be measured separately from the digest's
    because the two ask different questions. Observed on a 100K base-vs-base control: 13 of 64
    action pairs differed inside the viewport, against 5 for the virtualization arm the control was
    run for. Without this the arm under test scores WORSE than an identical pair of builds and the
    verdict is not merely weak, it is backwards.

    The mechanism is the same one the digest already normalises around and does not fully catch:
    the rows differ at identical character counts (`7609->7609c`), which is a volatile attribute
    rather than changed content.
    """
    if not null_paths:
        return frozenset()
    results, _got = compare_all_with(null_paths, P.compare_visible, "visible")
    return frozenset(
        action
        for action, _shard, _rep, r in results
        if r.get("_ran") and r.get("verdict") == P.DIFFER
    )


def visible_report(
    paths: list[Path],
    label: str,
    unstable: frozenset[str] = frozenset(),
) -> int:
    """VISIBLE-REGION PARITY. The verdict the off-screen exemption asks for.

    Policy: all changes preserve UI and UX idempotency, except that a difference may be accepted
    deliberately when performance improves dramatically, and a difference that exists only OFF
    SCREEN is fine by definition. The whole-document digest cannot express the second exemption --
    it fails every deferred-off-screen technique by construction -- so this scores the claim the
    policy actually cares about.
    """
    results, _got = compare_all_with(paths, P.compare_visible, "visible")

    print(f"\n{label}  (VISIBLE-REGION MODE)")
    print(f"  CLAIM: {P.CLAIM_VISIBLE}.")
    print(
        "  Off-screen differences are EXEMPT by policy and are not reported below. A message that\n"
        "  was visible for any part of the action is compared, in full, even if only partly on\n"
        "  screen. Messages are keyed by THREAD POSITION, so a windowed arm and a fully mounted\n"
        "  one are comparable."
    )
    if not results:
        print("  NO ACTION DATA in this payload.")
        return 2

    differing, unstable_bad, blind, idle, matched = [], [], [], [], 0
    for action, shard, rep, r in results:
        if not r.get("_ran"):
            idle.append((action, shard, rep, r))
        elif r["verdict"] == P.NOT_COMPARABLE:
            blind.append((action, shard, rep, r))
        elif r["verdict"] == P.DIFFER:
            # A SEVERE difference is never routed into the noise floor. See compare_visible: an
            # action can be in the derived unstable set for an unrelated attribute and still be
            # the action on which one arm lost the whole thread.
            noise = action in unstable and not r.get("severe")
            (unstable_bad if noise else differing).append((action, shard, rep, r))
        else:
            matched += 1

    print(f"\n  {len(results)} action pairs across {len(paths)} shard(s)")
    print(f"  visible region matched:     {matched}")
    print(f"  VISIBLE DIFFERENCES:        {len(differing)}")
    print(
        f"  unstable actions differing: {len(unstable_bad)}  (differ against an identical build; "
        "not a verdict)"
    )
    print(f"  NOT COMPARABLE:             {len(blind)}  (never observed; not a pass)")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")

    if differing:
        print(
            "\n  DIFFERENCES INSIDE THE VIEWPORT -- the off-screen exemption does not cover these:"
        )
        for action, shard, rep, r in differing:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
            for m in r.get("moved", [])[:4]:
                print(f"        {m}")
    elif matched:
        print(
            "\n  Every message that reached the viewport was identical on both arms.\n"
            "  Differences outside the viewport are not reported here and are exempt by policy;\n"
            "  run --mode digest if you want the whole-document comparison instead."
        )
    if blind:
        print("\n  NOT COMPARABLE -- these carry no verdict in either direction:")
        for action, shard, rep, r in blind[:8]:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
    if unstable_bad:
        names = sorted({a for a, _s, _r, _v in unstable_bad})
        print(
            f"\n  (reported, not counted) {len(unstable_bad)} pair(s) over {len(names)} action(s) "
            f"whose visible region differs between two runs of the SAME build: {', '.join(names)}"
        )
    if idle:
        names = sorted({a for a, _s, _r, _v in idle})
        print(f"\n  NOT EXERCISED: {', '.join(names)}")
    if not unstable:
        print(
            "\n  NO FLOOR WAS MEASURED. Pass --null OUTDIR of a base-vs-base run: an identical\n"
            "  pair of builds has been observed differing on 13 of 64 pairs inside the viewport,\n"
            "  so an unfloored count here can rank a real arm below two copies of the same build."
        )

    if differing:
        return 1
    if matched == 0:
        # The same false green every other mode here has already been fixed for: nothing compared
        # is not the same as nothing wrong, and it must not exit 0.
        print(
            "\n  NOTHING WAS COMPARED. Not one action pair yielded a visible-region verdict, so\n"
            "  this run carries no UI verdict at all -- neither a pass nor a failure."
        )
        return 2
    return 0


def compare_all_with(paths: list[Path], compare, key: str) -> tuple[list[tuple], dict]:
    """[(action, shard, rep, result)] using `compare` over payload sub-object `key`."""
    got = collect(paths)
    results = []
    for (shard, rep, action), sides in sorted(got["pairs"].items()):
        if "base" not in sides or "treatment" not in sides:
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "moved": [],
                        "_ran": True,
                        "reason": f"only the {next(iter(sides))} arm recorded this action",
                    },
                )
            )
            continue
        ran = bool(sides["base"].get("ran")) and bool(sides["treatment"].get("ran"))
        out = compare(sides["base"].get(key), sides["treatment"].get(key))
        out["_ran"] = ran
        results.append((action, shard, rep, out))
    return results, got


def compare_all(paths: list[Path]) -> tuple[list[tuple], dict]:
    """[(action, shard, rep, compare-result)] over every base/treatment pair found."""
    got = collect(paths)
    results = []
    for (shard, rep, action), sides in sorted(got["pairs"].items()):
        if "base" not in sides or "treatment" not in sides:
            # One arm never produced this row at all. Recorded rather than skipped: an action that
            # ran on one arm and not the other is itself a difference between the arms.
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "moved": [],
                        "reason": f"only the {next(iter(sides))} arm recorded this action",
                        "style_verdict": P.NOT_COMPARABLE,
                        "style_reason": "",
                    },
                )
            )
            continue
        results.append((action, shard, rep, P.compare_rows(sides["base"], sides["treatment"])))
    return results, got


def unstable_set(paths: list[Path] | None) -> tuple[frozenset[str], dict, dict]:
    """The unstable set to score with, derived from a null control when one is supplied.

    DERIVED BEATS DECLARED, and the declared set is kept as the cross-check rather than thrown
    away: an entry that the null control never saw differ is costing real signal, and an action
    that differs against itself without being declared is producing noise. Both are printed.
    """
    if not paths:
        return UNSTABLE_ACTIONS, {}, {}
    results, _ = compare_all(paths)
    derived = P.derive_unstable([(a, r) for a, _s, _rep, r in results])
    checks = P.cross_check(derived, UNSTABLE_ACTIONS)
    measured = frozenset(a for a, row in derived.items() if row["unstable"])
    # UNION, not replacement. An action the null control could not reach -- `image_upload` has no
    # visible attachments button on this fixture -- would otherwise silently move from "declared
    # unstable" to "stable" on the strength of a measurement that never happened.
    return measured | UNSTABLE_ACTIONS, derived, checks


def report(paths: list[Path], label: str, unstable: frozenset[str]) -> int:
    results, got = compare_all(paths)
    if not results:
        # An empty result is reported as an empty result. "No mismatches found" when nothing was
        # ever compared is the exact shape of a check that silently does nothing.
        print(
            f"\n{label}: NO PARITY DATA in {len(paths)} payload(s). "
            f"{got['missing']} action rows carried no digest. "
            f"Was this run recorded before the parity instrument existed?"
        )
        return 2

    stable_bad, unstable_bad, blind, style_bad, idle = [], [], [], [], []
    inapplicable: list[tuple] = []
    matched = 0
    for action, shard, rep, r in results:
        if r["verdict"] == P.NOT_APPLICABLE:
            # NEITHER A PASS NOR A FAIL. The digest is the wrong question for this pair; see
            # analysis/parity.applicability. It is bucketed separately so it cannot be summed into
            # either column, and `--mode behaviour` is what answers it instead.
            inapplicable.append((action, shard, rep, [r.get("reason", "")]))
            continue
        if r["verdict"] == P.NOT_EXERCISED:
            idle.append((action, shard, rep, [r.get("reason", "")]))
            continue
        if r["verdict"] == P.NOT_COMPARABLE:
            blind.append((action, shard, rep, [r.get("reason", "")]))
            continue
        if r["style_verdict"] == P.DIFFER:
            style_bad.append((action, shard, rep, [r.get("style_reason", "")]))
        if r["verdict"] == P.MATCH:
            matched += 1
            continue
        entry = (action, shard, rep, r["moved"])
        (unstable_bad if action in unstable else stable_bad).append(entry)

    print(f"\n{label}  (STRUCTURAL MODE)")
    print(f"  CLAIM: {P.CLAIM_STRUCTURAL}.")
    print(
        "  Under the current policy an OFF-SCREEN-ONLY difference is exempt, and this mode cannot\n"
        "  tell an off-screen difference from an on-screen one. Use --mode visible for a payload\n"
        "  that defers off-screen work deliberately."
    )
    print(f"  {len(results)} action pairs across {len(paths)} shard(s)")
    print(f"  matched:                    {matched}")
    print(f"  stable actions differing:   {len(stable_bad)}")
    print(f"  unstable actions differing: {len(unstable_bad)}  (expected to vary; not a verdict)")
    print(f"  NOT COMPARABLE:             {len(blind)}  (never measured; not a pass)")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")
    if inapplicable:
        print(
            f"  NOT APPLICABLE:             {len(inapplicable)}  (a windowed mount; the digest "
            f"cannot answer this)"
        )
        print(f"    {inapplicable[0][3][0]}")
        print("    Re-run with --mode behaviour to score these on behavioural invariants.")
    print(
        f"  style probe differing:      {len(style_bad)}  (advisory: display/visibility/"
        f"pointer-events)"
    )

    if stable_bad:
        print("\n  UI PARITY DIFFERENCES ON STABLE ACTIONS -- these need explaining:")
        for action, shard, rep, moved in stable_bad:
            print(f"    {action:<26} {shard} {rep}: {', '.join(moved[:4])}")
    elif matched == 0 and inapplicable:
        # NOT A PASS, and it must not print like one. Every pair was refused, so "no stable action
        # rendered differently" would be true, reassuring and about nothing -- the exact shape of
        # a check that silently does nothing, which is what the NOT COMPARABLE bucket exists to
        # prevent elsewhere in this file.
        print(
            "\n  NOTHING WAS COMPARED. Every pair in this payload is a windowed mount, which the "
            "structural digest cannot answer. This is not a pass."
        )
    else:
        print(
            "\n  No stable action rendered a different THREAD STRUCTURE between the two arms.\n"
            "  Read that literally. This digest walks the thread and nothing else: it is\n"
            "  sidebar-blind and layout-blind by construction, and it never reads geometry or\n"
            "  CSS custom properties. A change confined to the sidebar, or one that moves things\n"
            "  without restructuring the thread, passes here while being invisible to it.\n"
            "  This is not a statement that the UI is unchanged."
        )

    if blind:
        print("\n  NOT COMPARABLE -- these surfaces carry no verdict in either direction:")
        for action, shard, rep, why in blind[:8]:
            print(f"    {action:<26} {shard} {rep}: {why[0]}")

    if idle:
        # Named surfaces, deduplicated: what matters is WHICH actions this run never opened, not
        # that it failed to open one of them sixteen separate times.
        names = sorted({action for action, _s, _r, _w in idle})
        print(
            f"\n  NOT EXERCISED -- {len(idle)} pair(s) over {len(names)} action(s) that did not "
            f"run. These surfaces are UNCHECKED, not unchanged:"
        )
        for name in names:
            why = next(w[0] for a, _s, _r, w in idle if a == name)
            print(f"    {name:<26} {why}")

    if style_bad:
        print("\n  (advisory) the bounded computed-style probe differed:")
        for action, shard, rep, why in style_bad[:8]:
            print(f"    {action:<26} {shard} {rep}: {why[0]}")

    if unstable_bad:
        print("\n  (reported, not counted) actions that vary between runs of any build:")
        for action, shard, rep, moved in unstable_bad[:8]:
            print(f"    {action:<26} {shard} {rep}: {', '.join(moved[:3])}")
    if stable_bad:
        return 1
    # 2, the same code the empty-payload path uses, and for the same reason: the tool was asked a
    # question it could not answer. Exiting 0 here would let CI go green on a run where the parity
    # check was structurally incapable of firing.
    if matched == 0 and inapplicable:
        return 2
    return 0


def tier_of(paths: list[Path]) -> set[str]:
    """The tier(s) a payload was recorded at, from its own run_meta row.

    Instability is a property of the FILM, not only of the action. On the fast film `copy_markdown`
    opens 2.7 s after a `send_turn` and lands inside that turn's stream, so it differs against
    itself; on the standard film the same action opens 26 s later against a finished reply. An
    unstable set derived from one tier and applied to another would silence a real signal at one
    end and admit noise at the other, so the mismatch is said out loud rather than assumed away.
    """
    tiers: set[str] = set()
    for path in paths:
        for r in rows(path):
            if r.get("row_type") == "run_meta" and r.get("tier"):
                tiers.add(r["tier"])
    return tiers


def shards_of(pattern: str) -> list[Path]:
    root = Path(pattern).parent if "/" in pattern else Path(".")
    stem = Path(pattern).name
    return sorted(p / "payload.jsonl" for p in root.glob(stem) if (p / "payload.jsonl").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("payloads", nargs = "+", help = "studiobench output dirs (globs allowed)")
    ap.add_argument(
        "--null",
        metavar = "OUTDIR",
        action = "append",
        default = [],
        help = "a base-vs-base run to derive the unstable set from",
    )
    ap.add_argument(
        "--mode",
        choices = ("auto", "digest", "visible", "behaviour"),
        default = "auto",
        help = (
            "auto (default) reads the payload: a fully mounted pair is scored structurally, a "
            "windowed pair is scored on the VISIBLE REGION and then on behavioural invariants; "
            "digest forces the whole-document structural comparison; visible forces the "
            "visible-region one; behaviour forces the behavioural one"
        ),
    )
    args = ap.parse_args(argv)

    # THE MODE DECISION FIRST, before the unstable set is even derived. Deriving an unstable set
    # from a null control and then not using it because the payload is windowed would print a page
    # of scoring apparatus that has no bearing on the report underneath it.
    if args.mode != "digest":
        # PER PAYLOAD, not once for the whole invocation. Deciding `auto` from the first payload
        # that happens to be windowed and then applying it to all of them means
        # `ui_parity normal_run windowed_run` scores the NORMAL run behaviourally too, skipping
        # its structural digest entirely -- so an ordinary DOM regression in a fully-mounted arm
        # goes unreported because an unrelated payload on the same command line was windowed.
        # `--mode behaviour` still forces every payload, because that is what forcing means.
        decided = []
        for pattern in args.payloads:
            paths = shards_of(pattern)
            if not paths:
                continue
            why = any_windowed(paths) if args.mode == "auto" else f"forced by --mode {args.mode}"
            if why:
                decided.append((pattern, paths, why))
        if decided:
            worst = 0
            for pattern, paths, why in decided:
                print(f"WINDOWED MOUNT DETECTED in {pattern}: {why}")
                # BOTH, and in this order. Visible-region parity is the verdict the off-screen
                # exemption asks for and it is the one that can FAIL a windowed arm for something
                # a user would see. The behavioural invariants are the complement: they catch what
                # a viewport comparison cannot, such as a clipboard that no longer carries the
                # thread. Neither subsumes the other, so a windowed pair gets both and the run
                # fails if either does.
                if args.mode in ("auto", "visible"):
                    # The floor, derived from the null control the caller passed. Without it an
                    # identical pair of builds outscores the arm under test.
                    vis_null: list[Path] = []
                    for pat in args.null:
                        vis_null.extend(shards_of(pat))
                    worst = max(
                        worst,
                        visible_report(
                            paths,
                            f"UI PARITY: {pattern}",
                            visible_unstable_set(vis_null),
                        ),
                    )
                if args.mode in ("auto", "behaviour"):
                    worst = max(
                        worst,
                        behaviour_report(
                            paths,
                            f"UI PARITY: {pattern}",
                            windowed = any_windowed(paths) is not None,
                        ),
                    )
            remaining = [p for p in args.payloads if p not in {d[0] for d in decided}]
            if remaining:
                # The rest still get the digest they were owed, in the same run.
                print(
                    f"\n  {len(remaining)} payload(s) mount their whole thread and are scored "
                    "structurally below, not behaviourally."
                )
                args.payloads = remaining
            else:
                return worst
            _digest_floor = worst
        else:
            _digest_floor = 0
    else:
        _digest_floor = 0

    null_paths: list[Path] = []
    for pattern in args.null:
        null_paths.extend(shards_of(pattern))
    unstable, derived, checks = unstable_set(null_paths or None)
    if derived:
        print(f"UNSTABLE SET DERIVED from {len(null_paths)} null-control shard(s)")
        for key, entries in checks.items():
            print(f"  {key:<32} {', '.join(entries) if entries else '(none)'}")
        print(
            f"  scoring against {len(unstable)} unstable action(s): "
            f"{', '.join(sorted(unstable))}"
        )
    else:
        print(f"UNSTABLE SET DECLARED, not measured: {', '.join(sorted(unstable))}")
        print("  pass --null OUTDIR of a base-vs-base run to derive it instead.")

    null_tiers = tier_of(null_paths)
    worst = 0
    for pattern in args.payloads:
        paths = shards_of(pattern)
        if not paths:
            print(f"\nno payload found for {pattern}")
            worst = max(worst, 2)
            continue
        tiers = tier_of(paths)
        if null_tiers and tiers and null_tiers != tiers:
            print(
                f"\n  WARNING: the null control was recorded at tier {sorted(null_tiers)} and "
                f"this payload at {sorted(tiers)}. Which actions are unstable depends on the "
                f"film's slot spacing, so this unstable set does not transfer."
            )
        worst = max(worst, report(paths, f"UI PARITY: {pattern}", unstable))
    # A behavioural failure on one payload is not cancelled by a structural pass on another.
    return max(worst, _digest_floor)


if __name__ == "__main__":
    raise SystemExit(main())
