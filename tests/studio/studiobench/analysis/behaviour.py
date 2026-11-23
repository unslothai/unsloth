# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behaviour checks for windowed arms: the DOM may differ by design, but what a user does must not."""

from __future__ import annotations

from typing import Any, Optional

from .parity import MATCH, NOT_APPLICABLE, NOT_COMPARABLE, NOT_EXERCISED

# Distinct from the digest's DIFFER so a report cannot conflate the two kinds of evidence.
BROKEN = "broken"

# Two runs of one build never produce identical character counts once a stream is involved.
EXACT_TOLERANCE = 0.02

# Looser: a windowed list estimates row heights; the failure is an extent that is a fraction of real.
EXTENT_TOLERANCE = 0.10

# Two-sided against the thread: the lower bound catches truncation (0.61 on a windowed arm), the
# upper catches store-serialised reasoning/tool output (2.16). Rendered text vs markdown differ ~0.9%.
MIN_CLIPBOARD_COVERAGE = 0.95
MAX_CLIPBOARD_COVERAGE = 1.10

CLIPBOARD_COVERAGE_REQUIRED = 1.0


def _drift(a: Any, b: Any) -> Optional[float]:
    """Proportional difference, or None when either side is missing or both are zero."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    biggest = max(abs(a), abs(b))
    if biggest == 0:
        return 0.0
    return abs(a - b) / biggest


def _check(
    name: str,
    ok: Optional[bool],
    detail: str,
    *,
    required: bool = False,
) -> dict:
    """A required check that cannot be read makes the pair NOT COMPARABLE, because silence is not a pass."""
    return {"invariant": name, "ok": ok, "detail": detail, "required": required}


def scroll_extent(base_row: dict, treat_row: dict) -> dict:
    """Does the scrollbar still describe the whole conversation on both arms?"""
    bc = base_row.get("census") or {}
    tc = treat_row.get("census") or {}
    b, t = bc.get("viewport_scroll_height"), tc.get("viewport_scroll_height")
    drift = _drift(b, t)
    if drift is None:
        return _check(
            "scroll_extent",
            None,
            f"no viewport scroll height in one of the censuses (base={b!r}, treatment={t!r})",
        )
    return _check(
        "scroll_extent",
        drift <= EXTENT_TOLERANCE,
        f"viewport scrollHeight {b} vs {t} ({drift:.1%} drift, {EXTENT_TOLERANCE:.0%} allowed)",
    )


def _expect(row: dict, key: str) -> Any:
    return (row.get("expect") or {}).get(key)


def clipboard_coverage(base_row: dict, treat_row: dict) -> list[dict]:
    """Scores the clipboard, not the selection: unmounted rows cannot be selected but can be copied."""
    out = []
    base_clip = _expect(base_row, "clipboard_chars")
    treat_clip = _expect(treat_row, "clipboard_chars")
    for label, row, clip in (
        ("base", base_row, base_clip),
        ("treatment", treat_row, treat_clip),
    ):
        mounted, total = _expect(row, "messages_mounted"), _expect(row, "messages_total")
        if not _expect(row, "clipboard_readable"):
            # Not a pass: an unreadable clipboard must never look like a fine one.
            out.append(
                _check(
                    f"clipboard_readable:{label}",
                    None,
                    str(_expect(row, "clipboard_note") or "the clipboard could not be read back"),
                    required = True,
                )
            )
            continue
        fraction = _expect(row, "mounted_fraction")
        out.append(
            _check(
                f"clipboard_readable:{label}",
                clip is not None and clip > 0,
                f"{clip} characters reached the clipboard with {mounted} of {total} messages "
                f"mounted (mounted fraction {fraction})",
                required = True,
            )
        )
    # Scored against the thread, not the other arm: two serialisations cannot match in length. The
    # reference is `Selection.toString()` on a fully mounted arm; with none, the pair is not comparable.
    reference = _expect(base_row, "selected_chars")
    base_full = _expect(base_row, "mounted_fraction")
    if not isinstance(reference, (int, float)) or reference <= 0 or base_full != 1:
        out.append(
            _check(
                "clipboard_carries_the_whole_thread",
                None,
                "the base arm did not mount the whole thread, so there is no measurement of how "
                "long the conversation's visible text actually is to score either clipboard "
                f"against (base selection {reference}, mounted fraction {base_full})",
                required = True,
            )
        )
        return out
    for label, clip in (("base", base_clip), ("treatment", treat_clip)):
        coverage = None if not isinstance(clip, (int, float)) else clip / reference
        out.append(
            _check(
                f"clipboard_carries_the_whole_thread:{label}",
                None
                if coverage is None
                else (MIN_CLIPBOARD_COVERAGE <= coverage <= MAX_CLIPBOARD_COVERAGE),
                f"the clipboard carried {clip} characters against a thread whose visible text is "
                f"{reference} characters"
                + (
                    ""
                    if coverage is None
                    else f" ({coverage:.3f} of it, allowed "
                    f"{MIN_CLIPBOARD_COVERAGE}-{MAX_CLIPBOARD_COVERAGE})"
                ),
                required = True,
            )
        )
    # Reported, never gated: on a windowed arm the selection is supposed to be short.
    out.append(
        _check(
            "selection_shrank_as_expected",
            None,
            f"selected characters {_expect(base_row, 'selected_chars')} vs "
            f"{_expect(treat_row, 'selected_chars')} -- reported, not gated: a windowed mount "
            "cannot select what it has not mounted, and the clipboard above is what the user gets",
        )
    )
    return out


def _same_number(base_row: dict, treat_row: dict, key: str, name: str) -> dict:
    b, t = _expect(base_row, key), _expect(treat_row, key)
    drift = _drift(b, t)
    return _check(
        name,
        None if drift is None else drift <= EXACT_TOLERANCE,
        f"{key} {b} vs {t}" + ("" if drift is None else f" ({drift:.1%} drift)"),
    )


def _reopen_completed(row: dict) -> Optional[bool]:
    """Whether the reopened thread finished rebuilding; None when the row carries no evidence either way."""
    readiness = _expect(row, "reopen_readiness")
    if isinstance(readiness, dict) and isinstance(readiness.get("ready"), bool):
        return readiness["ready"]
    ok = row.get("expect_ok")
    return ok if isinstance(ok, bool) else None


def thread_survives_reopen(base_row: dict, treat_row: dict) -> list[dict]:
    """thread_reopen: the thread came back the same length, and by the same route."""
    out = []
    for label, row in (("base", base_row), ("treatment", treat_row)):
        before, after = _expect(row, "messages_before"), _expect(row, "messages_after")
        completed = _reopen_completed(row)
        detail = f"the thread had {before} messages and came back with {after}"
        # `messages_after` is `aria-setsize`, a declaration, so equal counts on a timed-out rebuild prove
        # nothing (NOT COMPARABLE). Disagreeing counts are BROKEN regardless: that is the data loss.
        if before is None or after is None:
            ok: Optional[bool] = None
            required = False
        elif before != after:
            ok, required = False, False
        elif completed:
            ok, required = True, False
        else:
            ok, required = None, True
            readiness = _expect(row, "reopen_readiness")
            failed = (
                sorted(k for k, v in (readiness.get("conditions") or {}).items() if v is False)
                if isinstance(readiness, dict)
                else []
            )
            detail += (
                ", but both numbers are the total the store DECLARED and the reopened thread never "
                "reached a ready state, so nothing here says the thread came back "
                f"(outstanding {failed or 'unrecorded'})"
            )
        out.append(_check(f"reopen_keeps_every_message:{label}", ok, detail, required = required))
        # A row measured after a full navigation is about a page load. See `_click_or_navigate`.
        via = _expect(row, "reopened_via")
        out.append(
            _check(
                f"reopen_used_the_control:{label}",
                None if via is None else via == "click",
                f"the thread was reopened via {via!r}",
            )
        )
    return out


def _extent_of(row: dict) -> tuple[Optional[float], bool]:
    """Reconstructs the extent from the census, since comparing bottoms amplifies drift by H/(H-C)."""
    bottom = _expect(row, "bottom")
    if not isinstance(bottom, (int, float)):
        return None, False
    client = (row.get("census") or {}).get("viewport_client_height")
    if not isinstance(client, (int, float)):
        return float(bottom), False
    return float(bottom) + float(client), True


def _client_height(row: dict) -> Optional[float]:
    client = (row.get("census") or {}).get("viewport_client_height")
    return float(client) if isinstance(client, (int, float)) else None


def _bottom_of(row: dict) -> Optional[float]:
    bottom = _expect(row, "bottom")
    return float(bottom) if isinstance(bottom, (int, float)) else None


def _comparable_extents(base_row: dict, treat_row: dict) -> tuple[Any, Any, str]:
    """Both arms reconstruct together or not at all, and only when client heights agree."""
    b_ext, b_full = _extent_of(base_row)
    t_ext, t_full = _extent_of(treat_row)
    b_bottom, t_bottom = _bottom_of(base_row), _bottom_of(treat_row)
    if not (b_full and t_full):
        return b_bottom, t_bottom, "bottom (no client height on both arms)"
    b_client, t_client = _client_height(base_row), _client_height(treat_row)
    if b_client != t_client:
        return (
            b_bottom,
            t_bottom,
            f"bottom (client heights differ: {b_client} vs {t_client})",
        )
    return b_ext, t_ext, "scroll extent"


def scroll_travelled(base_row: dict, treat_row: dict) -> list[dict]:
    """scroll_after: the gesture covers the ground it commanded and no more, on both arms."""
    out = []
    # Use the pair's larger extent like `_drift`: per-arm extents falsely failed a correction inside
    # tolerance, and a false red can VOID the plan. It only ever loosens.
    reference = max(
        (
            abs(extent)
            for extent, _ in (_extent_of(base_row), _extent_of(treat_row))
            if isinstance(extent, (int, float))
        ),
        default = None,
    )
    for label, row in (("base", base_row), ("treatment", treat_row)):
        fraction = _expect(row, "travel_fraction")
        commanded = _expect(row, "commanded_px")
        travelled = _expect(row, "travelled_px")
        # The allowance is a fraction of the extent, not of `bottom` (scrollHeight - clientHeight).
        extent, _reconstructed = _extent_of(row)
        # Bounded above too: overshoot is anchor instability. The ceiling is EXTENT_TOLERANCE of the arm's
        # own extent; with no extent only the lower bound applies.
        ceiling: Optional[float] = None
        if (
            isinstance(commanded, (int, float))
            and commanded > 0
            and isinstance(extent, (int, float))
        ):
            ceiling = (commanded + EXTENT_TOLERANCE * reference) / commanded
        if fraction is None:
            ok: Optional[bool] = None
            detail = "the row records no travel fraction"
        elif ceiling is None:
            ok = fraction >= 0.9
            detail = (
                f"the gesture travelled {fraction} of what it commanded; NO CEILING was applied "
                f"because the row carries no scrollable extent to derive one from"
            )
        else:
            ok = 0.9 <= fraction <= ceiling
            detail = (
                f"the gesture travelled {fraction} of what it commanded "
                f"({travelled} of {commanded} px, allowed 0.9 to {ceiling:.3f}: "
                f"{EXTENT_TOLERANCE:.0%} of the pair's {reference} px reference extent)"
            )
        out.append(_check(f"scroll_travelled:{label}", ok, detail))
    # Compare extents at EXTENT_TOLERANCE, not `bottom` at 2%; `_same_number` stays strict for
    # `selected_chars`, `visible_chars` and `clipboard_chars`.
    b_ext, t_ext, what = _comparable_extents(base_row, treat_row)
    drift = _drift(b_ext, t_ext)
    out.append(
        _check(
            "scroll_bottom_agrees",
            None if drift is None else drift <= EXTENT_TOLERANCE,
            f"{what} {b_ext} vs {t_ext}"
            + ("" if drift is None else f" ({drift:.1%} drift, {EXTENT_TOLERANCE:.0%} allowed)"),
        )
    )
    return out


# An action absent here is UNCHECKED, not passing.
INVARIANTS = {
    "select_all_copy": clipboard_coverage,
    "select_text": lambda b, t: [
        _same_number(b, t, "selected_chars", "selection_unchanged"),
        _same_number(b, t, "visible_chars", "visible_chars_unchanged"),
    ],
    "copy_markdown": lambda b, t: [
        _same_number(b, t, "clipboard_chars", "copy_unchanged"),
    ],
    "thread_reopen": thread_survives_reopen,
    "scroll_after": scroll_travelled,
}


def compare_behaviour(base_row: Optional[dict], treat_row: Optional[dict]) -> dict:
    """Uses the digest comparison's verdict vocabulary, so one report can carry both kinds of reading."""
    for label, row in (("base", base_row), ("treatment", treat_row)):
        if not isinstance(row, dict):
            return {
                "verdict": NOT_COMPARABLE,
                "reason": f"the {label} arm has no row for this action",
                "checks": [],
            }
        if not row.get("ran"):
            return {
                "verdict": NOT_EXERCISED,
                "reason": f"the action did not run on the {label} arm "
                f"({row.get('reason') or 'no reason recorded'})",
                "checks": [],
            }
    assert base_row is not None and treat_row is not None
    action = base_row.get("action") or treat_row.get("action") or ""
    checks: list[dict] = [scroll_extent(base_row, treat_row)]
    rule = INVARIANTS.get(action)
    if rule is None:
        checks.append(
            _check(
                "behavioural_invariant_declared",
                None,
                f"no behavioural invariant is declared for {action!r}, so this action is "
                "UNCHECKED on a windowed arm rather than passing",
            )
        )
    else:
        got = rule(base_row, treat_row)
        checks.extend(got if isinstance(got, list) else [got])

    # A required check that could not be read voids the pair.
    unread = [c for c in checks if c.get("required") and c["ok"] is None]
    if unread:
        return {
            "verdict": NOT_COMPARABLE,
            "reason": "; ".join(f"{c['invariant']}: {c['detail']}" for c in unread),
            "checks": checks,
        }

    broken = [c for c in checks if c["ok"] is False]
    if broken:
        return {
            "verdict": BROKEN,
            "reason": "; ".join(f"{c['invariant']}: {c['detail']}" for c in broken),
            "checks": checks,
        }
    # A pass needs an action-specific invariant: scroll_extent is a thread property shared by all actions.
    specific = [c for c in checks if c["ok"] is not None and c["invariant"] != "scroll_extent"]
    if not specific:
        return {
            "verdict": NOT_APPLICABLE,
            "reason": (
                f"no behavioural invariant specific to {action!r} could be read from this payload, "
                "so this surface is UNCHECKED on a windowed arm rather than passing"
            ),
            "checks": checks,
        }
    return {"verdict": MATCH, "reason": "", "checks": checks}
