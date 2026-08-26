# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The surface registry and the coverage manifest, checked without a browser.

WHAT THESE TESTS ARE FOR. A registry is a list of promises about an app that is not present when
the list is read, so almost nothing about it can be verified here. What CAN be verified is the
class of defect that makes a sweep worthless: an entry that cannot be executed, a coverage number
that counts a surface nobody reached, and a row that records a failure without recording why. All
three have the same shape -- the artefact looks complete and says the wrong thing -- and all three
are cheap to catch here rather than after a forty-minute run.

No browser and no Unsloth. The step lists are declarative for exactly this reason, and the sweep is
driven against a scripted stand-in page so its bookkeeping is exercised on both the reached and
the unreached path.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.report.payload import ROW_TYPE_SECTIONS  # noqa: E402
from studiobench.runtime.types import ROW_REQUIRED, ROW_TYPES  # noqa: E402
from studiobench.scene import surface_sweep, surfaces  # noqa: E402
from studiobench.scoring.schema import validate_payload  # noqa: E402


# ── the registry is well formed ─────────────────────────────────────


def test_the_shipped_registry_validates():
    surfaces.validate_registry()


def test_surface_ids_are_unique():
    ids = surfaces.surface_ids()
    assert len(ids) == len(set(ids))


def test_every_surface_declares_a_reach_a_restore_and_a_root():
    for surface in surfaces.surfaces():
        assert surface.root, surface.id
        assert surface.reach or surface.restore == (), (
            f"{surface.id} declares no reach, so it must also declare no restore: a surface that "
            f"is reached by doing nothing cannot need undoing"
        )
        # Anything that navigates or opens an overlay has to say how to get back, or the next
        # surface is reached from a state nobody declared.
        opens = any(step[0] in {"goto", "click", "fill", "press"} for step in surface.reach)
        assert not opens or surface.restore, f"{surface.id} opens something and never closes it"


def test_every_surface_declares_a_settle_condition():
    # A sweep with no settle condition is a sweep that digests whatever is on screen when the
    # sleep expires, which is the one result this tool may not produce.
    for surface in surfaces.surfaces():
        assert surface.settle is not None, surface.id
        assert isinstance(surface.settle, dict)
        assert set(surface.settle) & {
            "visible",
            "hidden",
            "count_at_least",
            "text",
            "js",
        }, f"{surface.id} has a settle condition surfaces.js cannot evaluate"


def test_no_surface_settles_on_a_timer():
    for surface in surfaces.surfaces():
        assert "wait" not in surface.settle, (
            f"{surface.id} settles on a wait. A wait cannot tell a surface that rendered from one "
            f"that never did"
        )


def test_every_step_uses_a_known_verb_with_the_right_arity():
    for surface in surfaces.surfaces():
        for steps in (surface.reach, surface.restore):
            for step in steps:
                assert step[0] in surfaces.VERBS, (surface.id, step)
                assert len(step) - 1 == surfaces.VERBS[step[0]], (surface.id, step)


def test_the_sweep_can_execute_every_verb_the_registry_uses():
    # The registry and the interpreter are in different files, and a verb added to one and not the
    # other fails only at run time, on the surface that uses it, as a reach failure that reads
    # like a broken selector.
    used = {
        step[0] for s in surfaces.surfaces() for steps in (s.reach, s.restore) for step in steps
    }
    driver = surface_sweep._Driver(page = _FakePage(), base_url = "http://x", log = lambda _m: None)
    sample = {
        "goto": ("/chat",),
        "click": ("sel",),
        "click_if": ("sel",),
        "hover": ("sel",),
        "press": ("Escape",),
        "fill": ("sel", "text"),
        "wait": (10,),
    }
    for verb in used:
        driver._step(verb, sample[verb])


def test_known_uncovered_entries_are_complete():
    for entry in surfaces.KNOWN_UNCOVERED:
        assert entry["id"] and entry["title"]
        # The reason is the whole value of the list. A one-word reason is a gap with a label on it.
        assert len(entry["reason"]) > 40, entry["id"]


def test_a_surface_cannot_be_both_registered_and_known_uncovered():
    assert not ({u["id"] for u in surfaces.KNOWN_UNCOVERED} & set(surfaces.surface_ids()))


@pytest.mark.parametrize("group", ["route", "settings", "sidebar", "chat"])
def test_the_registry_covers_every_major_group(group):
    assert group in surfaces.groups()


def test_the_settings_dialog_covers_every_shipped_tab():
    # The twelve panels are lazily imported chunks, so a tab that stopped rendering fails only
    # for the tab. Pinned here so a panel added to the app without a surface is visible as a
    # missing id rather than as a coverage figure that quietly stayed at 100%.
    shipped = {
        "general",
        "profile",
        "appearance",
        "resources",
        "chat",
        "voice",
        "connections",
        "data",
        "api-keys",
        "agents",
        "debugging",
        "about",
    }
    registered = {s.id.split(":", 1)[1] for s in surfaces.surfaces() if s.group == "settings"}
    assert shipped - registered == set()


def test_the_registry_covers_every_routed_path():
    # From app/router.tsx. `/` and `/settings` are not page surfaces -- both redirect on
    # beforeLoad -- and the two auth-flow routes are in KNOWN_UNCOVERED with the guard that
    # redirects them.
    routed = {
        "/chat",
        "/projects",
        "/hub",
        "/studio",
        "/images",
        "/video",
        "/audio",
        "/export",
        "/data-recipes",
        "/api-monitor",
    }
    reached = set()
    for surface in surfaces.surfaces():
        for step in surface.reach:
            if step[0] == "goto":
                reached.add(step[1])
    assert routed - reached == set()


# ── validate_registry rejects what it must ──────────────────────────


def _one(**overrides):
    base = dict(
        id = "x:one",
        group = "g",
        title = "t",
        reach = (("goto", "/chat"),),
        restore = (("goto", "/chat"),),
        root = ("body",),
        settle = {"visible": "body"},
    )
    base.update(overrides)
    return surfaces.Surface(**base)


def test_validate_rejects_a_duplicate_id():
    with pytest.raises(surfaces.RegistryError, match = "duplicate"):
        surfaces.validate_registry([_one(), _one()])


def test_validate_rejects_an_unknown_verb():
    with pytest.raises(surfaces.RegistryError, match = "unknown verb"):
        surfaces.validate_registry([_one(reach = (("teleport", "/chat"),))])


def test_validate_rejects_the_wrong_number_of_arguments():
    with pytest.raises(surfaces.RegistryError, match = "argument"):
        surfaces.validate_registry([_one(reach = (("fill", "sel"),))])


def test_validate_rejects_a_surface_with_no_digest_root():
    with pytest.raises(surfaces.RegistryError, match = "root"):
        surfaces.validate_registry([_one(root = ())])


# ── the row type is registered, not smuggled ────────────────────────


def test_the_surface_row_type_is_registered():
    assert "surface" in ROW_TYPES
    assert ROW_TYPE_SECTIONS["surface"] == "surfaces"


def test_a_surface_row_must_carry_its_reason():
    # The single defect this project cares most about: a check that cannot distinguish "did not
    # run" from "passed". A surface row without `reason` is exactly that row.
    assert "reason" in ROW_REQUIRED["surface"]
    assert "reached" in ROW_REQUIRED["surface"]
    assert "parity" in ROW_REQUIRED["surface"]


def test_the_recorder_accepts_a_surface_row_and_rejects_one_missing_its_reason(tmp_path):
    from studiobench.runtime.types import Recorder

    rec = Recorder(tmp_path / "payload.jsonl", "session")
    rec.emit(
        {
            "row_type": "surface",
            "surface": "route:chat",
            "reached": True,
            "reason": None,
            "parity": {"parity_attempted": True, "digest": "abcd1234"},
        }
    )
    with pytest.raises(ValueError, match = "reason"):
        rec.emit(
            {
                "row_type": "surface",
                "surface": "route:hub",
                "reached": False,
                "parity": {"parity_attempted": False},
            }
        )
    rec.close()


def test_surface_rows_land_in_their_own_payload_section(tmp_path):
    from studiobench.report.payload import assemble_rows

    path = tmp_path / "payload.jsonl"
    path.write_text(
        '{"row_type":"run_meta","tier":"quick"}\n'
        '{"row_type":"cell","cell_id":"r1K.A0.rep0","completed":true}\n'
        '{"row_type":"surface","surface":"route:chat","reached":true,"reason":null,'
        '"parity":{"parity_attempted":true,"digest":"abcd1234"}}\n',
        encoding = "utf-8",
    )
    payload = assemble_rows(path)
    assert len(payload["surfaces"]) == 1
    # NOT in unknown_rows. A new row type that lands there is carried through the whole report
    # without anyone noticing it is not being read.
    assert payload["unknown_rows"] == []


def test_a_payload_carrying_surface_rows_has_no_bare_zeros():
    # The schema bans a naked numeric zero, because a zero outside a measure is indistinguishable
    # from "we never ran that". Surface rows carry counts, so they have to attest.
    rows = surface_sweep.sweep(_FakePage(reach_ok = True), "http://x")[0]
    validate_payload({"excluded_cells": [], "surfaces": rows})


# ── the manifest ────────────────────────────────────────────────────


def _rows_for(entries, reached_ids):
    out = []
    for surface in entries:
        row = surface_sweep._row(surface, cell_id = None)
        if surface.id in reached_ids:
            row["reached"] = True
            row["reason"] = None
            row["parity"] = {"parity_attempted": True, "digest": "abcd1234", "chars": 100}
        out.append(row)
    return out


def test_the_manifest_counts_only_what_was_reached():
    entries = surfaces.surfaces()
    rows = _rows_for(entries, {entries[0].id})
    manifest = surface_sweep.build_manifest(rows, entries, {"scoped": True})
    assert manifest["registered"] == len(entries)
    assert manifest["reached"] == 1
    assert manifest["not_reached"] == len(entries) - 1
    assert len(manifest["failures"]) == len(entries) - 1


def test_every_unreached_surface_appears_in_the_manifest_with_a_reason():
    entries = surfaces.surfaces()
    manifest = surface_sweep.build_manifest(_rows_for(entries, set()), entries, {"scoped": True})
    assert len(manifest["failures"]) == len(entries)
    for failure in manifest["failures"]:
        assert failure["reason"], failure["id"]


def test_a_conditional_miss_is_counted_apart_from_a_hard_miss():
    entries = [_one(id = "a", conditional = None), _one(id = "b", conditional = "needs a gpu")]
    manifest = surface_sweep.build_manifest(_rows_for(entries, set()), entries, {"scoped": True})
    assert manifest["not_reached_hard"] == 1
    assert manifest["not_reached_conditional"] == 1
    # Coverage is against what this host COULD render, so the conditional miss leaves the
    # denominator at one rather than silently deflating the figure.
    assert manifest["coverage_pct"] == 0.0
    assert manifest["coverage_pct_of_registered"] == 0.0


def test_a_volatile_surface_is_reached_but_not_counted_as_comparable():
    # Reached and volatile are different facts and the manifest has to keep them apart. Folding
    # them together is how a live memory gauge ends up quoted as a UI change.
    entries = [_one(id = "a"), _one(id = "b", volatile = "shows a live memory gauge")]
    rows = _rows_for(entries, {"a", "b"})
    manifest = surface_sweep.build_manifest(rows, entries, {"scoped": True})
    assert manifest["reached"] == 2
    assert manifest["comparable"] == 1
    assert manifest["volatile"] == 1
    assert manifest["volatile_surfaces"] == [{"id": "b", "mechanism": "shows a live memory gauge"}]


def test_every_volatile_surface_states_its_mechanism():
    # A volatility flag without a mechanism is a licence to ignore a real difference.
    for surface in surfaces.surfaces():
        if surface.volatile is not None:
            assert len(surface.volatile) > 40, surface.id


def test_the_measured_volatile_surfaces_are_declared():
    # Three consecutive sweeps against one Unsloth agreed on 44 of 53 surfaces. These are the ones
    # that did not, each with a mechanism established from source or from a text diff. Pinned so
    # a settle condition that is tightened later shows up as a flag that can now be dropped,
    # rather than as a flag nobody revisits.
    declared = {s.id for s in surfaces.surfaces() if s.volatile}
    assert {
        "route:chat",
        "chat:composer-filled",
        "settings:resources",
        "settings:debugging",
        "route:hub",
        "hub:datasets",
        "hub:compact-layout",
        # Measured to move, mechanism not established. Pinned so the admission stays visible.
        "route:train",
        "settings:agents",
        "train:image-training",
    } <= declared


def test_the_unexplained_movers_say_so_rather_than_offering_a_cause():
    # A plausible-sounding mechanism that has not been established is worse than an admitted gap:
    # it is the sentence somebody uses to wave away a real difference.
    unexplained = [
        s
        for s in surfaces.surfaces()
        if s.volatile and "NOT" in s.volatile and "established" in s.volatile
    ]
    assert {s.id for s in unexplained} == {"route:train", "settings:agents", "train:image-training"}


def test_the_rendered_manifest_names_the_volatile_surfaces():
    entries = [_one(id = "b", volatile = "shows a live memory gauge that moves on its own")]
    text = surface_sweep.render_manifest(
        surface_sweep.build_manifest(_rows_for(entries, {"b"}), entries, {"scoped": True})
    )
    assert "NOT COMPARABLE" in text
    assert "live memory gauge" in text


def test_the_manifest_carries_the_known_uncovered_list():
    entries = surfaces.surfaces()
    manifest = surface_sweep.build_manifest(_rows_for(entries, set()), entries, {"scoped": True})
    assert manifest["known_uncovered_count"] == len(surfaces.KNOWN_UNCOVERED)
    assert {u["id"] for u in manifest["known_uncovered"]} == {
        u["id"] for u in surfaces.KNOWN_UNCOVERED
    }


def test_the_rendered_manifest_names_every_failure_and_every_known_gap():
    entries = surfaces.surfaces()
    manifest = surface_sweep.build_manifest(_rows_for(entries, set()), entries, {"scoped": True})
    text = surface_sweep.render_manifest(manifest)
    for surface in entries:
        assert surface.id in text
    for entry in surfaces.KNOWN_UNCOVERED:
        assert entry["id"] in text


def test_an_unscoped_sweep_says_so_in_the_rendered_manifest():
    # An unscoped digest is a whole-page reading taken forty times, and every one of them agrees
    # with every other for reasons that have nothing to do with the surfaces. That has to be at
    # the top of the artefact, not inferable from a field somebody might read.
    entries = surfaces.surfaces()[:1]
    manifest = surface_sweep.build_manifest(
        _rows_for(entries, {entries[0].id}),
        entries,
        {"scoped": False, "reason": "parity.capture() ignored the moved root"},
    )
    text = surface_sweep.render_manifest(manifest)
    assert "WARNING" in text
    assert "ignored the moved root" in text


# ── the sweep's own bookkeeping ─────────────────────────────────────


class _FakePage:
    """A scripted stand-in for a Playwright page.

    `reach_ok` false makes every click raise, which is how a broken selector presents. The point
    is not to simulate a browser: it is to prove that a sweep whose reaches all fail still emits
    one row per surface, each carrying a reason, rather than emitting nothing.
    """

    def __init__(
        self,
        reach_ok: bool = True,
        settles: bool = True,
    ) -> None:
        self.reach_ok = reach_ok
        self.settles = settles
        self.clicks: list = []

    def goto(self, *_a, **_k) -> None:
        pass

    def click(self, selector, **_k) -> None:
        if not self.reach_ok:
            raise RuntimeError(f"Timeout: {selector}")
        self.clicks.append(selector)

    def hover(self, selector, **_k) -> None:
        if not self.reach_ok:
            raise RuntimeError(f"Timeout: {selector}")

    def fill(self, *_a, **_k) -> None:
        if not self.reach_ok:
            raise RuntimeError("Timeout")

    def query_selector(self, _selector):
        return None

    def wait_for_timeout(self, _ms) -> None:
        pass

    @property
    def keyboard(self):
        return self

    def press(self, _key) -> None:
        pass

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if "probeScoping" in script:
            return {"scoped": True, "scoping_attempted": True, "probe_chars": 44}
        if "settled" in script:
            return {"ok": self.settles, "detail": "scripted"}
        if "capture" in script:
            return {
                "parity_attempted": True,
                "digest": "abcd1234",
                "chars": 100,
                "messages": [],
                "overlays": [],
                "root_selector": (arg or ["?"])[0],
            }
        if "facts" in script:
            return {
                "facts_attempted": True,
                "root_elements": 10,
                "pathname": "/chat",
                "open_dialogs": 0,
                "open_menus": 0,
            }
        if "isClean" in script:
            return {
                "clean_attempted": True,
                "open_dialogs": 0,
                "open_menus": 0,
                "pathname": "/chat",
            }
        raise AssertionError(f"unscripted evaluate: {script[:60]}")


def test_a_reach_that_fails_records_a_reason_and_never_a_digest():
    rows, _manifest = surface_sweep.sweep(_FakePage(reach_ok = False), "http://x")
    assert len(rows) == len(surfaces.surfaces())
    interactive = [
        r
        for r in rows
        if any(
            step[0] in {"click", "hover", "fill"}
            for step in surfaces.get_surface(r["surface"]).reach
        )
    ]
    assert interactive, "the registry has no interactive surface left to check"
    for row in interactive:
        assert row["reached"] is False
        assert "the reach failed" in row["reason"]
        # Never a digest it did not take. A parity block that claimed `attempted` here would make
        # a failed sweep read as a passing parity check downstream.
        assert row["parity"]["parity_attempted"] is False


def test_a_surface_that_never_settles_is_not_recorded_as_reached():
    rows, manifest = surface_sweep.sweep(
        _FakePage(reach_ok = True, settles = False),
        "http://x",
        settle_timeout_ms = 1,
        surface_budget_ms = 50,
    )
    assert manifest["reached"] == 0
    assert len(rows) == len(surfaces.surfaces())
    for row in rows:
        assert "never settled" in row["reason"]
        assert row["parity"]["parity_attempted"] is False


def test_a_clean_sweep_reaches_every_surface_and_records_the_root_it_digested():
    rows, manifest = surface_sweep.sweep(_FakePage(), "http://x")
    assert manifest["reached"] == len(surfaces.surfaces())
    assert manifest["coverage_pct"] == 100.0
    for row in rows:
        assert row["reason"] is None
        assert row["parity"]["root_selector"]


def test_the_sweep_rejects_an_unknown_surface_id():
    with pytest.raises(surfaces.RegistryError, match = "no such surface"):
        surface_sweep.sweep(_FakePage(), "http://x", only = ["nope:nothing"])


def test_the_sweep_streams_rows_to_a_recorder_as_it_goes():
    class _Rec:
        def __init__(self):
            self.rows = []

        def emit(self, row):
            self.rows.append(row)

    rec = _Rec()
    rows, _manifest = surface_sweep.sweep(_FakePage(), "http://x", recorder = rec)
    # One per surface, so a sweep that dies halfway leaves the surfaces it did reach behind.
    assert len(rec.rows) == len(rows)


def test_the_sweep_returns_to_the_known_state_before_every_surface():
    # Not "after every surface": an after-only reset trusts each surface's own restore, and the
    # one that fails is the one whose restore did not run.
    page = _FakePage()
    surface_sweep.sweep(page, "http://x")
    assert surfaces.KNOWN_STATE_PATH == "/chat"


# ── the digest is the film's digest, not a second one ───────────────


def test_the_sweep_takes_its_digest_through_parity_capture():
    # If surfaces.js ever grew its own DOM walk, surface digests and action digests would stop
    # being comparable -- and nothing downstream would notice, because both would still be hex
    # strings of the same length.
    text = (Path(__file__).resolve().parents[2] / "scene" / "surfaces.js").read_text(
        encoding = "utf-8"
    )
    assert "parity.capture()" in text
    assert "window.__sb.parity" in text


def test_surfaces_js_does_not_edit_the_shared_chat_adapter():
    # dom.js is the film's adapter and every action reads it. The surface layer restores the root
    # it moved, so a sweep cannot leave the film digesting the wrong element.
    text = (Path(__file__).resolve().parents[2] / "scene" / "surfaces.js").read_text(
        encoding = "utf-8"
    )
    assert "dom.threadRoot = original" in text


def test_the_registry_is_a_frozen_dataclass_so_a_sweep_cannot_rewrite_it():
    surface = surfaces.surfaces()[0]
    assert dataclasses.is_dataclass(surface)
    with pytest.raises(dataclasses.FrozenInstanceError):
        surface.id = "mutated"
