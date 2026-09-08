# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The seven runtime-injected knobs, as a decision table: which knob removes the slope names the fix.

Every arm here runs on the SHIPPED PRODUCTION BUILD through `add_init_script`. Nothing is
compiled, nothing is patched on disk, and an external tester with a laptop and an Unsloth install
can run the whole ablation plane. That constraint is not a convenience. An ablation that requires
a custom build is an ablation that will be run once, by the person who wrote it, on the machine
where the problem does not reproduce.

THE DECISION TABLE. These are not seven ways of saying "it got faster". Each knob removes a
different layer of the pipeline, and the FIRST one that removes the slope names the layer the
cost lives in:

  A  visibility:hidden on completed messages
     removes paint and raster, keeps layout, DOM and React
     -> if the slope goes: the cost is painting retained messages
  B  content-visibility:auto + contain-intrinsic-size (undoing the shipped
     `.aui-thread-root [data-streamdown="code-block"]` override in index.css, which forces
     `content-visibility: visible !important`)
     removes off-screen style, layout and paint
     -> if the slope goes: the cost is off-screen style and layout. Note what the shipped comment
        above that rule now says: the "thread length is bounded" justification has already been
        disproved by #8977, the rule is kept for a height flicker and a WebKit find-in-page
        hazard, and containment on the message roots was measured as no help, with the claim that
        what grows with thread length is inherited-property style recalc rather than layout. That
        last sentence is arm E's hypothesis, already asserted upstream. This ladder is what
        confirms or refutes it, and a positive B is not permission to delete the rule
  C  display:none on completed messages
     also removes layout geometry and the sibling from the layout sequence
     -> if the slope goes here but not at A or B: the cost is layout geometry, and the fix is to
        virtualise the list
  D  detach the autoscroll subtree observer
     removes forced synchronous layout per mutation
     -> if the slope goes: the cost is the observer reading scrollHeight on every streamed
        character, at a price proportional to the whole thread
  E  neutralise --aui-scroll-stabilizer
     removes inherited-custom-property subtree style invalidation
     -> if the slope goes: writing one inherited custom property on the scroll container is
        invalidating style for every descendant, per mutation
  F  freeze React but keep the DOM
     removes React subscriptions and reconciliation
     -> if the slope goes: the cost is fibre bookkeeping, not DOM. Note this arm is DOM-CHANGING
        (the stream stops rendering while frozen) so it is only ever an upper bound
  G  CONTROL: identical DOM, thread scrolled so prior turns are IN the viewport
     -> if the slope is the SAME as the unscrolled case, off-screen occupancy is not the
        mechanism, and A, B and C should all have read null. If G disagrees with them, one of
        them did not fire

G is the arm that catches the harness rather than the app, which is why it is in the list rather
than in a comment. A, B and C all rest on the assumption that the retained messages are OFF
screen; if the thread is not actually scrolled where the harness thinks it is, all three read
null for a reason that has nothing to do with rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .manifest import Arm, DeclaredDiff, Invariance, PotencyCounter

KNOBS_JS_PATH = Path(__file__).with_name("knobs.js")

#: Knobs that must be installed BEFORE the app boots, because they patch a prototype the app is
#: about to use. Installing one of these after boot measures nothing: React has already captured
#: its scheduler port and the autoscroll hook has already called observe().
PREBOOT_ARM_IDS: frozenset[str] = frozenset({"D", "E", "F"})

#: Knobs applied at the start of a measured window, over the messages that exist then.
RUNTIME_ARM_IDS: frozenset[str] = frozenset({"A", "B", "C", "G"})


ARM_A = Arm(
    arm_id = "A",
    title = "visibility:hidden on completed messages",
    mechanism = "paint and raster of retained messages",
    invariance = Invariance.EXACT,
    potency = PotencyCounter(
        name = "visibilityHiddenConfirmed",
        min_delta = 1,
        direction = "increase",
        description = (
            "message roots whose COMPUTED visibility is hidden. Computed, not the rule we "
            "injected: a rule that lost the cascade is a knob that did not fire, and counting our "
            "own stylesheet would report it as fired"
        ),
    ),
    implies_fix = (
        "the cost is painting and rasterising retained messages. Fix by not painting what is off "
        "screen: containment, or a virtualised list"
    ),
    notes = (
        "the DOM is untouched (one injected stylesheet, no element attributes change), so this "
        "arm claims EXACT and its cost is a point estimate of paint"
    ),
)

ARM_B = Arm(
    arm_id = "B",
    title = "content-visibility:auto with contain-intrinsic-size",
    mechanism = "off-screen style, layout and paint",
    invariance = Invariance.EQUIVALENT,
    declared_diff = DeclaredDiff(
        normaliser = "skip_style_attribute",
        keys = ("attr:style",),
        rationale = (
            "this arm writes content-visibility and contain-intrinsic-size INLINE on each "
            "completed message root, because the intrinsic size has to be that element's own "
            "measured height. The style attribute is therefore expected to differ and nothing "
            "else is. If any other attribute or any text differs, the arm is voided. The key is "
            "`attr:style` and not `style` because knobs.js namespaces attribute keys against the "
            "structural ones, so an attribute literally named `text` cannot collide with the "
            "text key. Getting this string wrong voids the arm on every run, which is the "
            "correct failure direction but an expensive one to debug"
        ),
    ),
    potency = PotencyCounter(
        name = "contentVisibilityAutoConfirmed",
        min_delta = 1,
        direction = "increase",
        description = (
            "elements whose computed content-visibility is auto. index.css:2537 sets `visible` "
            "with !important, so this counter is also the check that the undo actually won"
        ),
    ),
    implies_fix = (
        "the cost is style and layout of off-screen content. This does NOT license removing the "
        "shipped override. Read the comment above the rule in index.css before acting on a "
        "positive result here: the original justification (thread length is bounded) has already "
        "been disproved by #8977, and the rule is now kept for a visible height flicker during "
        "stream finalisation plus a WebKit find-in-page hazard (webkit.org/b/283846, no in-thread "
        "search to fall back on). A win from this arm means the cost is real and needs a "
        "DIFFERENT mechanism to remove it"
    ),
    notes = (
        "two things this arm cannot see, both stated here because they cannot be stated in the "
        "digest. First, the canonical serialisation walks message roots and records descendant "
        "code-block and span COUNTS, not descendant attributes, so the inline style this arm "
        "writes on code blocks is invisible to the invariance check and only the message-root "
        "style is actually policed. Second, the shipped override lives inside `@layer utilities`, "
        "and an important declaration in a layer beats an important unlayered one at any "
        "specificity, so the undo has to be emitted into the same layer (or inline) and then "
        "VERIFIED with getComputedStyle. The potency counter is that verification"
    ),
)

ARM_C = Arm(
    arm_id = "C",
    title = "display:none on completed messages",
    mechanism = "layout geometry and the sibling's place in the layout sequence",
    invariance = Invariance.EXACT,
    potency = PotencyCounter(
        name = "displayNoneConfirmed",
        min_delta = 1,
        direction = "increase",
        description = "message roots whose computed display is none",
    ),
    implies_fix = (
        "the cost is layout geometry of retained messages. Fix by virtualising the message list; "
        "thread.tsx:1741 renders a bare ThreadPrimitive.Messages and nothing in the thread is "
        "virtualised, although the app already depends on @tanstack/react-virtual"
    ),
    notes = (
        "C subsumes A and B for completed messages, which is why the ladder places it after both "
        "rather than treating the three as independent"
    ),
)

ARM_D = Arm(
    arm_id = "D",
    title = "detach the autoscroll subtree observer",
    mechanism = "forced synchronous layout inside the autoscroll MutationObserver",
    invariance = Invariance.EXACT,
    potency = PotencyCounter(
        name = "suppressedViewportObserves",
        min_delta = 1,
        direction = "increase",
        description = (
            "observe() calls no-opped. Matched on the target carrying aui-stream-viewport OR an "
            "attributeFilter containing aria-expanded, which is unique to this observer among the "
            "nine in the app"
        ),
    ),
    implies_fix = (
        "the cost is the autoscroll observer: it reads scrollHeight, writes an inherited custom "
        "property and calls scrollTo on every streamed character, at a price proportional to the "
        "whole thread. Fix at use-intent-aware-autoscroll.tsx:435,466,487"
    ),
    notes = (
        "pre-boot: the hook calls observe() during mount, so a patch installed after boot "
        "suppresses nothing and reads as NOT RUN, correctly"
    ),
)

ARM_E = Arm(
    arm_id = "E",
    title = "neutralise --aui-scroll-stabilizer",
    mechanism = "subtree style invalidation from writing an inherited custom property",
    invariance = Invariance.EXACT,
    potency = PotencyCounter(
        name = "suppressedStabilizerSets",
        min_delta = 1,
        direction = "increase",
        description = "setProperty calls for --aui-scroll-stabilizer that were skipped",
    ),
    implies_fix = (
        "the cost is style invalidation of every descendant caused by writing one inherited "
        "custom property on the scroll container per mutation. Fix at "
        "use-intent-aware-autoscroll.tsx:466, or stop the property being inherited"
    ),
    notes = (
        "E is a strict subset of D: D removes the callback that does this write, among other "
        "things. That is why the ladder puts E after D and never quotes the two together"
    ),
)

ARM_F = Arm(
    arm_id = "F",
    title = "freeze React, keep the DOM",
    mechanism = "React subscriptions and reconciliation",
    invariance = Invariance.DOM_CHANGING,
    potency = PotencyCounter(
        name = "suppressedSchedulerCallbacks",
        min_delta = 1,
        direction = "increase",
        description = (
            "scheduler callbacks not delivered. If React never used MessageChannel on this "
            "engine, capturedSchedulerPorts is 0 and the arm reads NOT RUN rather than no effect"
        ),
    ),
    implies_fix = (
        "the cost is React fibre bookkeeping rather than DOM. Fix by cutting the number of fibres "
        "reached per update, which memo does not do: bailoutOnAlreadyFinishedWork returns null "
        "only when childLanes is clear, so an update anywhere in the subtree still clones one "
        "work-in-progress fibre per sibling"
    ),
    notes = (
        "DOM-CHANGING by construction: while React is frozen the stream stops rendering, so the "
        "two sides are not showing the same content and this arm can only ever bound the "
        "mechanism from above. That is still worth having, because a small bound is a strong "
        "negative result"
    ),
)

ARM_G = Arm(
    arm_id = "G",
    title = "CONTROL: identical DOM, prior turns scrolled INTO the viewport",
    mechanism = "none; this arm changes only the scroll position",
    invariance = Invariance.EXACT,
    potency = PotencyCounter(
        name = "controlVisibleMessages",
        min_delta = 1,
        direction = "increase",
        description = "message roots intersecting the viewport rect after the scroll",
    ),
    implies_fix = (
        "not a fix, a check on the other arms. If bringing the retained turns ON screen does not "
        "change the cost, then off-screen occupancy was never the mechanism and A, B and C should "
        "all read null. If G disagrees with them, one of them did not fire"
    ),
    kind = "control",
)

RUNTIME_ARMS: tuple[Arm, ...] = (ARM_A, ARM_B, ARM_C, ARM_D, ARM_E, ARM_F, ARM_G)
ARM_BY_ID: Mapping[str, Arm] = {arm.arm_id: arm for arm in RUNTIME_ARMS}


def load_knobs_js() -> str:
    """Read the injected knob implementation from disk."""

    return KNOBS_JS_PATH.read_text(encoding = "utf-8")


def config_init_script(
    arm_ids: Iterable[str],
    *,
    debug: bool = False,
    control_visible_target: int = 3,
) -> str:
    """The config script that must be injected BEFORE knobs.js.

    Only the pre-boot arms named here get their prototype patches installed. An installed but
    inactive patch is not free: it adds a call frame to `observe`, to `setProperty` and to every
    scheduler delivery, and it would sit in the control cell too, which is precisely the kind of
    quiet, treatment-correlated overhead this whole design exists to keep out.
    """

    requested = sorted(set(arm_ids))
    unknown = [arm_id for arm_id in requested if arm_id not in ARM_BY_ID]
    if unknown:
        raise KeyError(f"unknown arm ids {unknown}; known ids are {sorted(ARM_BY_ID)}")
    preboot = [arm_id for arm_id in requested if arm_id in PREBOOT_ARM_IDS]
    config = {
        "preboot": preboot,
        "requested": requested,
        "debug": bool(debug),
        "controlVisibleTarget": int(control_visible_target),
    }
    return f"window.__sbArmConfig = {json.dumps(config, sort_keys = True)};"


def init_scripts_for(
    arm_ids: Iterable[str],
    *,
    debug: bool = False,
    control_visible_target: int = 3,
    extra_scripts: Sequence[str] = (),
) -> list[str]:
    """The ordered list of init scripts for one cell. Order is load-bearing.

    The config must exist before knobs.js reads it, and knobs.js must be installed before the app
    bundle runs. `add_init_script` preserves insertion order, so this list is handed over as-is.
    """

    return [
        config_init_script(arm_ids, debug = debug, control_visible_target = control_visible_target),
        load_knobs_js(),
        *extra_scripts,
    ]


def split_arms(arm_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split a rung's arms into (pre-boot, apply-at-window-start)."""

    requested = sorted(set(arm_ids))
    return (
        [a for a in requested if a in PREBOOT_ARM_IDS],
        [a for a in requested if a in RUNTIME_ARM_IDS],
    )


@dataclass(frozen = True)
class DecisionRow:
    arm_id: str
    removes: str
    if_slope_goes: str


def decision_table() -> list[DecisionRow]:
    """The table the report prints. One row per knob, read as a diagnosis."""

    return [DecisionRow(arm.arm_id, arm.mechanism, arm.implies_fix) for arm in RUNTIME_ARMS]


def render_decision_table() -> str:
    lines = [
        "ABLATION DECISION TABLE (which knob removes the slope names the fix)",
        "",
    ]
    for row in decision_table():
        lines.append(f"  {row.arm_id}  removes: {row.removes}")
        lines.append(f"     if the slope goes here: {row.if_slope_goes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def arms_json() -> list[dict[str, Any]]:
    return [arm.to_json() for arm in RUNTIME_ARMS]
