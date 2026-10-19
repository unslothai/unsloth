// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  ORB_BALL_EXTENT,
  ORB_CHIP_HEIGHT,
  ORB_RING_EXTENT,
  ORB_STACK_GAP,
  ORB_STATUS_HEIGHT,
  orbStackLayout,
} from "../src/features/chat/voice/orb-stack-layout.ts";
import { pendingApproval } from "../src/features/chat/voice/orb-widget.ts";

const orb = readFileSync(
  new URL("../src/components/assistant-ui/voice-orb.tsx", import.meta.url),
  "utf-8",
);

test("every combination of arcs and chip keeps the same gap down the whole stack", () => {
  for (const arcs of [true, false]) {
    for (const chip of [true, false]) {
      const l = orbStackLayout({ arcs, chip, card: true });
      // Ball (or its ring) to status.
      assert.equal(
        l.status,
        (arcs ? ORB_RING_EXTENT : ORB_BALL_EXTENT) + ORB_STACK_GAP,
      );
      if (chip) {
        assert.equal(l.chip, l.status + ORB_STATUS_HEIGHT + ORB_STACK_GAP);
        assert.equal(
          l.card,
          (l.chip as number) + ORB_CHIP_HEIGHT + ORB_STACK_GAP,
        );
      } else {
        // The card takes the chip's slot.
        assert.equal(l.chip, null);
        assert.equal(l.card, l.status + ORB_STATUS_HEIGHT + ORB_STACK_GAP);
      }
    }
  }
});

test("arcs off pulls the stack up under the ball; the chip hidden pulls the card up", () => {
  const all = orbStackLayout({ arcs: true, chip: true, card: true });
  const noArcs = orbStackLayout({ arcs: false, chip: true, card: true });
  const noChip = orbStackLayout({ arcs: true, chip: false, card: true });
  assert.ok(
    noArcs.status < all.status &&
      (noArcs.card as number) < (all.card as number),
  );
  assert.equal(noChip.status, all.status);
  assert.equal(noChip.card, all.chip);
  assert.equal(
    orbStackLayout({ arcs: true, chip: true, card: false }).card,
    null,
  );
});

test("the stack glides to its place rather than jumping, and the ball's room never changes", () => {
  assert.match(orb, /transition: "transform 150ms ease-out"/);
  assert.match(
    orb,
    /height: ORB_STATUS_HEIGHT \+ ORB_STACK_GAP \+ ORB_CHIP_HEIGHT/,
  );
});

test("the approval card shows while a call waits and goes once it is answered", () => {
  assert.deepEqual(pendingApproval({ a: {}, b: {} }), { id: "a", more: 1 });
  assert.equal(pendingApproval({}), null);
  assert.match(
    orb,
    /const approval = pendingApproval\(useChatRuntimeStore\(\(s\) => s\.toolConfirmations\)\)/,
  );
  assert.match(orb, /card: showOverlay && approval !== null/);
  assert.match(
    orb,
    /<OrbApprovalCard key=\{approval\.id\} id=\{approval\.id\} more=\{approval\.more\} \/>/,
  );
});

test("Allow, Always allow and Deny make the chat card's decision call", () => {
  assert.match(
    orb,
    /decideToolCall\(id, decision, always \? toolDetails\(id\)\.name : undefined\)/,
  );
  assert.match(orb, /onClick=\{\(\) => decide\("allow"\)\}/);
  assert.match(orb, /onClick=\{\(\) => decide\("allow", true\)\}/);
  assert.match(orb, /onClick=\{\(\) => decide\("deny"\)\}/);
  const decision = readFileSync(
    new URL("../src/features/chat/tool-decision.ts", import.meta.url),
    "utf-8",
  );
  assert.match(
    decision,
    /if \(decision === "allow" && alwaysToolName && autoAllowKey\) after\.allowToolAlways/,
  );
});

test("the card is one wrapping row with no expand toggle", () => {
  const card = orb.slice(
    orb.indexOf("const OrbApprovalCard"),
    orb.indexOf("const COMPOSER_CLEARANCE"),
  );
  assert.doesNotMatch(card, /aria-expanded|ChevronDownIcon/);
  assert.match(card, /flex-wrap items-center/);
  assert.match(card, /36rem/);
});

test("the column lifts to keep the card 24 px above the composer, measured from the layout", () => {
  assert.match(orb, /const COMPOSER_CLEARANCE = 24;/);
  assert.match(orb, /querySelector\("\.aui-composer-root"\)/);
  assert.match(
    orb,
    /composer\.getBoundingClientRect\(\)\.top - COMPOSER_CLEARANCE - cardBottom/,
  );
  assert.match(orb, /new ResizeObserver/);
});
