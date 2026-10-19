// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_POPUP_HEIGHT,
  ORB_WIDGET_WIDTH,
  isOrbWidgetDecision,
  pendingApproval,
  placeApprovalPopup,
  sameApproval,
} from "../src/features/chat/voice/orb-widget.ts";

const readSrc = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf-8");
const entry = { approvalId: "a", sessionId: "s", autoAllowKey: "k" };
const screen = { x: 0, y: 0, width: 1440, height: 900 };

test("a pending approval shows the first call, with a count of the rest", () => {
  assert.deepEqual(pendingApproval({ call1: entry }), { id: "call1", more: 0 });
  assert.deepEqual(pendingApproval({ call1: entry, call2: entry, call3: entry }), { id: "call1", more: 2 });
});

test("the popup goes once the approval is answered from anywhere", () => {
  // Every path (chat card, popup, voice cancelling the run) clears the store entry.
  assert.equal(pendingApproval({}), null);
  const host = readSrc("features/chat/voice/orb-widget-host.ts");
  assert.match(host, /s\.toolConfirmations !== prev\.toolConfirmations/);
  const a = { id: "c", line: "Using tool: x", more: null, allow: "Allow", deny: "Deny" };
  assert.ok(sameApproval(a, { ...a }));
  assert.ok(!sameApproval(a, null));
  assert.ok(!sameApproval(a, { ...a, more: "+1 more" }));
});

test("Allow and Deny on the popup make the same call as the chat card's buttons", () => {
  const controls = readSrc("components/assistant-ui/tool-confirmation-controls.tsx");
  const host = readSrc("features/chat/voice/orb-widget-host.ts");
  assert.match(controls, /await decideToolCall\(toolCallId, decision\)/);
  assert.match(host, /decideToolCall\(payload\.id, payload\.decision\)/);
  const view = readSrc("components/assistant-ui/orb-widget-view.tsx");
  assert.match(view, /decide\(approval\.id, "allow"\)/);
  assert.match(view, /decide\(approval\.id, "deny"\)/);
  assert.doesNotMatch(view, /Always allow|alsoAlways/);
});

test("only a well-formed decision is taken from the widget", () => {
  assert.ok(isOrbWidgetDecision({ id: "c", decision: "allow" }));
  assert.ok(isOrbWidgetDecision({ id: "c", decision: "deny" }));
  for (const bad of [null, "allow", { id: "", decision: "allow" }, { id: "c", decision: "always" }, { decision: "deny" }]) {
    assert.ok(!isOrbWidgetDecision(bad));
  }
});

test("the popup grows the window downward, or upward near the bottom, keeping the ball still", () => {
  const below = placeApprovalPopup({ x: 600, y: 300 }, [screen]);
  assert.deepEqual(below, { at: { x: 600, y: 300 }, above: false, pad: 0 });
  const nearBottom = { x: 600, y: screen.height - ORB_WIDGET_HEIGHT - 10 };
  const above = placeApprovalPopup(nearBottom, [screen]);
  assert.equal(above.above, true);
  assert.equal(above.at.y + above.pad, nearBottom.y, "the ball stays where it was");
  assert.ok(above.at.y >= screen.y);
});

test("the grown window stays on its screen on every side", () => {
  for (const pos of [
    { x: -80, y: 300 },
    { x: screen.width - 60, y: 300 },
    { x: 600, y: screen.height - 120 },
    { x: 600, y: -40 },
  ]) {
    const p = placeApprovalPopup(pos, [screen]);
    const height = ORB_WIDGET_HEIGHT + ORB_WIDGET_POPUP_HEIGHT;
    assert.ok(p.at.x >= 0 && p.at.x + ORB_WIDGET_WIDTH <= screen.width, `x ${p.at.x}`);
    // The popup itself: above, the top strip; below, the bottom strip.
    const top = p.above ? p.at.y : p.at.y + ORB_WIDGET_HEIGHT;
    assert.ok(top >= 0 && top + ORB_WIDGET_POPUP_HEIGHT <= screen.height, `popup at ${top} for ${JSON.stringify(pos)}`);
    assert.ok(p.at.y + (p.above ? 0 : height) <= screen.height || p.above);
  }
});
