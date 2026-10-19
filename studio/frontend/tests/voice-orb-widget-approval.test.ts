// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  ORB_WIDGET_BAR_HEIGHT,
  ORB_WIDGET_BAR_TOP,
  ORB_WIDGET_GROWN_HEIGHT,
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_POPUP_CARD_HEIGHT,
  ORB_WIDGET_POPUP_WIDTH,
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
  const a = {
    id: "c",
    line: "Using tool: x",
    brief: "x",
    more: null,
    allow: "Allow",
    deny: "Deny",
  };
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
  for (const bad of [
    null,
    "allow",
    { id: "", decision: "allow" },
    { id: "c", decision: "always" },
    { decision: "deny" },
  ]) {
    assert.ok(!isOrbWidgetDecision(bad));
  }
});

const cardOnScreen = (
  pos: { x: number; y: number },
  frames: { x: number; y: number; width: number; height: number }[],
  scale = 1,
) => {
  const p = placeApprovalPopup(pos, frames, scale);
  return {
    p,
    left: p.at.x + p.popupX * scale,
    top: p.at.y + p.popupY * scale,
    right: p.at.x + (p.popupX + ORB_WIDGET_POPUP_WIDTH) * scale,
    bottom: p.at.y + (p.popupY + ORB_WIDGET_POPUP_CARD_HEIGHT) * scale,
  };
};

test("the card sits under the bar, centred, and the ball keeps its place", () => {
  const { p } = cardOnScreen({ x: 600, y: 300 }, [screen]);
  assert.deepEqual(p.at, { x: 600, y: 300 });
  assert.equal(p.above, false);
  assert.equal(p.popupX, (ORB_WIDGET_WIDTH - ORB_WIDGET_POPUP_WIDTH) / 2);
  assert.ok(p.popupY >= ORB_WIDGET_BAR_TOP + ORB_WIDGET_BAR_HEIGHT);
  assert.ok(p.popupY + ORB_WIDGET_POPUP_CARD_HEIGHT <= ORB_WIDGET_GROWN_HEIGHT);
});

test("near the bottom the card goes above the ball", () => {
  const pos = { x: 600, y: screen.height - ORB_WIDGET_HEIGHT };
  const { p, top } = cardOnScreen(pos, [screen]);
  assert.equal(p.above, true);
  assert.equal(p.at.y + p.pad, pos.y, "the ball stays where it was");
  assert.ok(top >= screen.y);
});

test("the card never leaves the visible frame, on any side, on either of two monitors", () => {
  const frames = [screen, { x: 1440, y: 25, width: 1920, height: 1055 }];
  for (const frame of frames) {
    for (const pos of [
      { x: frame.x - 90, y: frame.y + 300 },
      { x: frame.x + frame.width - 100, y: frame.y + 300 },
      { x: frame.x + frame.width - 100, y: frame.y + frame.height - 120 },
      { x: frame.x + 400, y: frame.y - 80 },
      { x: frame.x - 90, y: frame.y + frame.height - 100 },
    ]) {
      const c = cardOnScreen(pos, frames);
      assert.ok(
        c.left >= frame.x && c.right <= frame.x + frame.width,
        `x ${c.left}..${c.right} at ${JSON.stringify(pos)}`,
      );
      assert.ok(
        c.top >= frame.y && c.bottom <= frame.y + frame.height,
        `y ${c.top}..${c.bottom} at ${JSON.stringify(pos)}`,
      );
    }
  }
});

test("off the right edge the card slides left of centre, moving the ball as little as it can", () => {
  const pos = { x: screen.width - 120, y: 300 };
  const c = cardOnScreen(pos, [screen]);
  assert.ok(c.right <= screen.width);
  assert.ok(c.p.popupX < (ORB_WIDGET_WIDTH - ORB_WIDGET_POPUP_WIDTH) / 2, "shifted off centre");
  assert.ok(c.p.at.x <= pos.x && pos.x - c.p.at.x < ORB_WIDGET_WIDTH);
});

test("placed in physical pixels at 2x, it still lands inside the frame", () => {
  const frame = { x: 0, y: 50, width: 2880, height: 1750 };
  const c = cardOnScreen({ x: 2880 - 150, y: 1750 - 200 }, [frame], 2);
  assert.ok(c.left >= frame.x && c.right <= frame.x + frame.width);
  assert.ok(c.top >= frame.y && c.bottom <= frame.y + frame.height);
});
