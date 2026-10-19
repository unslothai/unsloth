// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ORB_WIDGET_BAR_HEIGHT,
  ORB_WIDGET_BAR_TOP,
  ORB_WIDGET_CENTER,
  ORB_WIDGET_GROWN_HEIGHT,
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_POPUP_CARD_HEIGHT,
  ORB_WIDGET_POPUP_WIDTH,
  ORB_WIDGET_WIDTH,
  alwaysAllowToolName,
  applyOrbWidgetTheme,
  clampToScreens,
  isOrbWidgetDecision,
  pendingApproval,
  placeApprovalPopup,
  restoreAfterPopup,
  sameApproval,
} from "../src/features/chat/voice/orb-widget.ts";

const entry = (thread: string) => ({ approvalId: "a", sessionId: "s", autoAllowKey: `s:${thread}` });
const screen = { x: 0, y: 0, width: 1440, height: 900 };
const nowhere = () => false;

test("a pending approval shows the voiced chat's first call, with a count of the rest", () => {
  const a = entry("A");
  assert.deepEqual(pendingApproval({ call1: a }, "A", nowhere), { id: "call1", more: 0 });
  assert.deepEqual(pendingApproval({ call1: a, call2: a, call3: a }, "A", nowhere), { id: "call1", more: 2 });
});

test("a call parked in another chat (a Compare pane, a background run) is not the voiced chat's", () => {
  const store = {
    "call-compare-pane-B": entry("B"),
    "call-voice-thread-A": entry("A"),
    "call-background-C": entry("C"),
  };
  assert.deepEqual(pendingApproval(store, "A", nowhere), { id: "call-voice-thread-A", more: 0 });
  assert.equal(pendingApproval({ "call-compare-pane-B": entry("B") }, "A", nowhere), null);
  // A thread id that is only the tail of another's is not the same thread.
  assert.equal(pendingApproval({ x: entry("xA") }, "A", nowhere), null);
});

test("a call the voice loop finds in its own thread counts even when its scope has no thread", () => {
  const store = { unthreaded: { approvalId: "a", sessionId: "s", autoAllowKey: "s" }, other: entry("B") };
  assert.deepEqual(
    pendingApproval(store, null, (id) => id === "unthreaded"),
    { id: "unthreaded", more: 0 },
  );
  assert.equal(pendingApproval(store, null, nowhere), null);
});

test("Always allow is offered only for a tool the voiced thread names", () => {
  assert.equal(alwaysAllowToolName({ toolName: "web_search" }), "web_search");
  for (const unknown of [null, undefined, {}, { toolName: "" }, { toolName: "  " }]) {
    assert.equal(alwaysAllowToolName(unknown), null);
  }
});

test("sameApproval compares every field the popup shows", () => {
  assert.equal(pendingApproval({}, "A", nowhere), null);
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

test("the widget takes the main window's theme and palette the way theme-boot.js sets them", () => {
  const classes = new Set<string>(["light"]);
  const attrs = new Map<string, string>();
  const root = {
    classList: {
      toggle: (name: string, force: boolean) => (force ? classes.add(name) : classes.delete(name)),
    },
    style: { colorScheme: "light" },
    setAttribute: (n: string, v: string) => attrs.set(n, v),
    removeAttribute: (n: string) => attrs.delete(n),
  };
  applyOrbWidgetTheme(root, { dark: true, palette: "matcha" });
  assert.deepEqual([...classes], ["dark"]);
  assert.equal(root.style.colorScheme, "dark");
  assert.equal(attrs.get("data-palette"), "matcha");
  applyOrbWidgetTheme(root, { dark: false, palette: null });
  assert.deepEqual([...classes], ["light"]);
  assert.equal(root.style.colorScheme, "light");
  assert.ok(!attrs.has("data-palette"));
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

test("a ball at the left edge gets its x back once the approval is answered", () => {
  const pos = clampToScreens({ x: -200, y: 300 }, [screen]);
  const p = placeApprovalPopup(pos, [screen], 1);
  // Held on screen while the card shows: the window moved right to fit it.
  assert.ok(p.at.x > pos.x);
  assert.deepEqual(p.shift, { x: p.at.x - pos.x, y: p.at.y - pos.y });
  assert.deepEqual(restoreAfterPopup(p.at, p.shift), pos);
  // The ball's centre returns too, not only the window.
  assert.equal(restoreAfterPopup(p.at, p.shift).x + ORB_WIDGET_CENTER.x, pos.x + ORB_WIDGET_CENTER.x);
});

test("every shift, sideways or up, is undone on dismiss, at 1x and 2x", () => {
  const frames = [screen, { x: 1440, y: 25, width: 1920, height: 1055 }];
  for (const scale of [1, 2]) {
    for (const pos of [
      { x: -90, y: 300 },
      { x: 1440 - 100, y: 300 },
      { x: 600, y: 900 - ORB_WIDGET_HEIGHT },
      { x: -90, y: 900 - 100 },
      { x: 1440 + 1920 - 100, y: 25 + 1055 - 120 },
    ]) {
      const p = placeApprovalPopup(pos, frames, scale);
      assert.deepEqual(restoreAfterPopup(p.at, p.shift), pos, JSON.stringify({ pos, scale }));
    }
  }
  // Nothing moved, nothing to undo.
  const still = placeApprovalPopup({ x: 600, y: 300 }, [screen]);
  assert.deepEqual(still.shift, { x: 0, y: 0 });
});

test("dragged while the card shows, it lands where it was dropped less the shift", () => {
  const pos = { x: -90, y: 300 };
  const p = placeApprovalPopup(pos, [screen]);
  const dropped = { x: p.at.x + 300, y: p.at.y + 40 };
  assert.deepEqual(restoreAfterPopup(dropped, p.shift), { x: pos.x + 300, y: pos.y + 40 });
});
