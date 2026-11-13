// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { placeTitleTooltip, resolveTitleTarget, titleStandIn } from "../src/lib/title-tooltip-placement.ts";

const viewport = { width: 1000, height: 800 };
const size = { width: 100, height: 30 };

test("a title tooltip sits centred below its element", () => {
  assert.deepEqual(placeTitleTooltip({ left: 400, top: 100, bottom: 120, width: 40 }, size, viewport), { left: 370, top: 126 });
});

test("near the bottom edge it flips above", () => {
  assert.deepEqual(placeTitleTooltip({ left: 400, top: 760, bottom: 780, width: 40 }, size, viewport), { left: 370, top: 724 });
});

test("it stays inside the left and right edges", () => {
  assert.equal(placeTitleTooltip({ left: 0, top: 100, bottom: 120, width: 20 }, size, viewport).left, 8);
  assert.equal(placeTitleTooltip({ left: 980, top: 100, bottom: 120, width: 20 }, size, viewport).left, 892);
});

type Node = { name: string; kids: Node[]; contains(other: Node | null): boolean };
function node(name: string, kids: Node[] = []): Node {
  const n: Node = {
    name,
    kids,
    contains: (other) => other === n || kids.some((k) => k.contains(other)),
  };
  return n;
}

// Codex review on #14172: the lifted title made every second move release the tooltip.
test("a control keeps its title tooltip while the pointer moves over it", () => {
  const inner = node("inner");
  const button = node("button", [inner]);
  const panel = node("panel", [button]);
  // The lifted button is skipped, so the hit resolves to a titled ancestor, or to nothing.
  assert.equal(resolveTitleTarget(button, panel, true), button);
  assert.equal(resolveTitleTarget(button, null, true), button);
  assert.equal(resolveTitleTarget(button, inner, true), inner, "a titled child inside it still wins");
  assert.equal(resolveTitleTarget(button, panel, false), panel, "leaving it moves on");
  assert.equal(resolveTitleTarget(null, panel, false), panel);
});

// Codex review on #14172: lifting the title must not strip an icon button's only name.
test("a lifted title keeps its meaning for assistive tech", () => {
  const s = (labelled: boolean, named: boolean, described = false) => titleStandIn({ labelled, named, described });
  assert.equal(s(false, false), "aria-label", "icon-only: the title was its name");
  assert.equal(s(false, true), "aria-description", "named by text: the title was its description");
  assert.equal(s(true, false), "aria-description", "named by aria-label: the title is still its description");
  assert.equal(s(true, true, true), null, "an existing description outranks the title");
});
