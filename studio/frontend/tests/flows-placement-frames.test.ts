// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An armed Nodes panel item previews and places over a group or a note, not just bare canvas.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { placeableTarget } = await import("../src/features/flows/editor/placement.ts");

const el = (classes: string[], ancestors: string[] = []) => ({
  classList: { contains: (c: string) => classes.includes(c) },
  closest: (sel: string) => {
    const all = [...classes, ...ancestors];
    return sel.split(",").some((part) => {
      const s = part.trim();
      if (s.startsWith(".")) return all.includes(s.slice(1));
      return all.includes(`<${s.replace(/\[.*\]/, "")}>`);
    })
      ? {}
      : null;
  },
});

test("bare canvas, a group's body and a note's body take the card; cards and controls don't", () => {
  assert.equal(placeableTarget(el(["react-flow__pane"]) as never), true);
  assert.equal(placeableTarget(el(["group/frame"], ["react-flow__node-flowGroup"]) as never), true);
  assert.equal(placeableTarget(el(["text-xs"], ["react-flow__node-flowNote"]) as never), true);
  assert.equal(placeableTarget(el(["flex"], ["react-flow__node-flowNode"]) as never), false);
  assert.equal(placeableTarget(el(["size-6"], ["<button>", "react-flow__node-flowGroup"]) as never), false);
  assert.equal(placeableTarget(null), false);
});
