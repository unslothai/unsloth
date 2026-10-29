// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cards drag from anywhere on the first press; notes drag and resize without a click first.

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");

test("a card has no drag handle, so pressing its body drags it", () => {
  useDocStore.getState().load(doc([node("a", "input.text")], []));
  const card = new RfDeriver()
    .nodes(useDocStore.getState().graph, new Set(), null)
    .find((n) => n.id === "a");
  assert.ok(card);
  assert.equal(card.type, "flowNode");
  assert.equal(card.dragHandle, undefined);
});

test("an empty note shows a placeholder, not a nodrag textarea, until it is double-clicked", () => {
  const src = readSrc("features/flows/editor/frame-nodes.tsx");
  const note = src.slice(src.indexOf("function NoteFrameImpl"));
  assert.match(note, /const showEditor = editing;/);
  assert.match(note, /Double-click to write a note/);
  assert.match(note, /onDoubleClick=\{\(e\) => \{\s*e\.stopPropagation\(\);\s*setEditing\(true\);/);
});

test("a note's resize handles are always mounted (invisible), so no select-first click", () => {
  const src = readSrc("features/flows/editor/frame-nodes.tsx");
  const note = src.slice(src.indexOf("function NoteFrameImpl"));
  assert.match(note, /<NodeResizer\s+isVisible=\{true\}/);
  assert.match(src, /handleStyle: \{ opacity: 0 \}/);
});
