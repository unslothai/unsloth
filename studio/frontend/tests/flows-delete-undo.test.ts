// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Deleting cards is one undo step, and the history keys work on Mac and Windows.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { doc, edge, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { historyAction } = await import(
  "../src/features/flows/editor/history-keys.ts"
);

const read = (path: string) =>
  readFileSync(new URL(`../src/features/flows/${path}`, import.meta.url), "utf8");

test("one undo restores a deleted card with its wires, group and params; redo deletes it again", () => {
  const b = { ...node("b", "llm.generate", [300, 0]), params: { temperature: 0.3 } };
  useDocStore.getState().load(
    doc(
      [node("a", "input.text"), b, node("c", "output.preview", [600, 0])],
      [edge("e1", "a", "text", "b", "prompt"), edge("e2", "b", "text", "c", "value")],
    ),
  );
  const s = useDocStore.getState();
  s.groupNodes({ id: "g1", label: "Group", color: "#8b5cf6", rect: [-20, -20, 900, 300] }, ["a", "b"]);
  const before = useDocStore.getState().graph;
  const depth = useDocStore.getState().past.length;

  useDocStore.getState().removeElements({ nodes: ["b"] });
  const gone = useDocStore.getState().graph;
  assert.equal(gone.nodes.b, undefined);
  assert.equal(gone.edges.e1, undefined);
  assert.equal(gone.edges.e2, undefined);
  assert.equal(useDocStore.getState().past.length, depth + 1, "one history entry");

  useDocStore.getState().undo();
  const back = useDocStore.getState().graph;
  assert.deepEqual(back.nodes.b, before.nodes.b);
  assert.equal(back.nodes.b.group, "g1");
  assert.deepEqual(back.nodes.b.params, { temperature: 0.3 });
  assert.deepEqual(back.edges.e1, before.edges.e1);
  assert.deepEqual(back.edges.e2, before.edges.e2);
  assert.deepEqual(back.edgeOrder, before.edgeOrder);

  useDocStore.getState().redo();
  assert.equal(useDocStore.getState().graph.nodes.b, undefined);
  assert.equal(useDocStore.getState().graph.edges.e1, undefined);
});

test("history keys: Ctrl/Cmd+Z undoes, Ctrl/Cmd+Shift+Z and Ctrl/Cmd+Y redo", () => {
  const k = (key: string, mods: Partial<Record<"metaKey" | "ctrlKey" | "shiftKey" | "altKey", boolean>>, code?: string) =>
    historyAction({ key, code, metaKey: false, ctrlKey: false, shiftKey: false, ...mods });
  for (const mod of ["metaKey", "ctrlKey"] as const) {
    assert.equal(k("z", { [mod]: true }), "undo");
    assert.equal(k("Z", { [mod]: true, shiftKey: true }), "redo");
    assert.equal(k("y", { [mod]: true }), "redo");
  }
  assert.equal(k("z", {}), null);
  assert.equal(k("y", {}), null);
  assert.equal(k("z", { ctrlKey: true, altKey: true }), null);
  assert.equal(k("я", { ctrlKey: true }, "KeyZ"), "undo", "non-Latin layout");
  assert.equal(k("x", { ctrlKey: true }), null);
});

test("undo listens in the capture phase and steps aside while typing", () => {
  const src = read("editor/use-editor-shortcuts.ts");
  assert.match(src, /addEventListener\("keydown", onHistoryKey, true\)/);
  assert.match(src, /if \(!action \|\| isTyping\(event\.target\)\) return;/);
});

test("the card toolbar deletes through the same path as the Delete key", () => {
  const src = read("editor/node-toolbar.tsx");
  // React Flow's Delete key calls deleteElements, which calls the canvas onDelete.
  assert.match(src, /deleteElements\(\{ nodes: \[\{ id: nodeId \}\] \}\)/);
  assert.match(src, /data-testid="node-toolbar-delete"/);
  assert.match(src, /below \? Position\.Bottom : Position\.Top/);
  assert.match(read("editor/canvas.tsx"), /onDelete=\{onDelete\}/);
});
