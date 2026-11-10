// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dragging: cards drag from anywhere, and the work a drag frame does is kept to what changed.

import assert from "node:assert/strict";
import test from "node:test";

import { catalog, doc, edge, node, specs } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");
const { applySuperCards, readSuperCards } = await import(
  "../src/features/flows/editor/super-cards.ts"
);
const { combineSelection } = await import("../src/features/flows/editor/super-card-actions.ts");
const { useUiStore } = await import("../src/features/flows/stores/ui-store.ts");
const { setupLayout } = await import("../src/features/flows/recipe/recipe-setup.ts");
const { wiresInto, wireIndexStats } = await import("../src/features/flows/stores/wire-index.ts");

function graph() {
  useDocStore.getState().load(
    doc(
      [
        node("prompt", "input.text", [0, 0]),
        node("tpl", "text.template", [300, 40]),
        node("img", "image.generate", [600, 0]),
        node("view", "output.preview", [900, 0]),
      ],
      [
        edge("e1", "prompt", "text", "tpl", "values"),
        edge("e2", "tpl", "text", "img", "prompt"),
        edge("e3", "img", "image", "view", "value"),
      ],
    ),
  );
}

test("a drag frame in a flow without a recipe does no setup layout work", () => {
  graph();
  const { graph: g } = useDocStore.getState();
  const first = setupLayout(g, { view: { x: 950, y: 10 } }, new Map());
  const second = setupLayout(g, { view: { x: 960, y: 20 } }, new Map());
  assert.equal(first, second, "one shared empty layout");
  assert.deepEqual(first, { elements: [], links: [], heights: {} });
});

test("a closed super card rebuilt with nothing changed is the same node object", () => {
  graph();
  useUiStore.getState().setSelection(["tpl", "img"], []);
  combineSelection();
  const state = useDocStore.getState();
  const cards = readSuperCards(state.meta.editor, state.graph);
  const build = (previous?: ReadonlyMap<string, never>, x = 900) => {
    const deriver = new RfDeriver();
    // Another card moves: a new nodes array, the closed card's members unchanged.
    const drag = { view: { x, y: 0 } };
    return applySuperCards(
      deriver.nodes(state.graph, new Set(), drag) as never,
      deriver.edges(state.graph, new Set(), () => "any"),
      state.graph,
      cards,
      specs,
      catalog.port_colors,
      new Set(),
      () => undefined,
      previous,
    );
  };
  const one = build();
  const card = one.nodes.find((n) => n.type === "flowSuperCard");
  assert.ok(card);
  const two = build(new Map([[card.id, card]]) as never, 950);
  assert.equal(two.nodes.find((n) => n.id === card.id), card);
  const selected = applySuperCards(
    new RfDeriver().nodes(state.graph, new Set(), null) as never,
    [],
    state.graph,
    cards,
    specs,
    catalog.port_colors,
    new Set([card.id]),
    () => undefined,
    new Map([[card.id, card]]) as never,
  );
  assert.notEqual(selected.nodes.find((n) => n.id === card.id), card, "a real change makes a new node");
});

test("card selectors look up their own wires: one index build per edge set", () => {
  graph();
  const g = useDocStore.getState().graph;
  const before = wireIndexStats.builds;
  // Three cards' selectors over 50 store updates that leave the edges alone.
  for (let k = 0; k < 50; k += 1)
    for (const id of ["tpl", "img", "view"]) wiresInto(g, id);
  assert.equal(wireIndexStats.builds - before, 1);
  assert.deepEqual(wiresInto(g, "img").map((e) => e.id), ["e2"]);
  assert.deepEqual(wiresInto(g, "prompt"), []);
});

test("a card has no drag handle, so pressing its body drags it", () => {
  useDocStore.getState().load(doc([node("a", "input.text")], []));
  const card = new RfDeriver()
    .nodes(useDocStore.getState().graph, new Set(), null)
    .find((n) => n.id === "a");
  assert.ok(card);
  assert.equal(card.type, "flowNode");
  assert.equal(card.dragHandle, undefined);
});
