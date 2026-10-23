// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);
const {
  applySuperCardView,
  combineSelection,
  createSuperCard,
  getCombinableSelection,
  superCardInputHandle,
  superCardOutputHandle,
  pruneSuperCards,
  superCardsFromUi,
  ungroupCard,
} = await import("../src/features/recipe-studio/utils/graph/super-cards.ts");

type SuperCardState = import("../src/features/recipe-studio/utils/graph/super-cards.ts").SuperCardState;

const CONVERSATION = readSrc(
  "features/data-recipes/learning-recipes/conversation.json",
);
const GROUPED = ["domain", "topic", "conversation_length"];

function loadSnapshot() {
  const imported = importRecipePayload(CONVERSATION);
  assert.deepEqual(imported.errors, []);
  assert.ok(imported.snapshot);
  return imported.snapshot;
}

function idsByName(
  snapshot: ReturnType<typeof loadSnapshot>,
  names: string[],
): string[] {
  return names.map((name) => {
    const config = Object.values(snapshot.configs).find(
      (item) => item.name === name,
    );
    assert.ok(config, `${name} should exist in the recipe`);
    return config.id;
  });
}

function build(
  snapshot: ReturnType<typeof loadSnapshot>,
  superCards: Parameters<typeof buildRecipePayload>[6] = {},
) {
  return buildRecipePayload(
    snapshot.configs,
    snapshot.nodes,
    snapshot.edges,
    snapshot.processors,
    snapshot.layoutDirection,
    snapshot.auxNodePositions,
    superCards,
  );
}

function groupedSnapshot() {
  const snapshot = loadSnapshot();
  const created = createSuperCard(
    {},
    snapshot.nodes,
    idsByName(snapshot, GROUPED),
  );
  assert.ok(created);
  return { snapshot, superCards: created.superCards, card: created.card };
}

test("a grouped recipe sends the backend exactly the payload of the ungrouped one", () => {
  const { snapshot, superCards } = groupedSnapshot();
  const ungrouped = build(snapshot);
  const grouped = build(snapshot, superCards);

  assert.deepEqual(grouped.errors, ungrouped.errors);
  assert.deepEqual(grouped.payload.recipe, ungrouped.payload.recipe);
  assert.deepEqual(grouped.payload.run, ungrouped.payload.run);
  const { super_cards: superCardsUi, ...groupedUi } = grouped.payload.ui;
  assert.deepEqual(groupedUi, ungrouped.payload.ui);
  assert.deepEqual(superCardsUi?.[0]?.members, GROUPED);
});

test("super cards round-trip through recipe JSON export and import", () => {
  const { snapshot, superCards, card } = groupedSnapshot();
  const exported = JSON.stringify(build(snapshot, superCards).payload);

  const reimported = importRecipePayload(exported);
  assert.deepEqual(reimported.errors, []);
  assert.ok(reimported.snapshot);
  const cards = Object.values(reimported.snapshot.superCards ?? {});
  assert.equal(cards.length, 1);
  assert.equal(cards[0].title, card.title);
  assert.deepEqual(cards[0].position, card.position);
  assert.deepEqual(
    cards[0].nodeIds.map((id) => reimported.snapshot?.configs[id]?.name),
    GROUPED,
  );
  assert.deepEqual(
    JSON.parse(JSON.stringify(build(reimported.snapshot, reimported.snapshot.superCards ?? {}).payload)),
    JSON.parse(exported),
  );
});

test("the root view folds members into one card and wires crossing edges to its handles", () => {
  const { snapshot, superCards, card } = groupedSnapshot();
  const graph = { nodes: snapshot.nodes, edges: snapshot.edges };
  const view = applySuperCardView({
    graph,
    superCards,
    openSuperCardId: null,
    selectedSuperCardIds: new Set(),
    configs: snapshot.configs,
    edges: snapshot.edges,
    layoutDirection: "LR",
  });
  const memberIds = new Set(card.nodeIds);
  const visibleIds = view.nodes.map((node) => node.id);
  assert.ok(visibleIds.includes(card.id));
  assert.ok(visibleIds.every((id) => !memberIds.has(id)));
  assert.equal(view.nodes.length, snapshot.nodes.length - memberIds.size + 1);

  const crossing = snapshot.edges.filter(
    (edge) => memberIds.has(edge.source) !== memberIds.has(edge.target),
  );
  assert.ok(crossing.length > 0, "the grouped samplers feed the LLM columns");
  for (const edge of view.edges) {
    assert.ok(!memberIds.has(edge.source) && !memberIds.has(edge.target));
  }
  for (const edge of crossing) {
    const outward = memberIds.has(edge.source);
    const handle = outward
      ? superCardOutputHandle(edge.source)
      : superCardInputHandle(edge.source);
    assert.ok(
      view.edges.some((viewEdge) =>
        outward
          ? viewEdge.source === card.id && viewEdge.sourceHandle === handle
          : viewEdge.target === card.id && viewEdge.targetHandle === handle,
      ),
      `crossing edge ${edge.id} should land on the card`,
    );
  }
  const data = view.nodes.find((node) => node.id === card.id)
    ?.data as unknown as {
    produces: string[];
    outputs: unknown[];
  };
  assert.deepEqual(data.produces, GROUPED);
  assert.ok(data.outputs.length > 0);

  const inside = applySuperCardView({
    graph,
    superCards,
    openSuperCardId: card.id,
    selectedSuperCardIds: new Set(),
    configs: snapshot.configs,
    edges: snapshot.edges,
    layoutDirection: "LR",
  });
  assert.deepEqual(
    inside.nodes.map((node) => node.id).sort(),
    [...card.nodeIds].sort(),
  );
  assert.ok(
    inside.edges.every(
      (edge) => memberIds.has(edge.source) && memberIds.has(edge.target),
    ),
  );
});

test("combine then ungroup puts the nodes back unchanged", () => {
  const snapshot = loadSnapshot();
  const before = structuredClone(snapshot.nodes);
  const grouped = new Set(idsByName(snapshot, GROUPED));
  let state: SuperCardState = {
    nodes: snapshot.nodes.map((node) => ({
      ...node,
      selected: grouped.has(node.id),
    })),
    superCards: {},
    openSuperCardId: null,
    selectedSuperCardIds: [],
  };

  state = { ...state, ...combineSelection(state) };
  const cards = Object.values(state.superCards);
  assert.equal(cards.length, 1);
  assert.deepEqual(new Set(cards[0].nodeIds), grouped);
  assert.ok(state.nodes.every((node) => !node.selected));

  state = { ...state, openSuperCardId: cards[0].id };
  state = { ...state, ...ungroupCard(state, cards[0].id) };
  assert.equal(state.openSuperCardId, null);
  assert.deepEqual(state.superCards, {});
  const strip = (nodes: typeof before) =>
    nodes.map(({ selected: _selected, ...node }) => node);
  assert.deepEqual(strip(state.nodes), strip(before));
  assert.ok(state.nodes.every((node) => node.selected === grouped.has(node.id)));
});

test("one nesting level: grouped nodes can't join a second card", () => {
  const { snapshot, superCards, card } = groupedSnapshot();
  const allSelected = snapshot.nodes.map((node) => ({
    ...node,
    selected: true,
  }));
  const combinable = getCombinableSelection(allSelected, superCards);
  assert.ok(combinable.every((id) => !card.nodeIds.includes(id)));

  const nameToId = new Map(
    Object.values(snapshot.configs).map((config) => [config.name, config.id]),
  );
  const parsed = superCardsFromUi(
    [
      { id: "sc1", title: "A", members: ["domain", "topic"], x: 0, y: 0 },
      { id: "sc2", title: "B", members: ["topic", "user_goal"], x: 0, y: 0 },
    ],
    nameToId,
  );
  assert.deepEqual(
    parsed.sc2.nodeIds.map((id) => snapshot.configs[id].name),
    ["user_goal"],
  );
});

test("removing a grouped node drops it from its card", () => {
  const { snapshot, superCards, card } = groupedSnapshot();
  const remaining = new Set(
    snapshot.nodes.map((node) => node.id).filter((id) => id !== card.nodeIds[0]),
  );
  assert.deepEqual(
    pruneSuperCards(superCards, remaining)[card.id]?.nodeIds,
    card.nodeIds.slice(1),
  );
  const emptied = new Set(
    [...remaining].filter((id) => !card.nodeIds.includes(id)),
  );
  assert.deepEqual(pruneSuperCards(superCards, emptied), {});
});
