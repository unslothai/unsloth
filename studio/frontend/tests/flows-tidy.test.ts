// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { catalog, doc, edge, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useCatalogStore } = await import("../src/features/flows/stores/catalog-store.ts");
const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { computeLayout } = await import("../src/features/flows/editor/layout-core.ts");
const { TIDY_SPACING, fittedCardSize, tidyPatch, tidyRequest, CARD_FIT_MAX } = await import(
  "../src/features/flows/editor/tidy.ts"
);

useCatalogStore.getState().setCatalog(catalog);

function load() {
  const grouped = (id: string, type: string, pos: [number, number]) => ({ ...node(id, type, pos), group: "g1" });
  useDocStore.getState().load({
    ...doc(
      [
        node("src", "input.text", [0, 0]),
        grouped("llm", "llm.generate", [900, 900]),
        grouped("img", "image.generate", [900, 1200]),
        node("out", "output.preview", [-500, 2000]),
      ],
      [edge("e1", "src", "text", "llm", "prompt"), edge("e2", "llm", "text", "out", "value")],
    ),
    groups: [{ id: "g1", label: "Pair", rect: [880, 860, 400, 560] }],
    notes: [
      { id: "n1", text: "about out", rect: [-500, 2200, 200, 100] },
      { id: "n2", text: "inside", rect: [900, 1100, 200, 80] },
    ],
  });
}

const graph = () => useDocStore.getState().graph;
const ctx = (wireStyle: "bus" | "curved" = "bus") => ({
  graph: graph(),
  wireStyle,
  size: () => ({ width: 260, height: 140 }),
  portY: () => 40,
});

test("a group is laid out as one block: its cards keep their spots and the frame keeps its size", () => {
  load();
  const request = tidyRequest(ctx());
  const ids = request.nodes.map((n) => n.id).sort();
  assert.deepEqual(ids, ["group:g1", "out", "src"]);
  assert.deepEqual(
    request.nodes.find((n) => n.id === "group:g1"),
    { id: "group:g1", width: 400, height: 560 },
  );
  assert.deepEqual(
    request.edges.map((e) => `${e.source}>${e.target}`).sort(),
    ["group:g1>out", "src>group:g1"],
  );
  const before = { llm: graph().nodes.llm.pos, img: graph().nodes.img.pos, rect: graph().groups.g1.rect };
  useDocStore.getState().applyTidy(tidyPatch(ctx(), computeLayout(request)));
  const g = graph();
  const [dx, dy] = [g.groups.g1.rect[0] - before.rect[0], g.groups.g1.rect[1] - before.rect[1]];
  assert.deepEqual(g.groups.g1.rect.slice(2), [400, 560], "not resized");
  assert.deepEqual(g.nodes.llm.pos, [before.llm[0] + dx, before.llm[1] + dy]);
  assert.deepEqual(g.nodes.img.pos, [before.img[0] + dx, before.img[1] + dy]);
  assert.ok(g.nodes.src.pos[0] < g.groups.g1.rect[0] && g.groups.g1.rect[0] + 400 < g.nodes.out.pos[0], "left to right");
});

test("notes move with their frame or with the card nearest them", () => {
  load();
  const before = { n1: graph().notes.n1.rect, n2: graph().notes.n2.rect, out: graph().nodes.out.pos, g: graph().groups.g1.rect };
  useDocStore.getState().applyTidy(tidyPatch(ctx(), computeLayout(tidyRequest(ctx()))));
  const g = graph();
  const outMove = [g.nodes.out.pos[0] - before.out[0], g.nodes.out.pos[1] - before.out[1]];
  assert.deepEqual(g.notes.n1.rect, [before.n1[0] + outMove[0], before.n1[1] + outMove[1], 200, 100]);
  const gMove = [g.groups.g1.rect[0] - before.g[0], g.groups.g1.rect[1] - before.g[1]];
  assert.deepEqual(g.notes.n2.rect, [before.n2[0] + gMove[0], before.n2[1] + gMove[1], 200, 80]);
});

test("bus wires get wider lanes than curves, and tidy up is one undo step", () => {
  load();
  assert.equal(tidyRequest(ctx("bus")).ranksep, TIDY_SPACING.bus.ranksep);
  assert.equal(tidyRequest(ctx("curved")).nodesep, TIDY_SPACING.curved.nodesep);
  assert.ok(TIDY_SPACING.bus.ranksep > TIDY_SPACING.curved.ranksep);
  const before = graph();
  useDocStore.getState().applyTidy(tidyPatch(ctx(), computeLayout(tidyRequest(ctx())), { src: [300, 200] }));
  assert.deepEqual(graph().nodes.src.size, [300, 200]);
  useDocStore.getState().undo();
  assert.equal(graph(), before);
});

test("cards fit their output without growing past the cap", () => {
  assert.equal(fittedCardSize({ width: 260, height: 150 }, 0, 0, 220), null, "nothing hidden, nothing spare");
  const small = fittedCardSize({ width: 260, height: 150 }, 40, 0, 220);
  assert.deepEqual(small, [260, 196], "a little more text: taller only");
  const big = fittedCardSize({ width: 260, height: 150 }, 2000, 0, 220);
  assert.ok(big && big[0] === 340 && big[1] === CARD_FIT_MAX.height, JSON.stringify(big));
  assert.deepEqual(fittedCardSize({ width: 300, height: 400 }, 0, 120, 220), [300, 288], "spare room shrinks");
});
