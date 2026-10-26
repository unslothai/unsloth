// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { HANDLE_IDS, LEGACY_HANDLE_ALIAS_MAP, getRecipeHandleSide, migrateRecipeEdgeHandles } =
  await import("../src/features/recipe-studio/utils/handles.ts");
const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);

const LEGAL = new Set<string>(Object.values(HANDLE_IDS));
// Every handle id a saved recipe can hold that the canvas no longer draws, top and bottom ones
// from the old top-to-bottom layout included.
const REMOVED = [
  "data-in-top",
  "data-in-right",
  "data-in-bottom",
  "data-out-bottom",
  "data-out-left",
  "data-out-top",
  "semantic-in-top",
  "semantic-in-right",
  "semantic-in-bottom",
  "semantic-out-bottom",
  "semantic-out-left",
  "semantic-out-top",
  "llm-input-out-bottom",
  "llm-input-out-left",
  "llm-input-out-top",
];

test("every removed handle id maps to the left or right handle of its lane", () => {
  for (const id of REMOVED) {
    const mapped = LEGACY_HANDLE_ALIAS_MAP[id];
    assert.ok(mapped, `${id} has a replacement`);
    assert.ok(LEGAL.has(mapped), `${id} -> ${mapped} is a rendered handle`);
    assert.equal(mapped.split("-")[0], id.split("-")[0], `${id} stays in its lane`);
    assert.doesNotMatch(mapped, /-(top|bottom)$/, `${id} -> ${mapped}`);
  }
});

test("inputs only sit on the left and outputs only on the right", () => {
  for (const id of [...LEGAL, ...REMOVED]) {
    assert.equal(getRecipeHandleSide(id, "target"), "left", id);
    assert.equal(getRecipeHandleSide(id, "source"), "right", id);
  }
});

test("a stored edge on a removed handle is remapped, never dropped", () => {
  const edge = {
    id: "e-a-b",
    source: "a",
    target: "b",
    sourceHandle: "data-out-bottom",
    targetHandle: "data-in-top",
  };
  assert.deepEqual(migrateRecipeEdgeHandles(edge), {
    ...edge,
    sourceHandle: HANDLE_IDS.dataOut,
    targetHandle: HANDLE_IDS.dataIn,
  });
  const clean = { ...edge, sourceHandle: HANDLE_IDS.dataOut, targetHandle: HANDLE_IDS.dataIn };
  assert.equal(migrateRecipeEdgeHandles(clean), clean, "legal edges are left untouched");
});

// The handles older saves wrote in place of the ones a save writes now.
const TO_REMOVED: Record<string, string> = {
  "data-out": "data-out-bottom",
  "data-in": "data-in-top",
  "semantic-out": "semantic-out-bottom",
  "semantic-in": "semantic-in-top",
  "model-in": "data-in-top",
  "tools-in": "data-in-top",
};

const recipeDir = new URL(
  "../src/features/data-recipes/learning-recipes/",
  import.meta.url,
);
const recipes = readdirSync(recipeDir).filter((name) => name.endsWith(".json"));

test("learning recipes saved with the old handles import with every edge intact", () => {
  const seen = new Set<string>();
  assert.ok(recipes.length > 0);
  for (const name of recipes) {
    const original = JSON.parse(
      readSrc(`features/data-recipes/learning-recipes/${name}`),
    );
    const legacy = structuredClone(original);
    for (const edge of legacy.ui?.edges ?? []) {
      for (const key of ["source_handle", "target_handle"]) {
        const old = TO_REMOVED[edge[key]];
        if (old) {
          edge[key] = old;
          seen.add(old);
        }
      }
    }
    const before = importRecipePayload(JSON.stringify(original));
    const after = importRecipePayload(JSON.stringify(legacy));
    assert.deepEqual(after.errors, before.errors, name);
    const edges = after.snapshot?.edges ?? [];
    assert.equal(edges.length, before.snapshot?.edges.length, `${name} keeps every edge`);
    for (const edge of edges) {
      assert.ok(LEGAL.has(edge.sourceHandle ?? ""), `${name}: ${edge.sourceHandle}`);
      assert.ok(LEGAL.has(edge.targetHandle ?? ""), `${name}: ${edge.targetHandle}`);
    }
    assert.deepEqual(
      edges.map((edge) => [edge.id, edge.sourceHandle, edge.targetHandle]),
      (before.snapshot?.edges ?? []).map((edge) => [
        edge.id,
        edge.sourceHandle,
        edge.targetHandle,
      ]),
      `${name} remaps to the same handles a fresh save uses`,
    );
  }
  assert.ok(seen.size >= 4, `covered ${[...seen].join(", ")}`);
});

test("each removed data and semantic handle survives import on a hand-built recipe", () => {
  for (const [legal, removed] of Object.entries(TO_REMOVED)) {
    const semantic = legal.startsWith("semantic");
    const payload = {
      recipe: {
        columns: [
          {
            column_type: "sampler",
            name: "topic",
            sampler_type: "category",
            params: { values: ["a", "b"] },
          },
          {
            column_type: "expression",
            name: "shout",
            expr: "{{ topic | upper }}",
          },
        ],
      },
      ui: {
        nodes: [],
        edges: [
          {
            from: "topic",
            to: "shout",
            type: semantic ? "semantic" : "canvas",
            source_handle: legal.includes("-out") ? removed : HANDLE_IDS.dataOut,
            target_handle: legal.includes("-in") ? removed : HANDLE_IDS.dataIn,
          },
        ],
      },
    };
    const result = importRecipePayload(JSON.stringify(payload));
    const edges = result.snapshot?.edges ?? [];
    assert.equal(edges.length, 1, `${removed}: the edge is kept`);
    assert.ok(LEGAL.has(edges[0].sourceHandle ?? ""), `${removed}: ${edges[0].sourceHandle}`);
    assert.ok(LEGAL.has(edges[0].targetHandle ?? ""), `${removed}: ${edges[0].targetHandle}`);
  }
});
