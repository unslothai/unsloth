// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { COLUMN_TYPE, groupColumns, groupPayload, recipeSignature } = await import(
  "../src/features/flows/recipe/recipe-columns.ts"
);
const { applyConfigPatch, commitStudioEdit, studioView } = await import(
  "../src/features/flows/recipe/recipe-adapter.ts"
);
const { jsonEqual } = await import("../src/features/flows/recipe/json-equal.ts");

function countStringify<T>(run: () => T): { value: T; calls: number } {
  const original = JSON.stringify;
  let calls = 0;
  JSON.stringify = ((...args: Parameters<typeof JSON.stringify>) => {
    calls += 1;
    return original(...args);
  }) as typeof JSON.stringify;
  try {
    return { value: run(), calls };
  } finally {
    JSON.stringify = original;
  }
}

function learningGroup() {
  const payload = JSON.parse(
    readFileSync(
      new URL("../src/features/data-recipes/learning-recipes/text-to-sql.json", import.meta.url),
      "utf8",
    ),
  ) as { recipe: Record<string, unknown> & { columns: Record<string, unknown>[] }; ui?: Record<string, unknown> };
  const { columns, ...config } = payload.recipe;
  const nodes = columns.map((column, i) => ({
    id: `recipe_col_${i + 1}`,
    type: COLUMN_TYPE,
    v: 1,
    params: { column },
    pos: [i * 320, 0] as [number, number],
    label: String(column.name),
    group: "recipe",
  }));
  const group = {
    id: "recipe",
    label: "Recipe",
    rect: [0, 0, 4000, 400] as [number, number, number, number],
    kind: "recipe" as const,
    config,
    run: { rows: 5 },
    ui: payload.ui ?? {},
  };
  return { group, nodes };
}

test("a recipe form's signature selector does not re-serialize the recipe on unrelated updates", (t) => {
  const { group, nodes } = learningGroup();
  const d = doc([...nodes, node("free", "input.text", [0, 900], { value: "x" })]);
  d.groups = [group];
  useDocStore.getState().load(d);
  const first = recipeSignature(useDocStore.getState().graph, "recipe");
  const N = 50;
  const { calls } = countStringify(() => {
    for (let i = 0; i < N; i += 1) {
      useDocStore.getState().moveNodes({ free: [i, 900] });
      assert.equal(recipeSignature(useDocStore.getState().graph, "recipe"), first);
    }
  });
  t.diagnostic(`stringify calls for ${N} unrelated updates: ${calls}`);
  assert.equal(calls, 0);

  const graph = useDocStore.getState().graph;
  const column = graph.nodes.recipe_col_1;
  useDocStore.getState().patchGraph({
    nodes: [{ ...column, params: { column: { ...(column.params.column as object), drop: false } } }],
  });
  const next = useDocStore.getState().graph;
  const edited = recipeSignature(next, "recipe");
  assert.ok(edited !== first, "an edited column changes the signature");
  assert.equal(
    edited,
    JSON.stringify(groupPayload(next.groups.recipe, groupColumns(next, "recipe"))),
  );
  assert.equal(recipeSignature(next, "missing"), "");
});

test("a recipe form edit compares its parts without serializing them", (t) => {
  const { group, nodes } = learningGroup();
  const view = studioView(group, nodes);
  const preset = Object.values(view.configs).find((c) => c.kind === "model_config");
  assert.ok(preset);
  const configs = applyConfigPatch(view.configs, preset.id, { model: "other" } as never);
  const { value: edit, calls } = countStringify(() =>
    commitStudioEdit(group, nodes, view, configs),
  );
  t.diagnostic(`stringify calls for one preset edit: ${calls}`);
  const models = (edit.group?.config as { model_configs: { model: string }[] }).model_configs;
  assert.equal(models[0].model, "other");
  assert.equal(calls, 0);
});

test("jsonEqual agrees with comparing JSON text", () => {
  const values: unknown[] = [
    undefined, null, 0, -0, 1, Number.NaN, Number.POSITIVE_INFINITY, "", "1", true, false,
    [], {}, [undefined], [null], [() => 1], [Number.NaN], { a: undefined }, { a: () => 1 },
    { a: 1, b: 2 }, { b: 2, a: 1 }, { a: 1, b: undefined }, { a: [1, { c: "x" }] },
    { a: [1, { c: "y" }] }, { a: null }, { a: Number.NaN }, [1, 2], [2, 1], { 0: "a" }, ["a"],
    new Date(0), "1970-01-01T00:00:00.000Z", new Number(1), new String("1"), { t: new Date(0) },
    () => 1, Symbol("s"), { nested: { deep: [[], {}, [undefined, 1]] } },
    { nested: { deep: [[], {}, [null, 1]] } },
  ];
  for (const a of values)
    for (const b of values)
      assert.equal(jsonEqual(a, b), JSON.stringify(a) === JSON.stringify(b), `${String(JSON.stringify(a))} vs ${String(JSON.stringify(b))}`);
});
