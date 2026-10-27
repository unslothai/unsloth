// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { applyConfigPatch, commitStudioEdit, studioView } = await import(
  "../src/features/flows/recipe/recipe-adapter.ts"
);
const {
  COLUMN_TYPE,
  columnName,
  columnSummary,
  referenceEdgeDiff,
  referenceEdges,
  withReference,
  withoutReference,
} = await import("../src/features/flows/recipe/recipe-columns.ts");

type FlowNode = Parameters<typeof columnName>[0];

const SQL = JSON.parse(
  readFileSync(
    new URL(
      "../src/features/data-recipes/learning-recipes/text-to-sql.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  recipe: Record<string, unknown> & { columns: Record<string, unknown>[] };
};

function sqlGroup() {
  const { columns, ...config } = structuredClone(SQL.recipe);
  const nodes: FlowNode[] = columns.map((column, i) => ({
    id: `recipe_col_${i + 1}`,
    type: COLUMN_TYPE,
    v: 1,
    params: { column },
    pos: [i * 320, 0],
    label: String(column.name),
    group: "recipe",
  }));
  const group = {
    id: "recipe",
    label: "Text to SQL",
    rect: [0, 0, 100, 100] as [number, number, number, number],
    kind: "recipe" as const,
    config,
    run: { rows: 5 },
  };
  return { group, nodes };
}

function byName(nodes: FlowNode[], name: string): FlowNode {
  const found = nodes.find((n) => columnName(n) === name);
  assert.ok(found, name);
  return found;
}

test("every column of a recipe group opens as the Recipe Studio block it was", () => {
  const { group, nodes } = sqlGroup();
  const view = studioView(group, nodes);
  assert.deepEqual(view.errors, []);
  for (const node of nodes) {
    const config = view.configs[view.byNode[node.id]];
    assert.equal(config.name, columnName(node));
  }
  const kinds = Object.values(view.configs).map((c) => c.kind);
  assert.ok(kinds.includes("model_provider") && kinds.includes("model_config"));
  assert.equal(view.configs[view.byNode[byName(nodes, "sql").id]].kind, "llm");
});

test("renaming a column rewrites every {{ reference }} to it and touches nothing else", () => {
  const { group, nodes } = sqlGroup();
  const view = studioView(group, nodes);
  const id = view.byNode[byName(nodes, "sql_prompt").id];
  const configs = applyConfigPatch(view.configs, id, { name: "task" });
  const edit = commitStudioEdit(group, nodes, view, configs);
  const changed = new Map(edit.nodes.map((n) => [n.id, n]));
  const renamed = changed.get(byName(nodes, "sql_prompt").id);
  assert.ok(renamed);
  assert.equal(columnName(renamed), "task");
  assert.equal(renamed.label, "task");
  const sql = changed.get(byName(nodes, "sql").id);
  assert.ok(sql, "the column that reads it is rewritten");
  assert.match(
    String((sql.params.column as Record<string, unknown>).prompt),
    /\{\{ task \}\}/,
  );
  assert.doesNotMatch(
    String((sql.params.column as Record<string, unknown>).prompt),
    /sql_prompt/,
  );
  assert.equal(changed.has(byName(nodes, "domain").id), false);
  assert.equal(edit.group, null, "no recipe-level change");
});

test("a recipe-level edit lands on the group config, not on the columns", () => {
  const { group, nodes } = sqlGroup();
  const view = studioView(group, nodes);
  const provider = Object.values(view.configs).find(
    (c) => c.kind === "model_provider",
  );
  assert.ok(provider);
  const configs = applyConfigPatch(view.configs, provider.id, {
    endpoint: "http://127.0.0.1:8000/v1",
  } as never);
  const edit = commitStudioEdit(group, nodes, view, configs);
  assert.deepEqual(edit.nodes, []);
  const providers = edit.group?.config?.model_providers as Record<
    string,
    unknown
  >[];
  assert.equal(providers[0].endpoint, "http://127.0.0.1:8000/v1");
  assert.equal("columns" in (edit.group?.config ?? {}), false);
  assert.equal("mcp_providers" in (edit.group?.config ?? {}), false);
});

test("reference wires follow the prompts, like the backend import", () => {
  const { nodes } = sqlGroup();
  const pairs = referenceEdges(nodes).map(
    (e) =>
      `${columnName(nodes.find((n) => n.id === e.source) as FlowNode)}>${columnName(nodes.find((n) => n.id === e.target) as FlowNode)}`,
  );
  assert.deepEqual(
    new Set(pairs),
    new Set([
      "domain>topic",
      "domain>sql_prompt",
      "topic>sql_prompt",
      "sql_task_type>sql_prompt",
      "instruction_phrase>sql_prompt",
      "sql_prompt>sql",
      "sql_task_type>sql",
      "sql>sql-validator",
    ]),
  );
  const graph = {
    nodes: Object.fromEntries(nodes.map((n) => [n.id, n])),
    nodeOrder: nodes.map((n) => n.id),
    edges: {
      stale: {
        id: "stale",
        source: nodes[6].id,
        source_port: "out",
        target: nodes[0].id,
        target_port: "in",
        kind: "data" as const,
      },
    },
    edgeOrder: ["stale"],
    groups: {},
    groupOrder: [],
    notes: {},
    noteOrder: [],
  };
  const diff = referenceEdgeDiff(graph, nodes);
  assert.deepEqual(diff.remove, ["stale"]);
  assert.equal(diff.add.length, 8);
});

test("wiring one column into another adds the {{ reference }}, unwiring takes it out", () => {
  const { nodes } = sqlGroup();
  const sql = byName(nodes, "sql").params.column as Record<string, unknown>;
  const wired = withReference(sql, "domain");
  assert.ok(wired);
  assert.match(String(wired.prompt), /\{\{ domain \}\}$/);
  assert.equal(
    withReference(wired, "domain"),
    wired,
    "wiring twice changes nothing",
  );
  assert.equal(withoutReference(wired, "domain").prompt, sql.prompt);
  const validator = byName(nodes, "sql-validator").params.column as Record<
    string,
    unknown
  >;
  assert.deepEqual(withReference(validator, "sql_prompt")?.target_columns, [
    "sql",
    "sql_prompt",
  ]);
  assert.equal(
    withReference(
      byName(nodes, "topic").params.column as Record<string, unknown>,
      "domain",
    ),
    null,
  );
});

test("a column node shows its kind and what it says", () => {
  const { nodes } = sqlGroup();
  assert.deepEqual(
    columnSummary(nodes[0].params.column as Record<string, unknown>).badge,
    "category sampler",
  );
  assert.equal(
    columnSummary(byName(nodes, "sql").params.column as Record<string, unknown>)
      .badge,
    "LLM code",
  );
});

test("the catalog snapshot has the recipe column and Recipe Dataset blocks", () => {
  const catalog = JSON.parse(
    readFileSync(
      new URL("./fixtures/flows-catalog.json", import.meta.url),
      "utf8",
    ),
  ) as {
    nodes: { type: string; design?: boolean; outputs: { id: string }[] }[];
  };
  const column = catalog.nodes.find((n) => n.type === "recipe.column");
  const generate = catalog.nodes.find((n) => n.type === "recipe.generate");
  assert.equal(column?.design, true);
  assert.deepEqual(
    generate?.outputs.map((p) => p.id),
    ["dataset", "preview"],
  );
});

test("deleting a column from the inspector or Ctrl+X takes its {{ reference }} out of its readers", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { removeNodes } = await import("../src/features/flows/recipe/recipe-actions.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  useDocStore.getState().load({ ...doc(nodes, referenceEdges(nodes)), groups: [group] });
  const prompt = byName(nodes, "sql_prompt");
  removeNodes([prompt.id]);
  const graph = useDocStore.getState().graph;
  assert.equal(graph.nodes[prompt.id], undefined);
  const sql = graph.nodes[byName(nodes, "sql").id];
  assert.doesNotMatch(JSON.stringify(sql.params.column), /sql_prompt/);
  assert.ok(Object.values(graph.edges).every((e) => e.source !== prompt.id));
});
