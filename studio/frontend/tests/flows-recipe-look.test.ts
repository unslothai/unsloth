// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  COLUMN_TYPE,
  columnName,
  columnOf,
  referenceEdges,
  withoutReference,
} =
  await import(
  "../src/features/flows/recipe/recipe-columns.ts"
);
const {
  auxId,
  columnInputPorts,
  columnLook,
  columnTargetPort,
  recipeAux,
} = await import("../src/features/flows/recipe/recipe-look.ts");
const { auxLayout, withAuxPosition, AUX_UI_KEY } = await import(
  "../src/features/flows/recipe/recipe-aux.ts"
);
const { noteRecipeGroup, withRecipeNotes } = await import(
  "../src/features/flows/recipe/recipe-notes.ts"
);
const { commitStudioEdit, studioView } = await import(
  "../src/features/flows/recipe/recipe-adapter.ts"
);
const { parseHandle } = await import(
  "../src/features/flows/editor/connection-rules.ts"
);
const { emptyGraph } = await import("../src/features/flows/stores/doc-store.ts");

type FlowNode = Parameters<typeof columnName>[0];
type Graph = ReturnType<typeof emptyGraph>;

function load(name: string) {
  return JSON.parse(
    readFileSync(
      new URL(
        `../src/features/data-recipes/learning-recipes/${name}.json`,
        import.meta.url,
      ),
      "utf8",
    ),
  ) as {
    recipe: Record<string, unknown> & { columns: Record<string, unknown>[] };
  };
}

/** A flow graph holding one learning recipe as a group, the way the importer lays it out. */
function recipeGraph(name: string) {
  const { columns, ...config } = structuredClone(load(name).recipe);
  const graph: Graph = emptyGraph();
  const group = {
    id: "recipe",
    label: name,
    rect: [0, 0, 2000, 300] as [number, number, number, number],
    kind: "recipe" as const,
    config,
    run: { rows: 5 },
  };
  graph.groups[group.id] = group;
  graph.groupOrder.push(group.id);
  const nodes: FlowNode[] = columns.map((column, i) => ({
    id: `recipe_col_${i + 1}`,
    type: COLUMN_TYPE,
    v: 1,
    params: { column },
    pos: [i * 320, 0],
    label: String(column.name),
    group: "recipe",
  }));
  nodes.push({
    id: "recipe_generate",
    type: "recipe.generate",
    v: 1,
    params: {},
    pos: [nodes.length * 320, 40],
    group: "recipe",
  });
  for (const node of nodes) {
    graph.nodes[node.id] = node;
    graph.nodeOrder.push(node.id);
  }
  for (const edge of referenceEdges(nodes.filter((n) => n.type === COLUMN_TYPE))) {
    graph.edges[edge.id] = edge;
    graph.edgeOrder.push(edge.id);
  }
  return { graph, group, nodes };
}

function byName(graph: Graph, name: string): FlowNode {
  const found = Object.values(graph.nodes).find((n) => columnName(n) === name);
  assert.ok(found, name);
  return found;
}

test("each column type gets Recipe Studio's identity and its key fields", () => {
  const sql = recipeGraph("text-to-sql").graph;
  const domain = columnLook(columnOf(byName(sql, "domain")));
  assert.equal(domain.kind, "sampler");
  assert.equal(domain.subtitle, "Category · Generated field");
  assert.deepEqual(domain.chips, ["Ecommerce", "Customer Support", "Finance"]);
  assert.equal(domain.drop, true);
  const topic = columnLook(columnOf(byName(sql, "topic")));
  assert.equal(topic.subtitle, "Subcategory · Generated field");
  assert.deepEqual(topic.fields, [{ label: "Based on", value: "domain" }]);
  const prompt = columnLook(columnOf(byName(sql, "sql_prompt")));
  assert.equal(prompt.subtitle, "AI text · AI step");
  assert.deepEqual(prompt.fields, [{ label: "Model", value: "sql-pro" }]);
  assert.deepEqual(
    prompt.details.map((d) => d.label),
    ["Prompt", "System"],
  );
  assert.equal(prompt.drop, false);
  const code = columnLook(columnOf(byName(sql, "sql")));
  assert.equal(code.subtitle, "AI code · AI step");
  assert.deepEqual(code.fields[1], { label: "Language", value: "sql:ansi" });
  const check = columnLook(columnOf(byName(sql, "sql-validator")));
  assert.equal(check.kind, "validator");
  assert.equal(check.subtitle, "SQL · Check");
  assert.deepEqual(check.fields[0], { label: "Checks", value: "sql" });

  const py = recipeGraph("text-to-python").graph;
  const judge = columnLook(columnOf(byName(py, "code_judge_result")));
  assert.equal(judge.subtitle, "AI scorer · AI step");
  assert.ok(judge.chips.length > 0, "judge scores show as chips");
  assert.equal(judge.valueType, "json");

  const st = recipeGraph("structured-outputs-jinja").graph;
  const user = columnLook(columnOf(byName(st, "user")));
  assert.equal(user.subtitle, "Synthetic person · Generated field");
  assert.notEqual(user.accent, domain.accent);
  const formula = columnLook(columnOf(byName(st, "user_full_name")));
  assert.equal(formula.subtitle, "Text · Formula");
  assert.equal(formula.details[0].label, "Formula");
  assert.equal(
    columnLook(columnOf(byName(st, "ticket"))).subtitle,
    "AI structured data · AI step",
  );
});

test("a column has one input port per column it reads, named after it, in mention order", () => {
  const { graph } = recipeGraph("text-to-sql");
  const prompt = byName(graph, "sql_prompt");
  const ports = columnInputPorts(graph, prompt.id);
  assert.deepEqual(
    ports.map((p) => p.label),
    ["domain", "topic", "sql_task_type", "instruction_phrase"],
  );
  assert.equal(ports[0].port, `in@${byName(graph, "domain").id}`);
  assert.equal(ports[0].type, "text");
  assert.deepEqual(
    columnInputPorts(graph, byName(graph, "domain").id),
    [],
    "a category sampler reads nothing",
  );
  assert.deepEqual(
    columnInputPorts(graph, byName(graph, "topic").id).map((p) => p.label),
    ["domain"],
    "a subcategory reads its parent",
  );
});

test("a reference wire lands on the reading column's port for its source, and parses back to `in`", () => {
  const { graph } = recipeGraph("text-to-sql");
  const sql = byName(graph, "sql");
  const edge = Object.values(graph.edges).find((e) => e.target === sql.id);
  assert.ok(edge);
  const port = columnTargetPort(edge, graph);
  assert.equal(port, `in@${byName(graph, "sql_prompt").id}`);
  assert.deepEqual(parseHandle(`i:${port}`), {
    kind: "data",
    side: "in",
    port: "in",
  });
  const plain = { ...edge, target: "nowhere" };
  assert.equal(columnTargetPort(plain, graph), "in");
});

test("ports follow the references both ways: drop {{ ref }} and the port goes", () => {
  const { graph } = recipeGraph("text-to-sql");
  const prompt = byName(graph, "sql_prompt");
  graph.nodes[prompt.id] = {
    ...prompt,
    params: { column: withoutReference(columnOf(prompt), "domain") },
  };
  const columns = graph.nodeOrder
    .map((id) => graph.nodes[id])
    .filter((n) => n.type === COLUMN_TYPE);
  graph.edges = Object.fromEntries(referenceEdges(columns).map((e) => [e.id, e]));
  graph.edgeOrder = Object.keys(graph.edges);
  assert.deepEqual(
    columnInputPorts(graph, prompt.id).map((p) => p.label),
    ["topic", "sql_task_type", "instruction_phrase"],
  );
});

test("providers, presets and tool profiles become cards wired to the columns that use them", () => {
  const { graph, group } = recipeGraph("text-to-sql");
  const { cards, links } = recipeAux(graph, group);
  assert.deepEqual(
    cards.map((c) => [c.kind, c.name]),
    [
      ["model_provider", "vllm"],
      ["model_config", "sql-pro"],
    ],
  );
  const preset = auxId("recipe", "model_config", "sql-pro");
  assert.deepEqual(
    links.map((l) => [l.source, l.target]),
    [
      [auxId("recipe", "model_provider", "vllm"), preset],
      [preset, byName(graph, "sql_prompt").id],
      [preset, byName(graph, "sql").id],
    ],
  );
  const withTools = {
    ...group,
    config: {
      ...group.config,
      // biome-ignore lint/style/useNamingConvention: api schema
      tool_configs: [{ tool_alias: "web", providers: ["search"] }],
    },
  };
  graph.nodes[byName(graph, "sql").id].params.column = {
    ...(byName(graph, "sql").params.column as object),
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_alias: "web",
  };
  const tools = recipeAux(graph, withTools);
  const tool = tools.cards.find((c) => c.kind === "tool_config");
  assert.equal(tool?.subtitle, "1 server · Tool access");
  assert.ok(
    tools.links.some(
      (l) => l.source === tool?.id && l.target === byName(graph, "sql").id,
    ),
  );
});

test("cards stack under Recipe Dataset, grow the frame, and keep a dragged spot relative to it", () => {
  const { graph, group } = recipeGraph("text-to-python");
  const sizes = new Map([["recipe_generate", { width: 260, height: 100 }]]);
  const first = auxLayout(graph, null, sizes);
  assert.ok(first.elements.length >= 2);
  const generate = graph.nodes.recipe_generate;
  assert.equal(first.elements[0].x, generate.pos[0]);
  assert.equal(first.elements[0].y, generate.pos[1] + 100 + 24);
  assert.ok(first.elements[1].y > first.elements[0].y);
  assert.ok(first.heights.recipe > group.rect[3], "frame grows to hold the cards");

  const card = first.elements[1].card;
  const moved = withAuxPosition(group, card.id, [500, 220]);
  graph.groups.recipe = moved;
  const again = auxLayout(graph, null, sizes);
  const placed = again.elements.find((e) => e.card.id === card.id);
  assert.deepEqual([placed?.x, placed?.y], [500, 220]);
  graph.groups.recipe = { ...moved, rect: [100, 50, 2000, 300] };
  const shifted = auxLayout(graph, null, sizes).elements.find(
    (e) => e.card.id === card.id,
  );
  assert.deepEqual([shifted?.x, shifted?.y], [600, 270], "moves with its frame");
  assert.ok(moved.ui?.[AUX_UI_KEY]);
});

test("editing a recipe note writes the markdown back into the recipe's ui notes", () => {
  const { graph, group, nodes } = recipeGraph("text-to-sql");
  graph.notes.recipe_note_1 = {
    id: "recipe_note_1",
    text: "##### Edited\n\n- `domain`",
    rect: [0, 0, 320, 200],
    color: "#DBEAFE",
  };
  graph.notes.other = { id: "other", text: "mine", rect: [0, 0, 100, 100] };
  graph.noteOrder.push("recipe_note_1", "other");
  assert.equal(noteRecipeGroup(graph, "recipe_note_1")?.id, "recipe");
  assert.equal(noteRecipeGroup(graph, "other"), null);
  const prior = {
    ...group,
    ui: {
      nodes: [{ id: "note_1", name: "note_1", note_opacity: "35", markdown: "old" }],
      // biome-ignore lint/style/useNamingConvention: api schema
      seed_columns: ["x"],
    },
  };
  const next = withRecipeNotes(prior, Object.values(graph.notes));
  assert.deepEqual(next.ui?.nodes, [
    {
      id: "note_1",
      name: "note_1",
      // biome-ignore lint/style/useNamingConvention: api schema
      note_opacity: "35",
      // biome-ignore lint/style/useNamingConvention: api schema
      node_type: "markdown_note",
      markdown: "##### Edited\n\n- `domain`",
      // biome-ignore lint/style/useNamingConvention: api schema
      note_color: "#DBEAFE",
    },
  ]);
  assert.deepEqual(next.ui?.seed_columns, ["x"], "other ui hints stay");

  const columns = nodes.filter((n) => n.type === COLUMN_TYPE);
  const view = studioView(next, columns);
  assert.deepEqual(view.errors, [], "the recipe still opens in Recipe Studio");
  const note = Object.values(view.configs).find((c) => c.kind === "markdown_note");
  assert.equal(
    (note as { markdown?: string } | undefined)?.markdown,
    "##### Edited\n\n- `domain`",
  );
  const edit = commitStudioEdit(next, columns, view, view.configs);
  assert.equal(edit.group, null, "an unrelated edit leaves the notes alone");
});

test("a deleted recipe note leaves the recipe's ui notes", () => {
  const { group } = recipeGraph("text-to-sql");
  const notes = [
    { id: "recipe_note_1", text: "a", rect: [0, 0, 1, 1] as [number, number, number, number] },
    { id: "recipe_note_2", text: "b", rect: [0, 0, 1, 1] as [number, number, number, number] },
  ];
  const both = withRecipeNotes(group, notes);
  const one = withRecipeNotes(both, notes.slice(1));
  assert.deepEqual(
    (one.ui?.nodes as { id: string }[]).map((n) => n.id),
    ["note_2"],
  );
});

test("a reference wire is recoloured when its source column changes value type, and reused when not", async () => {
  const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");
  const { graph } = recipeGraph("text-to-python");
  const deriver = new RfDeriver();
  const portType = (edge: { source: string }) =>
    columnLook(columnOf(graph.nodes[edge.source])).valueType;
  const first = deriver.edges(graph, new Set(), portType);
  const domain = byName(graph, "domain");
  const wire = first.find((e) => e.source === domain.id);
  assert.equal(wire?.data?.portType, "text");
  assert.equal(wire?.targetHandle, `i:in@${domain.id}`);
  graph.nodes[domain.id] = { ...domain, pos: [5, 5] };
  const moved = deriver.edges(graph, new Set(), portType);
  assert.equal(moved.find((e) => e.id === wire?.id), wire, "same colour, same edge object");
  graph.nodes[domain.id] = {
    ...domain,
    params: {
      column: {
        ...columnOf(domain),
        // biome-ignore lint/style/useNamingConvention: api schema
        sampler_type: "uniform",
        params: { low: 0, high: 1 },
      },
    },
  };
  const recoloured = deriver.edges(graph, new Set(), portType);
  assert.equal(recoloured.find((e) => e.id === wire?.id)?.data?.portType, "number");
});
