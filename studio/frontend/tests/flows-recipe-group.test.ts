// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { applyConfigPatch, clientErrors, commitStudioEdit, studioView } =
  await import(
  "../src/features/flows/recipe/recipe-adapter.ts"
);
const {
  COLUMN_TYPE,
  GENERATE_TYPE,
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
  assert.deepEqual(edit.group?.config?.mcp_providers, []);
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
    "Category",
  );
  assert.equal(
    columnSummary(byName(nodes, "sql").params.column as Record<string, unknown>)
      .badge,
    "AI code",
  );
  assert.equal(
    columnSummary(
      byName(nodes, "sql-validator").params.column as Record<string, unknown>,
    ).badge,
    "SQL check",
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

async function loadCatalog() {
  const { useCatalogStore } = await import("../src/features/flows/stores/catalog-store.ts");
  const catalog = JSON.parse(
    readFileSync(new URL("./fixtures/flows-catalog.json", import.meta.url), "utf8"),
  ) as { nodes: { type: string }[] };
  useCatalogStore.setState({
    specs: Object.fromEntries(catalog.nodes.map((s) => [s.type, s])) as never,
  });
}

test("a Recipe Dataset dropped into a recipe frame joins it, and a second one is refused", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { addNodeConnected } = await import("../src/features/flows/editor/node-factory.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  await loadCatalog();
  const { group, nodes } = sqlGroup();
  const frame = { ...group, rect: [0, 0, 2000, 600] as [number, number, number, number] };
  useDocStore.getState().load({ ...doc(nodes), groups: [frame] });
  const added = addNodeConnected(GENERATE_TYPE, [100, 400]);
  assert.equal(added?.group, "recipe");
  assert.equal(useDocStore.getState().graph.nodes[added?.id ?? ""]?.group, "recipe");
  assert.equal(addNodeConnected(GENERATE_TYPE, [200, 400]), null);
});

test("Quick Add of a Recipe Dataset from a dataset output wires that dataset into its seed", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { addNodeConnected } = await import("../src/features/flows/editor/node-factory.ts");
  const { outHandle } = await import("../src/features/flows/editor/connection-rules.ts");
  const { doc, node } = await import("./flows-editor-fixtures.ts");
  await loadCatalog();
  useDocStore.getState().load(doc([node("hf", "data.hf_dataset")]));
  const generate = addNodeConnected(
    GENERATE_TYPE,
    [600, 0],
    { nodeId: "hf", handleId: outHandle("dataset"), side: "out", kind: "data", port: null },
    "seed",
  );
  assert.ok(generate);
  const edges = Object.values(useDocStore.getState().graph.edges);
  assert.ok(
    edges.some((e) => e.source === "hf" && e.target === generate.id && e.target_port === "seed"),
  );
});

test("every Recipe Studio column can be added, and its card names the block it is", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { addRecipeBlock } = await import("../src/features/flows/recipe/recipe-blocks.ts");
  const { COLUMN_BLOCKS, columnBlock } = await import(
    "../src/features/flows/recipe/recipe-kinds.ts"
  );
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  assert.deepEqual(
    COLUMN_BLOCKS.filter((b) => b.kind === "validator").map((b) => b.label),
    ["Python check", "SQL check", "JS/TS check", "JSON check", "Markdown check"],
  );
  for (const block of COLUMN_BLOCKS) {
    useDocStore.getState().load({ ...doc(nodes), groups: [group] });
    addRecipeBlock("recipe", block.kind, block.type);
    const graph = useDocStore.getState().graph;
    const added = graph.nodeOrder
      .map((id) => graph.nodes[id])
      .find((n) => n.type === COLUMN_TYPE && !nodes.some((m) => m.id === n.id));
    assert.ok(added, block.type);
    assert.equal(added.group, "recipe");
    assert.equal(
      columnBlock(added.params.column as Record<string, unknown>).type,
      block.type,
      block.label,
    );
  }
});

test("a seed, model preset and provider show as cards; a shared file seed says to load it first", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { recipeSetup } = await import("../src/features/flows/recipe/recipe-setup.ts");
  const { missingSeed } = await import(
    "../src/features/flows/recipe/recipe-group-actions.ts"
  );
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  const shared = {
    ...group,
    config: {
      ...group.config,
      // biome-ignore lint/style/useNamingConvention: api schema
      seed_config: { source: { seed_type: "local", path: "" } },
    },
  };
  useDocStore.getState().load({ ...doc(nodes), groups: [shared] });
  const { cards, links } = recipeSetup(useDocStore.getState().graph, shared);
  assert.deepEqual(
    cards.map((c) => c.kind),
    ["seed", "model_provider", "model_config"],
  );
  assert.equal(cards[0].warning, "Load the seed first");
  assert.match(missingSeed(shared) ?? "", /^Load the seed first/);
  assert.equal(missingSeed(group), null);
  const sql = byName(nodes, "sql");
  assert.ok(links.some((l) => l.source === cards[2].id && l.target === sql.id));
});

test("a starter recipe lands in free space, not on top of what is there", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { insertStarterRecipe } = await import(
    "../src/features/flows/recipe/recipe-actions.ts"
  );
  const { doc, node } = await import("./flows-editor-fixtures.ts");
  useDocStore.getState().load(doc([{ ...node("a", "input.text"), pos: [0, 0] }]));
  const generate = insertStarterRecipe([10, 10], { rows: 5 });
  const graph = useDocStore.getState().graph;
  const group = graph.groups[generate.group as string];
  assert.ok(group.rect[1] > 100, `frame at ${group.rect}`);
  useDocStore.getState().load(doc([]));
  const alone = insertStarterRecipe([10, 10], { rows: 5 });
  assert.deepEqual(useDocStore.getState().graph.groups[alone.group as string].rect.slice(0, 2), [-30, -50]);
});

test("a tool profile added without a server yet stays as a card and can be removed", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { addRecipeBlock, removeSetup } = await import(
    "../src/features/flows/recipe/recipe-blocks.ts"
  );
  const { recipeSetup } = await import("../src/features/flows/recipe/recipe-setup.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  useDocStore.getState().load({ ...doc(nodes), groups: [group] });
  addRecipeBlock("recipe", "llm", "tool_config");
  const graph = useDocStore.getState().graph;
  const tools = recipeSetup(graph, graph.groups.recipe).cards.filter(
    (c) => c.kind === "tool_config",
  );
  assert.equal(tools.length, 1);
  assert.equal(tools[0].warning, "Add a tool server");
  removeSetup({ groupId: "recipe", kind: "tool_config", name: tools[0].name });
  const after = useDocStore.getState().graph;
  assert.equal(
    recipeSetup(after, after.groups.recipe).cards.some((c) => c.kind === "tool_config"),
    false,
  );
});

test("Check flags an empty endpoint and model the way Recipe Studio does, before the server", () => {
  const { group, nodes } = sqlGroup();
  const config = group.config as {
    model_providers: Record<string, unknown>[];
    model_configs: Record<string, unknown>[];
  };
  // The learning recipe ships both blank; Recipe Studio refuses it until they are set.
  config.model_providers[0].endpoint = "";
  config.model_configs[0].model = "";
  const errors = clientErrors(group, nodes);
  assert.ok(errors.some((e) => /endpoint is required/.test(e)), errors.join("; "));
  assert.ok(errors.some((e) => /model is required/.test(e)), errors.join("; "));
});

test("Check leaves disabled columns and a disabled Recipe Dataset out, like the compiler", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { runColumns } = await import(
    "../src/features/flows/recipe/recipe-group-actions.ts"
  );
  const { groupGenerate } = await import("../src/features/flows/recipe/recipe-columns.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  (group.config as { model_providers: Record<string, unknown>[] }).model_providers[0].endpoint =
    "http://localhost:8000/v1";
  // The only columns using the preset with no model are switched off.
  const off = new Set(["sql_prompt", "sql", "sql-validator"]);
  const columns = nodes.map((n) =>
    off.has(columnName(n)) ? { ...n, disabled: true } : n,
  );
  const step = (id: string, disabled: boolean): FlowNode => ({
    id,
    type: GENERATE_TYPE,
    v: 1,
    params: { rows: 5, mode: "preview" },
    pos: [0, 400],
    group: "recipe",
    disabled,
  });
  useDocStore.getState().load({
    ...doc([...columns, step("old_step", true), step("live_step", false)]),
    groups: [group],
  });
  const graph = useDocStore.getState().graph;
  assert.equal(groupGenerate(graph, "recipe")?.id, "live_step");
  const checked = runColumns(graph, "recipe");
  assert.ok(checked.every((n) => !off.has(columnName(n))));
  assert.deepEqual(clientErrors(group, checked), []);
  assert.ok(clientErrors(group, columns).some((e) => /model is required/.test(e)));
});

test("a Hub seed still named by its repo id is picked out for resolving, a reader path is not", async () => {
  const { bareHubSeed, hubSeedInspect } = await import(
    "../src/features/flows/recipe/recipe-group-actions.ts"
  );
  const seed = (path: string, type = "hf") => ({
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_config: { source: { seed_type: type, path } },
  });
  assert.ok(bareHubSeed(seed("unsloth/alpaca-cleaned")));
  assert.equal(bareHubSeed(seed("datasets/unsloth/alpaca-cleaned/**/*.parquet")), null);
  assert.equal(bareHubSeed(seed("unsloth/alpaca-cleaned", "local")), null);
  assert.equal(bareHubSeed(seed("")), null);
  assert.equal(bareHubSeed({}), null);
  // Check asks with what the run sends, token included, so a gated seed checks like it runs.
  const gated = bareHubSeed({
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_config: { source: { seed_type: "hf", path: "org/gated", token: "hf_x", subset: "a" } },
  });
  assert.deepEqual(hubSeedInspect(gated ?? {}), {
    // biome-ignore lint/style/useNamingConvention: api schema
    dataset_name: "org/gated",
    split: "train",
    subset: "a",
    // biome-ignore lint/style/useNamingConvention: api schema
    hf_token: "hf_x",
    // biome-ignore lint/style/useNamingConvention: api schema
    preview_size: 1,
  });
});

test("setup links are the flow's own edge type, dashed as references, so they follow the wire style", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { recipeSetup, setupEdges, REFERENCE_EDGE_CLASS } = await import(
    "../src/features/flows/recipe/recipe-setup.ts"
  );
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  const sql = byName(nodes, "sql");
  const folded = nodes.map((n) => (n.id === sql.id ? { ...n, collapsed: true } : n));
  useDocStore.getState().load({ ...doc(folded), groups: [group] });
  const graph = useDocStore.getState().graph;
  const { links } = recipeSetup(graph, group);
  const edges = setupEdges(links, graph, { source: "out", target: "in" });
  assert.ok(edges.length > 0);
  for (const edge of edges) {
    // flowEdge: Curved or bus-routed with every other wire; the class keeps it a dashed reference.
    assert.equal(edge.type, "flowEdge");
    assert.equal(edge.className, REFERENCE_EDGE_CLASS);
    assert.deepEqual([edge.sourceHandle, edge.targetHandle], ["out", "in"]);
    assert.equal(edge.selectable, false);
    assert.equal(edge.deletable, false);
  }
  assert.ok(!edges.some((e) => e.target === sql.id), "a collapsed column draws no link");
  const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
  assert.match(css, new RegExp(`\\.${REFERENCE_EDGE_CLASS} \\.react-flow__edge-path[^}]*stroke-dasharray`));
});

test("a note keeps the size it was given whatever its text; long text scrolls inside", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  useDocStore.getState().load({
    ...doc([]),
    notes: [{ id: "n1", text: "a long note", rect: [0, 0, 320, 200] }],
  });
  const note = new RfDeriver()
    .nodes(useDocStore.getState().graph, new Set(), null)
    .find((n) => n.type === "flowNote");
  assert.deepEqual(note?.style, { width: 320, height: 200 });
  const frames = readSrc("features/flows/editor/frame-nodes.tsx");
  assert.match(frames, /fieldSizing="fixed"/);
  assert.match(frames, /<BaseNodeContent className="nowheel h-full overflow-y-auto px-3 py-2">/);
});

async function recipeOverlaps(sizes: Map<string, { width: number; height: number }>) {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { setupLayout, NODE_SIZE, CARD_SIZE } = await import(
    "../src/features/flows/recipe/recipe-setup.ts"
  );
  const graph = useDocStore.getState().graph;
  const boxes = [
    ...graph.nodeOrder.map((id) => ({
      id,
      x: graph.nodes[id].pos[0],
      y: graph.nodes[id].pos[1],
      ...(sizes.get(id) ?? NODE_SIZE),
    })),
    ...setupLayout(graph, null, sizes).elements.map(({ card, x, y }) => ({
      id: card.id,
      x,
      y,
      ...(sizes.get(card.id) ?? CARD_SIZE),
    })),
  ];
  const hits: string[] = [];
  for (const [i, a] of boxes.entries())
    for (const b of boxes.slice(i + 1))
      if (a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height)
        hits.push(`${a.id} x ${b.id}`);
  return hits;
}

test("no two nodes of a recipe group overlap after import, after adding setup blocks, or after Tidy up", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { addRecipeBlock } = await import("../src/features/flows/recipe/recipe-blocks.ts");
  const { setupLayoutNodes, withLaidOutSetup, isSetupId, withSetupPosition, setupLayout } =
    await import("../src/features/flows/recipe/recipe-setup.ts");
  const { computeLayout } = await import("../src/features/flows/editor/layout-core.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  // As the importer lays it out: a 320 x 180 grid, the Recipe Dataset step at the end of row 0,
  // plus a column left in the step's lane from an older layout (the ticket-router report).
  const placed = nodes.map((n, i) => ({ ...n, pos: [(i % 4) * 320, Math.floor(i / 4) * 180] as [number, number] }));
  const step: FlowNode = {
    id: "step",
    type: GENERATE_TYPE,
    v: 1,
    params: { rows: 5, mode: "preview" },
    pos: [1280, 0],
    group: "recipe",
  };
  const stray = { ...placed[0], id: "stray", pos: [1280, 460] as [number, number] };
  stray.params = { column: { ...(placed[0].params.column as object), name: "stray" } };
  useDocStore.getState().load({
    ...doc([...placed, step, stray]),
    groups: [{ ...group, rect: [-40, -60, 1600, 700] }],
  });
  // A finished run makes the step tall (Done, a dataset table).
  const sizes = new Map([["step", { width: 260, height: 420 }]]);
  assert.deepEqual(await recipeOverlaps(sizes), [], "after import");

  const before = setupLayout(useDocStore.getState().graph, null, sizes).elements.length;
  for (const type of ["model_provider", "model_config", "tool_config"])
    addRecipeBlock("recipe", "llm", type);
  assert.equal(setupLayout(useDocStore.getState().graph, null, sizes).elements.length, before + 3);
  assert.deepEqual(await recipeOverlaps(sizes), [], "after adding setup blocks");

  // A card dragged into the column grid: the next column goes around it.
  const first = setupLayout(useDocStore.getState().graph, null, sizes).elements[0];
  const g = useDocStore.getState().graph.groups.recipe;
  useDocStore.getState().upsertGroup(withSetupPosition(g, first.card.id, [320, 360]));
  const columns = useDocStore.getState().graph.nodeOrder.length;
  addRecipeBlock("recipe", "sampler", "uuid");
  assert.equal(useDocStore.getState().graph.nodeOrder.length, columns + 1);
  assert.deepEqual(await recipeOverlaps(sizes), [], "a new column keeps clear of a moved card");

  const graph = useDocStore.getState().graph;
  const setup = setupLayoutNodes(graph, (id) => sizes.get(id));
  const result = computeLayout({
    direction: "LR",
    nodes: [
      ...graph.nodeOrder.map((id) => ({ id, ...(sizes.get(id) ?? { width: 260, height: 140 }) })),
      ...setup.nodes,
    ],
    edges: [
      ...graph.edgeOrder.map((id) => ({ source: graph.edges[id].source, target: graph.edges[id].target })),
      ...setup.edges,
    ],
  });
  assert.ok(Object.keys(result).some(isSetupId), "Tidy up lays out the setup cards too");
  useDocStore.getState().moveNodes(
    Object.fromEntries(Object.entries(result).filter(([k]) => !isSetupId(k))),
  );
  useDocStore.getState().patchGraph({ groups: withLaidOutSetup(useDocStore.getState().graph, result) });
  assert.deepEqual(await recipeOverlaps(sizes), [], "after Tidy up");
});

test("a starter recipe and a new column both keep clear of setup cards, even dragged ones", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { freeSpot, nextColumnSlot } = await import("../src/features/flows/recipe/recipe-actions.ts");
  const { withSetupPosition, setupLayout, CARD_SIZE } = await import(
    "../src/features/flows/recipe/recipe-setup.ts"
  );
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  // Every cell of a two-row frame is full, so a new column needs a new row.
  const extra = { ...nodes[0], id: "extra", params: { column: { ...(nodes[0].params.column as object), name: "extra" } } };
  const placed = [...nodes, extra].map((n, i) => ({ ...n, pos: [(i % 4) * 320, Math.floor(i / 4) * 180] as [number, number] }));
  assert.equal(placed.length, 8);
  const frame = { ...group, rect: [-40, -60, 1280, 400] as [number, number, number, number] };
  const card = setupLayout({ ...useDocStore.getState().graph, groups: {}, groupOrder: [] }, null, new Map());
  assert.equal(card.elements.length, 0);
  useDocStore.getState().load({ ...doc(placed), groups: [frame] });
  const first = setupLayout(useDocStore.getState().graph, null, new Map()).elements[0];
  // Dragged to where the next row would start, and to the empty canvas below the frame.
  const g = useDocStore.getState().graph.groups.recipe;
  useDocStore.getState().upsertGroup(withSetupPosition(g, first.card.id, [0, 360]));
  const { pos } = nextColumnSlot(useDocStore.getState().graph, "recipe");
  const hit = (p: [number, number], at: { x: number; y: number }) =>
    p[0] < at.x + CARD_SIZE.width && at.x < p[0] + 280 && p[1] < at.y + CARD_SIZE.height && at.y < p[1] + 140;
  const moved = setupLayout(useDocStore.getState().graph, null, new Map()).elements.find(
    (e) => e.card.id === first.card.id,
  );
  assert.ok(moved);
  assert.ok(!hit(pos, moved), `column at ${pos} sits on the card at ${moved.x},${moved.y}`);
  useDocStore.getState().upsertGroup(
    withSetupPosition(useDocStore.getState().graph.groups.recipe, first.card.id, [0, 900]),
  );
  const below = setupLayout(useDocStore.getState().graph, null, new Map()).elements.find(
    (e) => e.card.id === first.card.id,
  );
  assert.ok(below);
  const spot = freeSpot(useDocStore.getState().graph, [below.x + 40, below.y + 60], [600, 400]);
  assert.notDeepEqual(spot, [below.x + 40, below.y + 60], "the starter does not land on the card");
});

test("the run dialogs' check leaves disabled columns out, like the frame's Check", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { checkRecipeGroupDetail } = await import(
    "../src/features/flows/recipe/recipe-group-actions.ts"
  );
  const editor = readFileSync(
    new URL("../src/features/flows/recipe/recipe-editor.tsx", import.meta.url),
    "utf8",
  );
  assert.match(editor, /const checkGroup = checkRecipeGroupDetail;/);
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  (group.config as { model_providers: Record<string, unknown>[] }).model_providers[0].endpoint =
    "http://localhost:8000/v1";
  const off = new Set(["sql_prompt", "sql", "sql-validator"]);
  useDocStore.getState().load({
    ...doc(nodes.map((n) => (off.has(columnName(n)) ? { ...n, disabled: true } : n))),
    groups: [group],
  });
  // Past the client checks (the blank-model preset is only used by disabled columns); the server
  // half cannot load under node, which is fine: the point is the client half did not stop it.
  const result = await checkRecipeGroupDetail("recipe");
  assert.ok(!result.errors.some((e) => /model is required/.test(e)), result.errors.join("; "));
});
