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
    reroutes: {},
    rerouteOrder: [],
    wires: {},
    wireOrder: [],
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
  // One Undo brings the column back with the references to it.
  useDocStore.getState().undo();
  const back = useDocStore.getState().graph;
  assert.ok(back.nodes[prompt.id]);
  assert.match(JSON.stringify(back.nodes[sql.id].params.column), /sql_prompt/);
});

test("deleting a column on the canvas is one undo step with its references", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { removeElementsWithReferences } = await import(
    "../src/features/flows/recipe/recipe-actions.ts"
  );
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  useDocStore.getState().load({ ...doc(nodes, referenceEdges(nodes)), groups: [group] });
  const prompt = byName(nodes, "sql_prompt");
  const sql = byName(nodes, "sql");
  const graph = useDocStore.getState().graph;
  // What the canvas's Delete key hands over: the node and its wires.
  const edges = graph.edgeOrder.filter(
    (id) => graph.edges[id].source === prompt.id || graph.edges[id].target === prompt.id,
  );
  removeElementsWithReferences({ nodes: [prompt.id], edges, groups: [], notes: [] });
  const after = useDocStore.getState().graph;
  assert.equal(after.nodes[prompt.id], undefined);
  assert.doesNotMatch(JSON.stringify(after.nodes[sql.id].params.column), /sql_prompt/);
  useDocStore.getState().undo();
  const back = useDocStore.getState().graph;
  assert.ok(back.nodes[prompt.id]);
  assert.match(JSON.stringify(back.nodes[sql.id].params.column), /sql_prompt/);
});

test("deleting reference wires from two recipe frames at once clears both frames' wires", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { disconnectColumns } = await import("../src/features/flows/recipe/recipe-actions.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const a = sqlGroup();
  const b = sqlGroup();
  const other = b.nodes.map((n) => ({ ...n, id: `b_${n.id}`, group: "recipe_b" }));
  const edges = [...referenceEdges(a.nodes), ...referenceEdges(other)];
  useDocStore.getState().load({
    ...doc([...a.nodes, ...other], edges),
    groups: [a.group, { ...b.group, id: "recipe_b" }],
  });
  const graph = useDocStore.getState().graph;
  const wire = (nodes: FlowNode[]) => {
    const target = byName(nodes, "sql").id;
    const id = graph.edgeOrder.find(
      (e) => graph.edges[e].target === target && graph.edges[e].source === byName(nodes, "sql_prompt").id,
    );
    assert.ok(id);
    return id;
  };
  const picked = [wire(a.nodes), wire(other)];
  assert.deepEqual(disconnectColumns(picked).sort(), [...picked].sort());
  const after = useDocStore.getState().graph;
  for (const id of picked) assert.equal(after.edges[id], undefined, id);
  for (const nodes of [a.nodes, other])
    assert.doesNotMatch(JSON.stringify(after.nodes[byName(nodes, "sql").id].params.column), /sql_prompt/);
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

test("a loaded document seed keeps its files when the group opens in Recipe Studio's forms", () => {
  const { group, nodes } = sqlGroup();
  const withDocs = {
    ...group,
    config: {
      ...group.config,
      // biome-ignore lint/style/useNamingConvention: api schema
      seed_config: { source: { seed_type: "unstructured", paths: ["/up/x/a.pdf"] } },
    },
  };
  const view = studioView(withDocs, nodes);
  const seed = Object.values(view.configs).find((c) => c.kind === "seed") as
    | { resolved_paths?: string[] }
    | undefined;
  assert.deepEqual(seed?.resolved_paths, ["/up/x/a.pdf"]);
});

test("a wired seed is what the check goes by, not the group's own empty file seed", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { checkRecipeGroupDetail } = await import(
    "../src/features/flows/recipe/recipe-group-actions.ts"
  );
  const { doc, node, edge } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  (group.config as { model_providers: Record<string, unknown>[] }).model_providers[0].endpoint =
    "http://localhost:8000/v1";
  for (const mc of (group.config as { model_configs: Record<string, unknown>[] }).model_configs)
    mc.model ||= "qwen";
  // biome-ignore lint/style/useNamingConvention: api schema
  const shared = { ...group, config: { ...group.config, seed_config: { source: { seed_type: "unstructured", paths: [] } } } };
  const step = { ...node("step", GENERATE_TYPE, [0, 300]), group: "recipe" };
  const data = node("data", "data.hf_dataset", [0, 600]);
  const load = (wired: boolean) =>
    useDocStore.getState().load({
      ...doc([...nodes, step, data], wired ? [edge("seed", "data", "dataset", "step", "seed")] : []),
      groups: [shared],
    });
  load(false);
  assert.ok((await checkRecipeGroupDetail("recipe")).errors.some((e) => /source-data preview/.test(e)));
  load(true);
  const wired = await checkRecipeGroupDetail("recipe");
  assert.deepEqual(wired, { valid: true, errors: [], rawDetail: null });
});

test("the schema transform form edits the group it shows now, after a switch or Undo", () => {
  const src = readFileSync(
    new URL("../src/features/flows/recipe/recipe-forms.tsx", import.meta.url),
    "utf8",
  );
  const form = src.slice(src.indexOf("export function ProcessorsForm"));
  assert.match(form, /const view = useMemo\(\(\) => viewOf\(groupId\), \[groupId, signature\]\);/);
  // Its only local state is a draft of a template or name that would not survive a save yet.
  assert.equal(form.match(/useState</g)?.length, 2);
  assert.match(form, /if \(isJsonObject\(text\)\) \{\s*setDraft\(null\);\s*patch\(\{ template: text \}\);\s*\} else setDraft\(\{ group: groupId, text \}\);/);
});

test("a column dropped in a recipe frame's setup-card space stays in the recipe", () => {
  const canvas = readFileSync(
    new URL("../src/features/flows/editor/canvas.tsx", import.meta.url),
    "utf8",
  );
  // Membership is tested against the height the frame is drawn at, not only the saved one.
  assert.match(canvas, /const h = Math\.max\(g\.rect\[3\], heights\[groupId\] \?\? 0\);/);
  assert.match(canvas, /groupMembership\(\s*positions,\s*sizesRef\.current,\s*setupRef\.current\.heights,?\s*\)/);
});

test("a GitHub seed added before a repository is entered keeps a form to enter it", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { addRecipeBlock } = await import("../src/features/flows/recipe/recipe-blocks.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  const bare = { ...group, config: { ...group.config } };
  delete (bare.config as Record<string, unknown>).seed_config;
  useDocStore.getState().load({ ...doc(nodes), groups: [bare] });
  addRecipeBlock("recipe", "seed", "seed_github");
  const after = useDocStore.getState().graph.groups.recipe;
  const source = (after.config as { seed_config?: { source?: { seed_type?: string } } }).seed_config?.source;
  assert.equal(source?.seed_type, "github_repo");
  const view = studioView(after, nodes);
  assert.ok(Object.values(view.configs).some((c) => c.kind === "seed"));
});

test("clearing a block's name, column or setup card, keeps its form open until the new name is typed", () => {
  const src = readFileSync(
    new URL("../src/features/flows/recipe/recipe-forms.tsx", import.meta.url),
    "utf8",
  );
  const update = src.slice(src.indexOf("const update = (id: string"), src.indexOf("const canDrop"));
  // The blank shows in the form only; the recipe and the configs a rename starts from keep the
  // last name, so its links follow once the new one is typed.
  const blank = update.slice(0, update.indexOf("return;"));
  assert.match(blank, /typeof patch\.name === "string" &&\s*\(!patch\.name\.trim\(\) \|\| nameTaken\(/);
  assert.match(blank, /setConfigs\(\{\s*\.\.\.latest\.current,/);
  assert.doesNotMatch(blank, /latest\.current =|commitConfigs|applyConfigPatch/);
});

test("a seed whose selection is mid-edit keeps its last valid config and its card", () => {
  const { group, nodes } = sqlGroup();
  const seeded = {
    ...group,
    config: {
      ...group.config,
      // biome-ignore lint/style/useNamingConvention: api schema
      seed_config: {
        source: { seed_type: "hf", path: "org/data" },
        // biome-ignore lint/style/useNamingConvention: api schema
        selection_strategy: { start: 0, end: 10 },
      },
    },
  };
  const view = studioView(seeded, nodes);
  const seed = Object.values(view.configs).find((c) => c.kind === "seed");
  assert.ok(seed);
  // End cleared to type a new one: the form holds the blank, the recipe keeps the last range.
  const configs = applyConfigPatch(view.configs, seed.id, { selection_end: "" } as never);
  const edit = commitStudioEdit(seeded, nodes, view, configs);
  assert.deepEqual(
    (edit.group?.config as Record<string, unknown> | undefined)?.seed_config,
    seeded.config.seed_config,
  );
});

test("renaming a column renames it in the schema transform too", () => {
  const { group, nodes } = sqlGroup();
  const withTransform = {
    ...group,
    config: {
      ...group.config,
      processors: [
        {
          // biome-ignore lint/style/useNamingConvention: api schema
          processor_type: "schema_transform",
          name: "shape",
          template: '{"text": "{{ sql_prompt }}", "kept": "sql_prompt"}',
        },
      ],
    },
  };
  const view = studioView(withTransform, nodes);
  const id = Object.values(view.configs).find((c) => c.name === "sql_prompt")?.id;
  assert.ok(id);
  const configs = applyConfigPatch(view.configs, id, { name: "task" });
  const edit = commitStudioEdit(withTransform, nodes, view, configs);
  const processors = (edit.group?.config as { processors?: { template: unknown }[] }).processors;
  assert.deepEqual(processors?.[0].template, { text: "{{ task }}", kept: "sql_prompt" });
});

test("a pasted or duplicated second Recipe Dataset step comes into its recipe switched off", async () => {
  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group: small, nodes } = sqlGroup();
  const group = { ...small, rect: [0, 0, 2000, 1000] as [number, number, number, number] };
  const step = { id: "step", type: "recipe.generate", v: 1, pos: [group.rect[0] + 40, group.rect[1] + 60] as [number, number], params: {}, group: group.id };
  useDocStore.getState().load({ ...doc([...nodes, step] as never), groups: [group] });
  const store = useDocStore.getState();
  store.insertFragment(cloneFragment(store.graph, ["step"], [40, 40]));
  const steps = Object.values(useDocStore.getState().graph.nodes).filter((n) => n.type === "recipe.generate");
  assert.equal(steps.length, 2);
  assert.deepEqual(steps.map((n) => !!n.disabled).sort(), [false, true]);
  const keys = readFileSync(new URL("../src/features/flows/editor/use-editor-shortcuts.ts", import.meta.url), "utf8");
  assert.match(keys, /doc\.graph\.nodes\[id\]\?\.type !== GENERATE_TYPE/);
});

test("an edit elsewhere keeps a processor the recipe forms do not show", () => {
  const { group, nodes } = sqlGroup();
  const drop = {
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "drop_columns",
    name: "drop_helpers",
    // biome-ignore lint/style/useNamingConvention: api schema
    column_names: ["topic"],
  };
  const withDrop = { ...group, config: { ...group.config, processors: [drop] } };
  const view = studioView(withDrop, nodes);
  const provider = Object.values(view.configs).find((c) => c.kind === "model_provider");
  assert.ok(provider);
  const configs = applyConfigPatch(view.configs, provider.id, { name: "renamed" });
  const edit = commitStudioEdit(withDrop, nodes, view, configs);
  assert.deepEqual((edit.group?.config as { processors?: unknown[] }).processors, [drop]);
});

test("a column duplicated into its recipe gets the next free name", async () => {
  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group: small, nodes } = sqlGroup();
  const group = { ...small, rect: [-100, -100, 4000, 1000] as [number, number, number, number] };
  useDocStore.getState().load({ ...doc(nodes as never), groups: [group] });
  const first = nodes[0];
  const store = useDocStore.getState();
  store.insertFragment(cloneFragment(store.graph, [first.id], [40, 40]));
  store.insertFragment(cloneFragment(useDocStore.getState().graph, [first.id], [80, 80]));
  const name = (first.params as { column: { name: string } }).column.name;
  const names = Object.values(useDocStore.getState().graph.nodes)
    .map((n) => (n.params as { column?: { name?: string } }).column?.name)
    .filter((n) => n?.startsWith(name));
  assert.deepEqual(names.sort(), [name, `${name}_2`, `${name}_3`].sort());
});

test("a recipe column's toolbar has no Run or Regenerate, and a setup card is found while its name ends in a space", () => {
  const toolbar = readFileSync(new URL("../src/features/flows/editor/node-toolbar.tsx", import.meta.url), "utf8");
  assert.match(toolbar, /\{!spec\.design && \(\s*<>\s*<Button[\s\S]*node-toolbar-run[\s\S]*node-toolbar-reroll[\s\S]*<\/>\s*\)\}/);
  const blocks = readFileSync(new URL("../src/features/flows/recipe/recipe-blocks.ts", import.meta.url), "utf8");
  assert.match(blocks, /config\.name\.trim\(\) === target\.name\.trim\(\)/);
  const forms = readFileSync(new URL("../src/features/flows/recipe/recipe-forms.tsx", import.meta.url), "utf8");
  assert.match(forms, /setRecipeSetup\(\{ \.\.\.target, name: name\.trim\(\) \}, true\)/);
});

test("copies of connected columns read each other, not the columns they were copied from", async () => {
  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group: small, nodes } = sqlGroup();
  const group = { ...small, rect: [-100, -100, 4000, 1000] as [number, number, number, number] };
  useDocStore.getState().load({ ...doc(nodes as never), groups: [group] });
  const reader = nodes.find((n) => JSON.stringify(n.params).includes("{{ sql_prompt }}"));
  const source = nodes.find((n) => (n.params as { column: { name: string } }).column.name === "sql_prompt");
  assert.ok(reader && source);
  const store = useDocStore.getState();
  const fragment = cloneFragment(store.graph, [source.id, reader.id], [40, 40]);
  store.insertFragment(fragment);
  const copy = useDocStore.getState().graph.nodes[fragment.nodes[1].id];
  const text = JSON.stringify(copy.params);
  assert.match(text, /\{\{ sql_prompt_2 \}\}/);
  assert.doesNotMatch(text, /\{\{ sql_prompt \}\}/);
});

test("a column rename reaches a drop processor's list, and an edit keeps a preset's unshown settings", () => {
  const { group, nodes } = sqlGroup();
  const cfg = group.config as Record<string, unknown>;
  const presets = (cfg.model_configs as Record<string, unknown>[]).map((m) => ({
    ...m,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_parameters: { ...(m.inference_parameters as object), max_parallel_requests: 16 },
  }));
  const withExtras = {
    ...group,
    config: {
      ...cfg,
      // biome-ignore lint/style/useNamingConvention: api schema
      model_configs: presets,
      // biome-ignore lint/style/useNamingConvention: api schema
      processors: [{ processor_type: "drop_columns", name: "remove_helpers", column_names: ["sql_prompt"] }],
    },
  };
  const view = studioView(withExtras, nodes);
  const id = Object.values(view.configs).find((c) => c.name === "sql_prompt")?.id;
  assert.ok(id);
  const renamed = commitStudioEdit(withExtras, nodes, view, applyConfigPatch(view.configs, id, { name: "task" }));
  const drop = (renamed.group?.config as { processors: { column_names: string[] }[] }).processors;
  assert.deepEqual(drop[0].column_names, ["task"]);
  const provider = Object.values(view.configs).find((c) => c.kind === "model_provider");
  assert.ok(provider);
  const edited = commitStudioEdit(withExtras, nodes, view, applyConfigPatch(view.configs, provider.id, { api_key: "sk-new" } as never));
  const kept = (edited.group?.config as { model_configs: { inference_parameters: Record<string, unknown> }[] }).model_configs;
  assert.equal(kept[0].inference_parameters.max_parallel_requests, 16);
});

test("an edited provider or preset takes what its form says, and a renamed preset keeps what the form does not show", () => {
  const { group, nodes } = sqlGroup();
  const cfg = group.config as Record<string, unknown>;
  const providers = (cfg.model_providers as Record<string, unknown>[]).map((p) => ({ ...p, is_local: true }));
  const presets = (cfg.model_configs as Record<string, unknown>[]).map((m) => ({
    ...m,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_parameters: { temperature: 0.7, top_p: 0.9, max_parallel_requests: 3 },
  }));
  // biome-ignore lint/style/useNamingConvention: api schema
  const saved = { ...group, config: { ...cfg, model_providers: providers, model_configs: presets } };
  const view = studioView(saved, nodes);
  const provider = Object.values(view.configs).find((c) => c.kind === "model_provider");
  const preset = Object.values(view.configs).find((c) => c.kind === "model_config");
  assert.ok(provider && preset);
  // Switched to an external endpoint: the saved is_local must not survive.
  const external = commitStudioEdit(
    saved,
    nodes,
    view,
    applyConfigPatch(view.configs, provider.id, { is_local: false, endpoint: "https://example.com/v1" } as never),
  );
  const sent = (external.group?.config as { model_providers: Record<string, unknown>[] }).model_providers[0];
  assert.notEqual(sent.is_local, true);
  // Temperature cleared in the form, preset renamed: concurrency stays, temperature goes.
  const renamed = commitStudioEdit(
    saved,
    nodes,
    view,
    applyConfigPatch(view.configs, preset.id, { name: "writer2", inference_temperature: "" } as never),
  );
  const params = (renamed.group?.config as { model_configs: { inference_parameters: Record<string, unknown> }[] })
    .model_configs[0].inference_parameters;
  assert.equal(params.max_parallel_requests, 3);
  assert.equal(params.temperature, undefined);
});

test("a copied subcategory and validator follow the copied columns they name", async () => {
  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const column = (id: string, def: Record<string, unknown>, x: number) => ({
    id,
    type: "recipe.column",
    v: 1,
    params: { column: def },
    pos: [x, 0] as [number, number],
    label: String(def.name),
    group: "r",
  });
  const nodes = [
    column("c1", { name: "domain", column_type: "sampler", sampler_type: "category", params: { values: ["a"] } }, 0),
    column("c2", { name: "topic", column_type: "sampler", sampler_type: "subcategory", params: { category: "domain", values: { a: ["x"] } } }, 300),
    column("c3", { name: "check", column_type: "validation", target_columns: ["topic"] }, 600),
  ];
  const group = { id: "r", label: "R", kind: "recipe" as const, rect: [-100, -100, 3000, 1000] as [number, number, number, number], config: {} };
  useDocStore.getState().load({ ...doc(nodes as never), groups: [group] });
  const store = useDocStore.getState();
  const fragment = cloneFragment(store.graph, ["c1", "c2", "c3"], [40, 40]);
  store.insertFragment(fragment);
  const g = useDocStore.getState().graph;
  const [, topic, check] = fragment.nodes.map((n) => (g.nodes[n.id].params as { column: Record<string, unknown> }).column);
  assert.equal(topic.name, "topic_2");
  assert.equal((topic.params as { category: string }).category, "domain_2");
  assert.deepEqual(check.target_columns, ["topic_2"]);
});

test("deleting a column takes its nested and filtered {{ references }} out, and a {% %} reader blocks it", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { disconnectColumns, removeNodes } = await import(
    "../src/features/flows/recipe/recipe-actions.ts"
  );
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  const prompt = byName(nodes, "sql_prompt");
  const sql = byName(nodes, "sql");
  const column = sql.params.column as Record<string, unknown>;
  const nested = {
    ...sql,
    params: {
      column: {
        ...column,
        prompt: "Write SQL.\n{{ sql_prompt.first }} {{ sql_prompt | upper }}\nKeep {{ sql_context }}",
      },
    },
  };
  const withNested = nodes.map((n) => (n.id === sql.id ? nested : n));
  useDocStore.getState().load({ ...doc(withNested, referenceEdges(withNested)), groups: [group] });
  const graph = useDocStore.getState().graph;
  const wire = graph.edgeOrder.find(
    (e) => graph.edges[e].source === prompt.id && graph.edges[e].target === sql.id,
  );
  assert.ok(wire);
  assert.deepEqual(disconnectColumns([wire]), [wire]);
  const after = useDocStore.getState().graph.nodes[sql.id].params.column as { prompt: string };
  assert.equal(after.prompt, "Write SQL.\nKeep {{ sql_context }}");

  // A control tag cannot be rewritten safely: the column stays, and so does every template.
  const control = {
    ...sql,
    params: { column: { ...column, prompt: "{% if sql_prompt %}Write SQL{% endif %}" } },
  };
  const withControl = nodes.map((n) => (n.id === sql.id ? control : n));
  useDocStore.getState().load({ ...doc(withControl, referenceEdges(withControl)), groups: [group] });
  removeNodes([prompt.id]);
  const kept = useDocStore.getState().graph;
  assert.ok(kept.nodes[prompt.id]);
  assert.match(String((kept.nodes[sql.id].params.column as { prompt: string }).prompt), /if sql_prompt/);
});

test("an edit keeps what its form left alone: a preset's temperature distribution and a column's image list", () => {
  const { group, nodes } = sqlGroup();
  const cfg = group.config as Record<string, unknown>;
  const uniform = { distribution_type: "uniform", params: { low: 0.7, high: 1 } };
  const presets = (cfg.model_configs as Record<string, unknown>[]).map((m) => ({
    ...m,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_parameters: { temperature: uniform, max_parallel_requests: 2 },
  }));
  const images = [
    { modality: "image", column_name: "sql_prompt", data_type: "url" },
    { modality: "image", column_name: "sql_context", data_type: "url" },
  ];
  const withImages = nodes.map((n) =>
    columnName(n) === "sql"
      ? { ...n, params: { column: { ...(n.params.column as object), multi_modal_context: images } } }
      : n,
  );
  // biome-ignore lint/style/useNamingConvention: api schema
  const saved = { ...group, config: { ...cfg, model_configs: presets } };
  const view = studioView(saved, withImages);
  const preset = Object.values(view.configs).find((c) => c.kind === "model_config");
  assert.ok(preset);
  const renamed = commitStudioEdit(saved, withImages, view, applyConfigPatch(view.configs, preset.id, { name: "writer2" }));
  const params = (renamed.group?.config as { model_configs: { inference_parameters: Record<string, unknown> }[] })
    .model_configs[0].inference_parameters;
  assert.deepEqual(params, { temperature: uniform, max_parallel_requests: 2 });

  const sql = Object.values(view.configs).find((c) => c.name === "sql");
  assert.ok(sql);
  const dropped = commitStudioEdit(saved, withImages, view, applyConfigPatch(view.configs, sql.id, { drop: true } as never));
  const column = dropped.nodes.find((n) => columnName(n) === "sql")?.params.column as Record<string, unknown>;
  assert.equal(column.drop, true);
  assert.deepEqual(column.multi_modal_context, images);
});

test("an open form compares each edit with what it last saved: a toggle back and two renames in a row", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { commitConfigs, viewOf } = await import("../src/features/flows/recipe/recipe-blocks.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  const cfg = group.config as Record<string, unknown>;
  const presets = (cfg.model_configs as Record<string, unknown>[]).map((m) => ({
    ...m,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_parameters: { ...(m.inference_parameters as object), max_parallel_requests: 16 },
  }));
  // biome-ignore lint/style/useNamingConvention: api schema
  const saved = { ...group, config: { ...cfg, model_configs: presets } };
  useDocStore.getState().load({ ...doc(nodes, referenceEdges(nodes)), groups: [saved] });
  const opened = viewOf("recipe");
  assert.ok(opened);
  const sql = Object.values(opened.configs).find((c) => c.name === "sql");
  assert.ok(sql);
  let base = opened;
  let configs = opened.configs;
  for (const drop of [true, false]) {
    configs = applyConfigPatch(configs, sql.id, { drop } as never);
    base = commitConfigs("recipe", base, configs);
  }
  const column = (id: string) => useDocStore.getState().graph.nodes[id].params.column as Record<string, unknown>;
  assert.equal(column(byName(nodes, "sql").id).drop, false);

  const preset = Object.values(opened.configs).find((c) => c.kind === "model_config");
  assert.ok(preset);
  for (const name of [`${preset.name}X`, `${preset.name}XY`]) {
    configs = applyConfigPatch(configs, preset.id, { name });
    base = commitConfigs("recipe", base, configs);
  }
  const models = useDocStore.getState().graph.groups.recipe.config?.model_configs as {
    alias: string;
    inference_parameters: Record<string, unknown>;
  }[];
  assert.equal(models[0].alias, `${preset.name}XY`);
  assert.equal(models[0].inference_parameters.max_parallel_requests, 16);
});

test("pasted columns are renamed all at once, so one copy's new name is not renamed again", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const rect = [-100, -100, 4000, 1000] as [number, number, number, number];
  const group = { id: "recipe", label: "R", rect, kind: "recipe" as const, config: {}, run: { rows: 5 } };
  const col = (id: string, column: Record<string, unknown>, x: number): FlowNode => ({
    id,
    type: COLUMN_TYPE,
    v: 1,
    params: { column },
    pos: [x, 0],
    label: String(column.name),
    group: "recipe",
  });
  const existing = col("have", { name: "topic", column_type: "sampler", sampler_type: "category", params: { values: ["a"] } }, 0);
  useDocStore.getState().load({ ...doc([existing] as never), groups: [group] });
  const sampler = col("p1", { name: "topic", column_type: "sampler", sampler_type: "category", params: { values: ["b"] } }, 400);
  const expr = col("p2", { name: "topic_2", column_type: "expression", expr: "{{ topic }}" }, 800);
  useDocStore.getState().insertFragment({ nodes: [sampler, expr], edges: [] });
  const graph = useDocStore.getState().graph;
  const pasted = graph.nodes.p1.params.column as { name: string };
  const reader = graph.nodes.p2.params.column as { name: string; expr: string };
  assert.equal(pasted.name, "topic_2");
  assert.equal(reader.name, "topic_2_2");
  assert.equal(reader.expr, "{{ topic_2 }}");
});

test("quoted text in a template reads no column, and deleting a {% %} reader itself is not blocked", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { removeElementsWithReferences } = await import(
    "../src/features/flows/recipe/recipe-actions.ts"
  );
  const { doc } = await import("./flows-editor-fixtures.ts");
  const names = new Set(["topic"]);
  const { columnReferences } = await import("../src/features/flows/recipe/recipe-columns.ts");
  assert.deepEqual(columnReferences({ name: "f", column_type: "expression", expr: '{{ "topic" }}' }, names), []);
  assert.deepEqual(columnReferences({ name: "f", column_type: "expression", expr: "{{ 'topic' ~ x }}" }, names), []);
  assert.deepEqual(columnReferences({ name: "f", column_type: "expression", expr: "{{ topic }}" }, names), ["topic"]);

  const { group, nodes } = sqlGroup();
  const prompt = byName(nodes, "sql_prompt");
  const sql = byName(nodes, "sql");
  const control = {
    ...sql,
    params: {
      column: {
        ...(sql.params.column as object),
        prompt: "{% if sql_prompt == 'x' %}yes{% else %}no{% endif %}",
      },
    },
  };
  const withControl = nodes.map((n) => (n.id === sql.id ? control : n));
  useDocStore.getState().load({ ...doc(withControl, referenceEdges(withControl)), groups: [group] });
  const graph = useDocStore.getState().graph;
  // What the canvas hands over for a selected node: the node and every wire touching it.
  const edges = graph.edgeOrder.filter((id) => graph.edges[id].target === sql.id || graph.edges[id].source === sql.id);
  removeElementsWithReferences({ nodes: [sql.id], edges, groups: [], notes: [] });
  assert.equal(useDocStore.getState().graph.nodes[sql.id], undefined);
  assert.ok(useDocStore.getState().graph.nodes[prompt.id]);
});

test("an edit to another block keeps the seed exactly as saved, an HF seed's split and subset included", () => {
  const { group, nodes } = sqlGroup();
  const cfg = group.config as Record<string, unknown>;
  const seed = {
    source: { seed_type: "hf", path: "nyu-mll/glue", split: "validation", subset: "sst2" },
    sampling_strategy: "ordered",
  };
  // biome-ignore lint/style/useNamingConvention: api schema
  const saved = { ...group, config: { ...cfg, seed_config: seed } };
  const view = studioView(saved, nodes);
  const provider = Object.values(view.configs).find((c) => c.kind === "model_provider");
  assert.ok(provider);
  const edited = commitStudioEdit(
    saved,
    nodes,
    view,
    applyConfigPatch(view.configs, provider.id, { endpoint: "https://example.com/v1" } as never),
  );
  assert.deepEqual((edited.group?.config as Record<string, unknown>).seed_config, seed);
});

test("a rename leaves an attribute of the same name alone, even after a spaced dot", async () => {
  const { replaceRef, renameRoot } = await import("../src/features/recipe-studio/utils/refs.ts");
  assert.equal(
    replaceRef("{{ first_name }} / {{ user . first_name }} / {{ user.first_name }}", "first_name", "given_name"),
    "{{ given_name }} / {{ user . first_name }} / {{ user.first_name }}",
  );
  assert.equal(renameRoot("user .\n first_name == first_name", "first_name", "x"), "user .\n first_name == x");
});

test("columns reading each other are a loop even when a wired seed defers the server's check", async () => {
  const { columnLoop } = await import("../src/features/flows/recipe/recipe-columns.ts");
  assert.deepEqual(
    columnLoop([
      { name: "left", column_type: "expression", expr: "{{ right }}" },
      { name: "right", column_type: "expression", expr: "{{ left }}" },
    ]),
    ["left", "right", "left"],
  );
  assert.equal(
    columnLoop([
      { name: "a", column_type: "expression", expr: "x" },
      { name: "b", column_type: "expression", expr: "{{ a }}" },
    ]),
    null,
  );
  const actions = readFileSync(
    new URL("../src/features/flows/recipe/recipe-group-actions.ts", import.meta.url),
    "utf8",
  );
  assert.match(actions, /if \(wired\) \{[\s\S]*columnLoop\(columns\.map\(columnOf\)\)/);
  const seed = readFileSync(
    new URL("../src/features/recipe-studio/dialogs/seed/seed-dialog.tsx", import.meta.url),
    "utf8",
  );
  // An inline seed form re-reads its documents when one is removed (Flows never closes it).
  assert.match(seed, /const removed = before\.some\(\(id\) => !now\.includes\(id\)\);/);
  assert.match(seed, /onUpdate\(\{ hf_path: "", resolved_paths: \[\], seed_preview_rows: \[\] \}\)/);
});

test("the starter recipe passes Check as made, writing with Chat's loaded model", async () => {
  const { starterRecipeGraph } = await import("../src/features/flows/recipe/recipe-actions.ts");
  const { group, nodes } = starterRecipeGraph([0, 0], {});
  const columns = nodes.filter((n: FlowNode) => n.type === COLUMN_TYPE);
  assert.deepEqual(clientErrors(group, columns), []);
  // Its card still reads as Chat's model, not the word "local".
  const setup = readFileSync(new URL("../src/features/flows/recipe/recipe-setup.ts", import.meta.url), "utf8");
  assert.match(setup, /name\.toLowerCase\(\) === "local"/);
});

test("recipe blocks pasted outside a recipe frame are left out, and a shared tool server edit reaches every profile", async () => {
  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group: small, nodes } = sqlGroup();
  const group = { ...small, rect: [-100, -100, 4000, 1000] as [number, number, number, number] };
  useDocStore.getState().load({ ...doc(nodes as never), groups: [group] });
  const store = useDocStore.getState();
  const before = store.graph.nodeOrder.length;
  // Far outside the frame.
  store.insertFragment(cloneFragment(store.graph, [nodes[0].id], [9000, 9000]));
  assert.equal(useDocStore.getState().graph.nodeOrder.length, before);

  const server = { id: "s1", name: "docs", provider_type: "streamable_http", endpoint: "https://a.example/mcp" };
  const configs = {
    t1: { id: "t1", kind: "tool_config", name: "one", mcp_providers: [server] },
    t2: { id: "t2", kind: "tool_config", name: "two", mcp_providers: [{ ...server, id: "s2" }] },
  } as never;
  const next = applyConfigPatch(configs, "t2", {
    mcp_providers: [{ ...server, id: "s2", endpoint: "https://b.example/mcp" }],
  } as never) as Record<string, { mcp_providers: { id: string; endpoint: string }[] }>;
  assert.equal(next.t1.mcp_providers[0].endpoint, "https://b.example/mcp");
  assert.equal(next.t1.mcp_providers[0].id, "s1");
});

test("a name another block already has stays a draft in the form, so the recipe keeps opening", () => {
  const forms = readFileSync(new URL("../src/features/flows/recipe/recipe-forms.tsx", import.meta.url), "utf8");
  assert.match(forms, /\(!patch\.name\.trim\(\) \|\| nameTaken\(latest\.current, id, patch\.name\)\)/);
  // One namespace for every block, as the importer reads them.
  assert.match(forms, /\(other\) => other\.id !== id && other\.name\.trim\(\) === wanted/);
});

test("moving a column wire to another column moves the reference, in one undo step", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { reconnectWire } = await import("../src/features/flows/editor/use-wire-reconnect.ts");
  const { inHandle, outHandle } = await import("../src/features/flows/editor/connection-rules.ts");
  const { catalog, doc } = await import("./flows-editor-fixtures.ts");
  const { useCatalogStore } = await import("../src/features/flows/stores/catalog-store.ts");
  useCatalogStore.getState().setCatalog(catalog);
  const { group, nodes } = sqlGroup();
  useDocStore.getState().load({ ...doc(nodes, referenceEdges(nodes)), groups: [group] });
  const before = useDocStore.getState().graph;
  const prompt = byName(nodes, "sql_prompt").id;
  const sql = byName(nodes, "sql").id;
  const validator = byName(nodes, "sql-validator").id;
  const id = before.edgeOrder.find((e) => before.edges[e].source === prompt && before.edges[e].target === sql);
  assert.ok(id);
  const old = { id, source: prompt, target: sql, sourceHandle: outHandle("out"), targetHandle: inHandle("in") };
  const error = reconnectWire(old as never, {
    source: prompt,
    sourceHandle: outHandle("out"),
    target: validator,
    targetHandle: inHandle("in"),
  });
  assert.equal(error, null);
  const after = useDocStore.getState().graph;
  const wires = Object.values(after.edges).filter((e) => e.source === prompt);
  assert.ok(wires.some((e) => e.target === validator));
  assert.ok(!wires.some((e) => e.target === sql), "the old wire is gone");
  assert.doesNotMatch(JSON.stringify(after.nodes[sql].params.column), /sql_prompt/);
  useDocStore.getState().undo();
  assert.match(JSON.stringify(useDocStore.getState().graph.nodes[sql].params.column), /sql_prompt/);
});

test("renaming a column renames it in every image context an AI column keeps", () => {
  const { group, nodes } = sqlGroup();
  const images = [
    { modality: "image", column_name: "sql_prompt", data_type: "url" },
    { modality: "image", column_name: "domain", data_type: "url" },
  ];
  const withImages = nodes.map((n) =>
    columnName(n) === "sql"
      ? { ...n, params: { column: { ...(n.params.column as object), multi_modal_context: images } } }
      : n,
  );
  const view = studioView(group, withImages);
  const rename = (from: string, to: string) => {
    const config = Object.values(view.configs).find((c) => c.name === from);
    assert.ok(config);
    const edit = commitStudioEdit(group, withImages, view, applyConfigPatch(view.configs, config.id, { name: to }));
    const sql = edit.nodes.find((n) => n.id === byName(withImages, "sql").id);
    assert.ok(sql, `${from} renamed in sql`);
    return (sql.params.column as Record<string, unknown>).multi_modal_context;
  };
  assert.deepEqual(rename("sql_prompt", "question"), [
    { modality: "image", column_name: "question", data_type: "url" },
    images[1],
  ]);
  // The second image is not on the form: it is renamed in place.
  assert.deepEqual(rename("domain", "field"), [
    images[0],
    { modality: "image", column_name: "field", data_type: "url" },
  ]);
});

test("a recipe block dragged out of its frame stays in its recipe, and the frame grows to hold it", async () => {
  const { ownRecipe, holdRecipeBlocks } = await import("../src/features/flows/editor/recipe-drag.ts");
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  useDocStore.getState().load({
    ...doc([...nodes, { id: "free", type: "input.text", v: 1, params: {}, pos: [0, 0] }], []),
    groups: [group],
  });
  const graph = useDocStore.getState().graph;
  assert.equal(ownRecipe(graph, nodes[0].id), "recipe");
  assert.equal(ownRecipe(graph, "free"), null);
  const grown = holdRecipeBlocks(group as never, { [nodes[0].id]: [2000, 2000] }, new Map());
  assert.deepEqual(grown.rect, [0, 0, 2284, 2144]);
  assert.equal(holdRecipeBlocks(grown, { [nodes[0].id]: [100, 100] }, new Map()), grown);
  const canvas = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(canvas, /const recipe = ownRecipe\(graph, id\);\s*if \(recipe\) membership\[id\] = recipe;/);
  assert.match(canvas, /holdRecipeBlocks\(group, moved, sizesRef\.current\)/);
});

test("a duplicated column skips names the recipe's setup blocks hold, and a copied reader sees its copied image", async () => {
  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group: small, nodes: base } = sqlGroup();
  const presets = (small.config.model_configs as Record<string, unknown>[]).map((m, i) =>
    i === 0 ? { ...m, alias: "domain_2" } : m,
  );
  const group = {
    ...small,
    rect: [-100, -100, 4000, 1000] as [number, number, number, number],
    config: { ...small.config, model_configs: presets },
  };
  const images = [{ modality: "image", column_name: "domain", data_type: "url" }];
  const nodes = base.map((n) =>
    columnName(n) === "sql_prompt"
      ? { ...n, params: { column: { ...(n.params.column as object), multi_modal_context: images } } }
      : n,
  );
  useDocStore.getState().load({ ...doc(nodes as never), groups: [group] });
  const store = useDocStore.getState();
  const domain = byName(nodes, "domain");
  const reader = byName(nodes, "sql_prompt");
  const fragment = cloneFragment(store.graph, [domain.id, reader.id], [40, 40]);
  store.insertFragment(fragment);
  const graph = useDocStore.getState().graph;
  const copied = graph.nodes[fragment.nodes[0].id];
  // domain_2 is a model preset's alias: the copy takes domain_3.
  assert.equal(columnName(copied), "domain_3");
  const copy = graph.nodes[fragment.nodes[1].id].params.column as Record<string, unknown>;
  assert.deepEqual(copy.multi_modal_context, [{ ...images[0], column_name: "domain_3" }]);
  assert.equal(studioView(graph.groups.recipe, Object.values(graph.nodes).filter((n) => n.group === "recipe")).errors.length, 0);
});

test("a formula subtracting two columns reads both, as the backend's import does", async () => {
  const { columnReferences } = await import("../src/features/flows/recipe/recipe-columns.ts");
  const net = { name: "net", column_type: "expression", expr: "{{gross-discount}}" };
  assert.deepEqual(columnReferences(net as never, new Set(["gross", "discount"])), ["gross", "discount"]);
  // A column whose name has a hyphen is still read whole.
  assert.deepEqual(columnReferences(net as never, new Set(["gross-discount"])), ["gross-discount"]);
});

test("a loop's own variable is not a column, inside its loop only", async () => {
  const { columnReferences, templateStillReads } = await import("../src/features/flows/recipe/recipe-columns.ts");
  const names = new Set(["topic", "topics"]);
  const loop = { name: "f", column_type: "expression", expr: "{% for topic in ['math', 'science'] %}{{ topic }} {% endfor %}" };
  assert.deepEqual(columnReferences(loop as never, names), []);
  assert.equal(templateStillReads(loop as never, "topic"), false);
  const outside = { ...loop, expr: "{{ topic }} {% for topic in topics %}{{ topic }}{% endfor %}" };
  assert.deepEqual(columnReferences(outside as never, names), ["topic", "topics"]);
  const set = { ...loop, expr: "{{ topic }}{% set topic = 'x' %}{{ topic }}" };
  assert.deepEqual(columnReferences(set as never, names), ["topic"]);
  // The right side of a `set` and a loop's iterable read the scope before the name is bound.
  const reassigned = { ...loop, expr: "{% set topic = topic | upper %}{{ topic }}" };
  assert.deepEqual(columnReferences(reassigned as never, names), ["topic"]);
  assert.equal(templateStillReads(reassigned as never, "topic"), true);
  const same = { ...loop, expr: "{% for topic in topic %}{{ topic }}{% endfor %}" };
  assert.deepEqual(columnReferences(same as never, names), ["topic"]);
  const local = { ...loop, expr: "{% set label = 'x' %}{{ label }}" };
  assert.deepEqual(columnReferences(local as never, new Set(["label"])), []);
});

test("resizing a recipe frame or grouping its blocks never takes them out of their recipe", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  useDocStore.getState().load({
    ...doc([...nodes, { id: "free", type: "input.text", v: 1, params: {}, pos: [5000, 5000] }] as never, []),
    groups: [{ ...group, rect: [-100, -100, 4000, 1000] }],
  });
  const store = useDocStore.getState();
  store.resizeGroup({ ...useDocStore.getState().graph.groups.recipe, rect: [-100, -100, 400, 200] });
  let graph = useDocStore.getState().graph;
  assert.ok(nodes.every((n) => graph.nodes[n.id].group === "recipe"));
  store.groupNodes({ id: "plain", label: "Group", rect: [0, 0, 10, 10] }, [nodes[0].id, "free"]);
  graph = useDocStore.getState().graph;
  assert.equal(graph.nodes[nodes[0].id].group, "recipe");
  assert.equal(graph.nodes.free.group, "plain");
});

test("a rename or a delete leaves the template's own variable of that name alone", async () => {
  const { replaceRef } = await import("../src/features/recipe-studio/utils/refs.ts");
  const { withoutReference } = await import("../src/features/flows/recipe/recipe-columns.ts");
  assert.equal(
    replaceRef('{{ topic }} / {% set topic = "fixed" %}{{ topic | upper }}', "topic", "subject"),
    '{{ subject }} / {% set topic = "fixed" %}{{ topic | upper }}',
  );
  assert.equal(
    replaceRef("{% set topic = topic | upper %}{{ topic }}", "topic", "subject"),
    "{% set topic = subject | upper %}{{ topic }}",
  );
  assert.equal(
    replaceRef("{% for t in topic %}{{ t }}{% endfor %}{% if topic %}x{% endif %}", "topic", "subject"),
    "{% for t in subject %}{{ t }}{% endfor %}{% if subject %}x{% endif %}",
  );
  const reader = {
    name: "r",
    column_type: "expression",
    expr: "{{ topic }} / {% for topic in ['one', 'two'] %}{{ topic }}{% endfor %}",
  };
  assert.equal(
    withoutReference(reader as never, "topic").expr,
    " / {% for topic in ['one', 'two'] %}{{ topic }}{% endfor %}",
  );
});

test("editing a sampler keeps the settings its form does not show, and a rename reaches a numeric sampler's conditions", () => {
  const { group, nodes } = sqlGroup();
  const extra: FlowNode[] = [
    {
      id: "tier",
      type: COLUMN_TYPE,
      v: 1,
      group: "recipe",
      pos: [0, 400],
      params: {
        column: {
          name: "tier",
          column_type: "sampler",
          sampler_type: "category",
          params: { values: ["basic", "premium"] },
        },
      },
    },
    {
      id: "price",
      type: COLUMN_TYPE,
      v: 1,
      group: "recipe",
      pos: [320, 400],
      params: {
        column: {
          name: "price",
          column_type: "sampler",
          sampler_type: "uniform",
          params: { low: 0, high: 100, decimal_places: 2 },
          conditional_params: { "tier == 'premium'": { sampler_type: "uniform", low: 100, high: 200 } },
        },
      },
    },
  ];
  const all = [...nodes, ...extra];
  const view = studioView(group, all);
  assert.deepEqual(view.errors, []);
  const price = Object.values(view.configs).find((c) => c.name === "price");
  assert.ok(price);
  const edited = commitStudioEdit(group, all, view, applyConfigPatch(view.configs, price.id, { high: "200" } as never));
  const column = edited.nodes.find((n) => n.id === "price")?.params.column as Record<string, unknown>;
  assert.deepEqual(column.params, { low: 0, high: 200, decimal_places: 2 });
  const tier = Object.values(view.configs).find((c) => c.name === "tier");
  assert.ok(tier);
  const renamed = commitStudioEdit(group, all, view, applyConfigPatch(view.configs, tier.id, { name: "plan" }));
  const after = renamed.nodes.find((n) => n.id === "price")?.params.column as Record<string, unknown>;
  assert.deepEqual(Object.keys(after.conditional_params as object), ["plan == 'premium'"]);
});

test("a loop's filter reads its own variable: a rename or a reference never reaches it", async () => {
  const { replaceRef } = await import("../src/features/recipe-studio/utils/refs.ts");
  const { columnReferences } = await import("../src/features/flows/recipe/recipe-columns.ts");
  const expr = "{% for topic in [0, 1] if topic > 0 %}x{% endfor %}|{{ topic }}";
  assert.equal(
    replaceRef(expr, "topic", "subject"),
    "{% for topic in [0, 1] if topic > 0 %}x{% endfor %}|{{ subject }}",
  );
  // A filter on another column is that column's read.
  assert.equal(
    replaceRef("{% for t in items if t > limit %}{% endfor %}", "limit", "cap"),
    "{% for t in items if t > cap %}{% endfor %}",
  );
  const names = new Set(["topic", "limit", "items"]);
  const col = (e: string) => ({ name: "f", column_type: "expression", expr: e }) as never;
  assert.deepEqual(columnReferences(col("{% for topic in [0, 1] if topic > 0 %}x{% endfor %}"), names), []);
  assert.deepEqual(columnReferences(col("{% for t in items if t > limit %}{% endfor %}"), names), ["items", "limit"]);
});

test("a set inside a loop ends with it, and a rename reaches an override's parent column", async () => {
  const { replaceRef } = await import("../src/features/recipe-studio/utils/refs.ts");
  assert.equal(
    replaceRef('{% for item in [1] %}{% set topic = "inside" %}{{ topic }}{% endfor %}{{ topic }}', "topic", "subject"),
    '{% for item in [1] %}{% set topic = "inside" %}{{ topic }}{% endfor %}{{ subject }}',
  );
  const { group } = sqlGroup();
  const nodes: FlowNode[] = [
    {
      id: "domain",
      type: COLUMN_TYPE,
      v: 1,
      group: "recipe",
      pos: [0, 0],
      params: { column: { name: "domain", column_type: "sampler", sampler_type: "category", params: { values: ["science", "art"] } } },
    },
    {
      id: "sub",
      type: COLUMN_TYPE,
      v: 1,
      group: "recipe",
      pos: [320, 0],
      params: {
        column: {
          name: "sub",
          column_type: "sampler",
          sampler_type: "subcategory",
          params: { category: "domain", values: { science: ["physics"], art: ["paint"] } },
          conditional_params: {
            "domain == 'science'": { sampler_type: "subcategory", category: "domain", values: { science: ["chem"], art: ["clay"] } },
          },
        },
      },
    },
  ];
  const view = studioView({ ...group, config: { ...group.config } }, nodes);
  assert.deepEqual(view.errors, []);
  const domain = Object.values(view.configs).find((c) => c.name === "domain");
  assert.ok(domain);
  const edit = commitStudioEdit(group, nodes, view, applyConfigPatch(view.configs, domain.id, { name: "area" }));
  const sub = edit.nodes.find((n) => n.id === "sub")?.params.column as Record<string, unknown>;
  assert.deepEqual(sub.conditional_params, {
    "area == 'science'": { sampler_type: "subcategory", category: "area", values: { science: ["chem"], art: ["clay"] } },
  });
});

test("a seed is not added while another block holds the name seed, which every seed opens as", async () => {
  const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
  const { addRecipeBlock } = await import("../src/features/flows/recipe/recipe-blocks.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group, nodes } = sqlGroup();
  const renamed = nodes.map((n) =>
    columnName(n) === "domain"
      ? { ...n, params: { column: { ...(n.params.column as object), name: "seed" } } }
      : n,
  );
  useDocStore.getState().load({ ...doc(renamed), groups: [group] });
  addRecipeBlock("recipe", "seed", "seed_local");
  const graph = useDocStore.getState().graph;
  assert.equal(graph.groups.recipe.config?.seed_config, undefined);
  const view = studioView(graph.groups.recipe, Object.values(graph.nodes).filter((n) => n.group === "recipe"));
  assert.deepEqual(view.errors, []);
});

test("a rename reaches a transform's Jinja but not a quoted key, and a paste renames readers only after their own recipe's copies", async () => {
  const { group, nodes } = sqlGroup();
  const template = { text: "{{ domain }}", detail: '{{ record["domain"] }}' };
  const withTransform = {
    ...group,
    config: { ...group.config, processors: [{ processor_type: "schema_transform", name: "t", template }] },
  };
  const view = studioView(withTransform, nodes);
  const domain = Object.values(view.configs).find((c) => c.name === "domain");
  assert.ok(domain);
  const edit = commitStudioEdit(withTransform, nodes, view, applyConfigPatch(view.configs, domain.id, { name: "area" }));
  const processors = (edit.group?.config as { processors: { template: unknown }[] }).processors;
  assert.deepEqual(processors[0].template, { text: "{{ area }}", detail: '{{ record["domain"] }}' });

  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const make = (g: string, value: string): FlowNode[] => [
    { id: `${g}_t`, type: COLUMN_TYPE, v: 1, group: g, pos: [0, g === "a" ? 0 : 400], params: { column: { name: "topic", column_type: "sampler", sampler_type: "category", params: { values: [value] } } } },
    { id: `${g}_q`, type: COLUMN_TYPE, v: 1, group: g, pos: [320, g === "a" ? 0 : 400], params: { column: { name: "question", column_type: "expression", expr: "{{ topic }}", dtype: "str" } } },
  ];
  const frame = (id: string, y: number) => ({ ...group, id, rect: [-100, y - 100, 900, 300] as [number, number, number, number] });
  const dest = { ...frame("dest", 1600), rect: [-100, 1400, 4000, 1000] as [number, number, number, number] };
  useDocStore.getState().load({ ...doc([...make("a", "science"), ...make("b", "history")]), groups: [frame("a", 0), frame("b", 400), dest] });
  const store = useDocStore.getState();
  const fragment = cloneFragment(store.graph, ["a_t", "a_q", "b_t", "b_q"], [0, 1600]);
  store.insertFragment(fragment);
  const graph = useDocStore.getState().graph;
  const [, aq, , bq] = fragment.nodes.map((n) => graph.nodes[n.id].params.column as Record<string, unknown>);
  assert.equal(aq.expr, "{{ topic }}");
  assert.equal(bq.expr, "{{ topic_2 }}");
});

test("Jinja whitespace control still closes a loop, and a recipe's model presets are in the download check", async () => {
  const { replaceRef } = await import("../src/features/recipe-studio/utils/refs.ts");
  assert.equal(
    replaceRef("{% for topic in items -%}{{ topic | upper }}{%- endfor %}{{ topic }}", "topic", "subject"),
    "{% for topic in items -%}{{ topic | upper }}{%- endfor %}{{ subject }}",
  );
  const { modelSignature } = await import("../src/features/flows/ux-rules.ts");
  const groups = {
    r: { config: { model_configs: [{ alias: "m", model: "unsloth/Gemma-GGUF", gguf_variant: "Q4_K_M" }] } },
  };
  const column = { type: "recipe.column", group: "r", params: { column: { model_alias: "m" } } };
  const step = { type: "recipe.generate", group: "r", params: {} };
  assert.equal(modelSignature({ c: column, s: step }, groups), "unsloth/gemma-gguf:q4_k_m");
  // Off until its Recipe Dataset step runs, or while no enabled column uses the preset.
  assert.equal(modelSignature({ c: column, s: { ...step, disabled: true } }, groups), "");
  assert.equal(modelSignature({ c: { ...column, disabled: true }, s: step }, groups), "");
  assert.equal(modelSignature({}), "");
});

test("setting a namespace's attribute binds no variable of that name", async () => {
  const { replaceRef } = await import("../src/features/recipe-studio/utils/refs.ts");
  assert.equal(
    replaceRef("{% set ns = namespace() %}{% set ns.topic = 'local' %}{{ topic }}", "topic", "subject"),
    "{% set ns = namespace() %}{% set ns.topic = 'local' %}{{ subject }}",
  );
});

test("a loop's else branch reads the enclosing scope, an if's else inside the loop does not", async () => {
  const { replaceRef } = await import("../src/features/recipe-studio/utils/refs.ts");
  assert.equal(
    replaceRef("{% for topic in [] %}{{ topic }}{%- else -%}{{ topic }}{% endfor %}", "topic", "subject"),
    "{% for topic in [] %}{{ topic }}{%- else -%}{{ subject }}{% endfor %}",
  );
  assert.equal(
    replaceRef("{% for topic in xs %}{% if x %}a{% else %}{{ topic }}{% endif %}{{ topic }}{% endfor %}{{ topic }}", "topic", "subject"),
    "{% for topic in xs %}{% if x %}a{% else %}{{ topic }}{% endif %}{{ topic }}{% endfor %}{{ subject }}",
  );
});

test("with, macro and call blocks bind their own names, so a rename or a reference scan skips them", async () => {
  const { replaceRef } = await import("../src/features/recipe-studio/utils/refs.ts");
  const { columnReferences } = await import("../src/features/flows/recipe/recipe-columns.ts");
  const rename = (t: string) => replaceRef(t, "topic", "area");
  assert.equal(
    rename("{% with topic = topic, n = f(a=1, b=2) %}{{ topic | upper }}{% endwith %}{{ topic }}"),
    "{% with topic = area, n = f(a=1, b=2) %}{{ topic | upper }}{% endwith %}{{ area }}",
  );
  assert.equal(
    rename("{% macro show(topic, n=1) %}{{ topic }}{% endmacro %}{{ show(topic=topic) }}"),
    "{% macro show(topic, n=1) %}{{ topic }}{% endmacro %}{{ show(topic=area) }}",
  );
  assert.equal(
    rename("{% call(topic) lister(topic) %}{{ topic }}{% endcall %}"),
    "{% call(topic) lister(area) %}{{ topic }}{% endcall %}",
  );
  const col = (expr: string) => ({ name: "result", column_type: "expression", expr }) as never;
  const names = new Set(["topic", "words"]);
  assert.deepEqual(columnReferences(col("{% with topic = 'x' %}{{ topic }}{% endwith %}{{ words }}"), names), ["words"]);
  assert.deepEqual(columnReferences(col("{% macro show(topic) %}{{ topic }}{% endmacro %}{{ show(words) }}"), names), ["words"]);
  // A macro's own name is the template's from its declaration on, not a same-named column.
  const macro = "{{ format }}{% macro format(value) %}{{ value | upper }}{% endmacro %}{{ format(topic) }}";
  assert.equal(
    replaceRef(macro, "format", "display_format"),
    "{{ display_format }}{% macro format(value) %}{{ value | upper }}{% endmacro %}{{ format(topic) }}",
  );
  // A macro's parameter defaults read the column; its parameter names do not.
  assert.equal(
    rename("{% macro show(value=topic, n) %}{{ value }}{% endmacro %}{{ show() }}"),
    "{% macro show(value=area, n) %}{{ value }}{% endmacro %}{{ show() }}",
  );
  assert.deepEqual(
    columnReferences(col("{% macro show(value=topic) %}{{ value }}{% endmacro %}{{ show() }}"), names),
    ["topic"],
  );
  // A `set` inside an `if` may not run: the read after it can be the column, so both are renamed.
  assert.equal(
    rename('{% if u %}{% set topic = "o" %}{{ topic }}{% endif %}{{ topic }}'),
    '{% if u %}{% set area = "o" %}{{ area }}{% endif %}{{ area }}',
  );
  assert.deepEqual(
    columnReferences(col('{% if u %}{% set topic = "o" %}{% endif %}{{ topic }}'), names),
    ["topic"],
  );
  assert.equal(rename("{% set topic = topic %}{{ topic }}"), "{% set topic = area %}{{ topic }}");
  // A raw block is literal text: no rename, no reference, no removal inside it.
  const raw = "{%- raw -%}{{ topic | upper }}{% endraw %} {{ topic }}";
  assert.equal(rename(raw), "{%- raw -%}{{ topic | upper }}{% endraw %} {{ area }}");
  assert.deepEqual(columnReferences(col("{% raw %}{{ topic }}{% endraw %}{{ words }}"), names), ["words"]);
  assert.equal(
    withoutReference(col("{% raw %}{{ topic }}{% endraw %}\n{{ topic }}"), "topic").expr,
    "{% raw %}{{ topic }}{% endraw %}",
  );
  const withFormat = new Set(["topic", "format"]);
  assert.deepEqual(
    columnReferences(col("{% macro format(value) %}{{ value }}{% endmacro %}{{ format(topic) }}"), withFormat),
    ["topic"],
  );
});

test("a duplicated reader gets the wire to the column it reads, and a Jinja test is not a column", async () => {
  const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
  const { removeNodes } = await import("../src/features/flows/recipe/recipe-actions.ts");
  const { columnReferences } = await import("../src/features/flows/recipe/recipe-columns.ts");
  const { doc } = await import("./flows-editor-fixtures.ts");
  const { group: small, nodes } = sqlGroup();
  const group = { ...small, rect: [-100, -100, 4000, 1000] as [number, number, number, number] };
  useDocStore.getState().load({ ...doc(nodes, referenceEdges(nodes)), groups: [group] });
  const store = useDocStore.getState();
  const reader = byName(nodes, "sql");
  const source = byName(nodes, "sql_prompt");
  const fragment = cloneFragment(store.graph, [reader.id], [40, 40]);
  store.insertFragment(fragment);
  const copy = fragment.nodes[0].id;
  const graph = useDocStore.getState().graph;
  assert.ok(Object.values(graph.edges).some((e) => e.source === source.id && e.target === copy));
  removeNodes([source.id]);
  const after = useDocStore.getState().graph;
  assert.doesNotMatch(JSON.stringify(after.nodes[copy].params.column), /\{\{ sql_prompt \}\}/);

  const col = (expr: string) => ({ name: "result", column_type: "expression", expr }) as never;
  const names = new Set(["value", "string", "number"]);
  assert.deepEqual(columnReferences(col("{{ value is string }}"), names), ["value"]);
  assert.deepEqual(columnReferences(col("{{ value is not number }}"), names), ["value"]);
  const kw = new Set(["words", "reverse"]);
  assert.deepEqual(columnReferences(col("{{ words | sort(reverse=true) }}"), kw), ["words"]);
  assert.deepEqual(columnReferences(col("{{ words | sort(attribute=reverse) }}"), kw), ["words", "reverse"]);
  assert.deepEqual(columnReferences(col("{{ reverse == words }}"), kw), ["reverse", "words"]);
});
