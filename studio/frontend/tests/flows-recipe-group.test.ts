// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { FlowEdge, FlowNode } from "../src/features/flows/types.ts";
import { catalog, doc, edge, learningRecipe, node, recipeGroup } from "./helpers/flows.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { applyConfigPatch, clientErrors, commitStudioEdit, studioView } = await import(
  "../src/features/flows/recipe/recipe-adapter.ts"
);
const {
  COLUMN_TYPE,
  GENERATE_TYPE,
  columnLoop,
  columnName,
  columnReferences,
  columnSummary,
  groupGenerate,
  referenceEdgeDiff,
  referenceEdges,
  templateStillReads,
  withReference,
  withoutReference,
} = await import("../src/features/flows/recipe/recipe-columns.ts");
const { useDocStore, cloneFragment } = await import("../src/features/flows/stores/doc-store.ts");
const { useCatalogStore } = await import("../src/features/flows/stores/catalog-store.ts");
const {
  disconnectColumns,
  freeSpot,
  insertStarterRecipe,
  nextColumnSlot,
  removeElementsWithReferences,
  removeNodes,
  starterRecipeGraph,
} = await import("../src/features/flows/recipe/recipe-actions.ts");
const { addRecipeBlock, commitConfigs, nameTaken, removeSetup, setupConfig, viewOf } = await import(
  "../src/features/flows/recipe/recipe-blocks.ts"
);
const { COLUMN_BLOCKS, columnBlock } = await import("../src/features/flows/recipe/recipe-kinds.ts");
const {
  CARD_SIZE,
  NODE_SIZE,
  REFERENCE_EDGE_CLASS,
  isSetupId,
  recipeSetup,
  setupEdges,
  setupLayout,
  setupLayoutNodes,
  withLaidOutSetup,
  withSetupPosition,
} = await import("../src/features/flows/recipe/recipe-setup.ts");
const { bareHubSeed, checkRecipeGroupDetail, hubSeedInspect, missingSeed, runColumns } = await import(
  "../src/features/flows/recipe/recipe-group-actions.ts"
);
const { addNodeConnected } = await import("../src/features/flows/editor/node-factory.ts");
const { inHandle, outHandle } = await import("../src/features/flows/editor/connection-rules.ts");
const deriveRf = await import("../src/features/flows/editor/derive-rf.ts");
const { computeLayout } = await import("../src/features/flows/editor/layout-core.ts");
const { reconnectWire } = await import("../src/features/flows/editor/use-wire-reconnect.ts");
const recipeDrag = await import("../src/features/flows/editor/recipe-drag.ts");
const { replaceRef, renameRoot } = await import("../src/features/recipe-studio/utils/refs.ts");
const { modelSignature } = await import("../src/features/flows/ux-rules.ts");
const moduleStubs = await import("./helpers/module-stubs.ts");
const { fakeReact, selectorStore } = await import("./helpers/fake-react.ts");

type Obj = Record<string, unknown>;
type Rect = [number, number, number, number];
type Group = ReturnType<typeof sqlGroup>["group"];
type View = ReturnType<typeof studioView>;
type Config = View["configs"][string];

const SQL = learningRecipe("text-to-sql.json");
const WIDE: Rect = [-100, -100, 4000, 1000];

function sqlGroup() {
  return recipeGroup(SQL, "Text to SQL");
}

const graph = () => useDocStore.getState().graph;

/** Loads `nodes` (+ `edges`) with `groups` into the doc store and returns the graph. */
function load(nodes: FlowNode[], groups: object[] = [], edges: FlowEdge[] = []) {
  useDocStore.getState().load({ ...doc(nodes, edges), groups: groups as never });
  return graph();
}

/** The SQL recipe loaded with its reference wires, in a frame big enough for pastes. */
function loadWired(nodes: FlowNode[], group: Group, rect?: Rect) {
  return load(nodes, [rect ? { ...group, rect } : group], referenceEdges(nodes));
}

const column = (n: FlowNode) => n.params.column as Obj;
const columnText = (n: FlowNode) => JSON.stringify(n.params.column);

function byName(nodes: FlowNode[], name: string): FlowNode {
  const found = nodes.find((n) => columnName(n) === name);
  assert.ok(found, name);
  return found;
}

/** `nodes` with the column called `name` merged with `patch`. */
function withColumn(nodes: FlowNode[], name: string, patch: Obj): FlowNode[] {
  return nodes.map((n) =>
    columnName(n) === name ? { ...n, params: { column: { ...column(n), ...patch } } } : n,
  );
}

const withConfig = (group: Group, patch: Obj): Group => ({ ...group, config: { ...group.config, ...patch } });

/** Gives the provider an endpoint (and blank presets a model) so the client checks pass. */
function ready(group: Group, fillModels = false) {
  const config = group.config as { model_providers: Obj[]; model_configs: Obj[] };
  config.model_providers[0].endpoint = "http://localhost:8000/v1";
  if (fillModels) for (const mc of config.model_configs) mc.model ||= "qwen";
  return group;
}

function findConfig(view: Pick<View, "configs">, match: (c: Config) => boolean, what: string): Config {
  const found = Object.values(view.configs).find(match);
  assert.ok(found, what);
  return found;
}
const ofKind = (view: Pick<View, "configs">, kind: string) => findConfig(view, (c) => c.kind === kind, kind);
const named = (view: Pick<View, "configs">, name: string) => findConfig(view, (c) => c.name === name, name);

/** Commits one form patch on config `id` of `view`. */
function patchEdit(group: Group, nodes: FlowNode[], view: View, id: string, patch: Obj) {
  return commitStudioEdit(group, nodes, view, applyConfigPatch(view.configs, id, patch as never));
}

function colNode(id: string, def: Obj, pos: [number, number], group = "recipe"): FlowNode {
  return { id, type: COLUMN_TYPE, v: 1, params: { column: def }, pos, label: String(def.name), group };
}
const category = (name: string, values: unknown) => ({
  name,
  column_type: "sampler",
  sampler_type: "category",
  params: { values },
});
const exprCol = (expr: string, name = "f") => ({ name, column_type: "expression", expr }) as never;
const generateStep = (id: string, pos: [number, number] = [0, 300]): FlowNode => ({
  ...node(id, GENERATE_TYPE, pos),
  group: "recipe",
});

function loadCatalog() {
  useCatalogStore.setState({ specs: Object.fromEntries(catalog.nodes.map((s) => [s.type, s])) as never });
}

test("every column of a recipe group opens as the Recipe Studio block it was", () => {
  const { group, nodes } = sqlGroup();
  const view = studioView(group, nodes);
  assert.deepEqual(view.errors, []);
  for (const n of nodes) assert.equal(view.configs[view.byNode[n.id]].name, columnName(n));
  const kinds = Object.values(view.configs).map((c) => c.kind);
  assert.ok(kinds.includes("model_provider") && kinds.includes("model_config"));
  assert.equal(view.configs[view.byNode[byName(nodes, "sql").id]].kind, "llm");
});

test("renaming a column rewrites every {{ reference }} to it and touches nothing else", () => {
  const { group, nodes } = sqlGroup();
  const view = studioView(group, nodes);
  const edit = patchEdit(group, nodes, view, view.byNode[byName(nodes, "sql_prompt").id], { name: "task" });
  const changed = new Map(edit.nodes.map((n) => [n.id, n]));
  const renamed = changed.get(byName(nodes, "sql_prompt").id);
  assert.ok(renamed);
  assert.equal(columnName(renamed), "task");
  assert.equal(renamed.label, "task");
  const sql = changed.get(byName(nodes, "sql").id);
  assert.ok(sql, "the column that reads it is rewritten");
  assert.match(String(column(sql).prompt), /\{\{ task \}\}/);
  assert.doesNotMatch(String(column(sql).prompt), /sql_prompt/);
  assert.equal(changed.has(byName(nodes, "domain").id), false);
  assert.equal(edit.group, null, "no recipe-level change");
});

test("a recipe-level edit lands on the group config, not on the columns", () => {
  const { group, nodes } = sqlGroup();
  const view = studioView(group, nodes);
  const edit = patchEdit(group, nodes, view, ofKind(view, "model_provider").id, {
    endpoint: "http://127.0.0.1:8000/v1",
  });
  assert.deepEqual(edit.nodes, []);
  const providers = edit.group?.config?.model_providers as Obj[];
  assert.equal(providers[0].endpoint, "http://127.0.0.1:8000/v1");
  assert.equal("columns" in (edit.group?.config ?? {}), false);
  assert.deepEqual(edit.group?.config?.mcp_providers, []);
});

test("reference wires follow the prompts, like the backend import", () => {
  const { nodes } = sqlGroup();
  const nameOf = (id: string) => columnName(nodes.find((n) => n.id === id) as FlowNode);
  const pairs = referenceEdges(nodes).map((e) => `${nameOf(e.source)}>${nameOf(e.target)}`);
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
  const diff = referenceEdgeDiff(
    load(nodes, [], [edge("stale", nodes[6].id, "out", nodes[0].id, "in")]),
    nodes,
  );
  assert.deepEqual(diff.remove, ["stale"]);
  assert.equal(diff.add.length, 8);
});

test("wiring one column into another adds the {{ reference }}, unwiring takes it out", () => {
  const { nodes } = sqlGroup();
  const sql = column(byName(nodes, "sql"));
  const wired = withReference(sql, "domain");
  assert.ok(wired);
  assert.match(String(wired.prompt), /\{\{ domain \}\}$/);
  assert.equal(withReference(wired, "domain"), wired, "wiring twice changes nothing");
  assert.equal(withoutReference(wired, "domain").prompt, sql.prompt);
  const validator = column(byName(nodes, "sql-validator"));
  assert.deepEqual(withReference(validator, "sql_prompt")?.target_columns, ["sql", "sql_prompt"]);
  assert.equal(withReference(column(byName(nodes, "topic")), "domain"), null);
});

test("a column node shows its kind and what it says", () => {
  const { nodes } = sqlGroup();
  assert.deepEqual(columnSummary(column(nodes[0])).badge, "Category");
  assert.equal(columnSummary(column(byName(nodes, "sql"))).badge, "AI code");
  assert.equal(columnSummary(column(byName(nodes, "sql-validator"))).badge, "SQL check");
});

test("the catalog snapshot has the recipe column and Recipe Dataset blocks", () => {
  const generate = catalog.nodes.find((n) => n.type === "recipe.generate");
  assert.equal(catalog.nodes.find((n) => n.type === "recipe.column")?.design, true);
  assert.deepEqual(
    generate?.outputs.map((p) => p.id),
    ["dataset", "preview"],
  );
});

test("deleting a column from the inspector or Ctrl+X takes its {{ reference }} out of its readers", () => {
  const { group, nodes } = sqlGroup();
  loadWired(nodes, group);
  const prompt = byName(nodes, "sql_prompt");
  const sql = byName(nodes, "sql");
  removeNodes([prompt.id]);
  const after = graph();
  assert.equal(after.nodes[prompt.id], undefined);
  assert.doesNotMatch(columnText(after.nodes[sql.id]), /sql_prompt/);
  assert.ok(Object.values(after.edges).every((e) => e.source !== prompt.id));
  // One Undo brings the column back with the references to it.
  useDocStore.getState().undo();
  assert.ok(graph().nodes[prompt.id]);
  assert.match(columnText(graph().nodes[sql.id]), /sql_prompt/);
});

test("deleting a column on the canvas is one undo step with its references", () => {
  const { group, nodes } = sqlGroup();
  const g = loadWired(nodes, group);
  const prompt = byName(nodes, "sql_prompt");
  const sql = byName(nodes, "sql");
  // What the canvas's Delete key hands over: the node and its wires.
  const edges = g.edgeOrder.filter(
    (id) => g.edges[id].source === prompt.id || g.edges[id].target === prompt.id,
  );
  removeElementsWithReferences({ nodes: [prompt.id], edges, groups: [], notes: [] });
  assert.equal(graph().nodes[prompt.id], undefined);
  assert.doesNotMatch(columnText(graph().nodes[sql.id]), /sql_prompt/);
  useDocStore.getState().undo();
  assert.ok(graph().nodes[prompt.id]);
  assert.match(columnText(graph().nodes[sql.id]), /sql_prompt/);
});

test("deleting reference wires from two recipe frames at once clears both frames' wires", () => {
  const a = sqlGroup();
  const b = sqlGroup();
  const other = b.nodes.map((n) => ({ ...n, id: `b_${n.id}`, group: "recipe_b" }));
  const g = load(
    [...a.nodes, ...other],
    [a.group, { ...b.group, id: "recipe_b" }],
    [...referenceEdges(a.nodes), ...referenceEdges(other)],
  );
  const wire = (nodes: FlowNode[]) => {
    const [source, target] = [byName(nodes, "sql_prompt").id, byName(nodes, "sql").id];
    const id = g.edgeOrder.find((e) => g.edges[e].target === target && g.edges[e].source === source);
    assert.ok(id);
    return id;
  };
  const picked = [wire(a.nodes), wire(other)];
  assert.deepEqual(disconnectColumns(picked).sort(), [...picked].sort());
  for (const id of picked) assert.equal(graph().edges[id], undefined, id);
  for (const nodes of [a.nodes, other])
    assert.doesNotMatch(columnText(graph().nodes[byName(nodes, "sql").id]), /sql_prompt/);
});

test("a Recipe Dataset dropped into a recipe frame joins it, and a second one is refused", () => {
  loadCatalog();
  const { group, nodes } = sqlGroup();
  load(nodes, [{ ...group, rect: [0, 0, 2000, 600] }]);
  const added = addNodeConnected(GENERATE_TYPE, [100, 400]);
  assert.equal(added?.group, "recipe");
  assert.equal(graph().nodes[added?.id ?? ""]?.group, "recipe");
  assert.equal(addNodeConnected(GENERATE_TYPE, [200, 400]), null);
});

test("Quick Add of a Recipe Dataset from a dataset output wires that dataset into its seed", () => {
  loadCatalog();
  load([node("hf", "data.hf_dataset")]);
  const from = {
    nodeId: "hf",
    handleId: outHandle("dataset"),
    side: "out",
    kind: "data",
    port: null,
  } as const;
  const generate = addNodeConnected(GENERATE_TYPE, [600, 0], from, "seed");
  assert.ok(generate);
  assert.ok(
    Object.values(graph().edges).some(
      (e) => e.source === "hf" && e.target === generate.id && e.target_port === "seed",
    ),
  );
});

test("every Recipe Studio column can be added, and its card names the block it is", () => {
  const { group, nodes } = sqlGroup();
  assert.deepEqual(
    COLUMN_BLOCKS.filter((b) => b.kind === "validator").map((b) => b.label),
    ["Python check", "SQL check", "JS/TS check", "JSON check", "Markdown check"],
  );
  for (const block of COLUMN_BLOCKS) {
    load(nodes, [group]);
    addRecipeBlock("recipe", block.kind, block.type);
    const g = graph();
    const added = g.nodeOrder
      .map((id) => g.nodes[id])
      .find((n) => n.type === COLUMN_TYPE && !nodes.some((m) => m.id === n.id));
    assert.ok(added, block.type);
    assert.equal(added.group, "recipe");
    assert.equal(columnBlock(column(added)).type, block.type, block.label);
  }
});

test("a seed, model preset and provider show as cards; a shared file seed says to load it first", () => {
  const { group, nodes } = sqlGroup();
  const shared = withConfig(group, { seed_config: { source: { seed_type: "local", path: "" } } });
  const { cards, links } = recipeSetup(load(nodes, [shared]), shared);
  assert.deepEqual(
    cards.map((c) => c.kind),
    ["seed", "model_provider", "model_config"],
  );
  assert.equal(cards[0].warning, "Load the seed first");
  assert.match(missingSeed(shared) ?? "", /^Load the seed first/);
  assert.equal(missingSeed(group), null);
  assert.ok(links.some((l) => l.source === cards[2].id && l.target === byName(nodes, "sql").id));
});

test("a starter recipe lands in free space, not on top of what is there", () => {
  load([{ ...node("a", "input.text"), pos: [0, 0] }]);
  const generate = insertStarterRecipe([10, 10], { rows: 5 });
  const frame = graph().groups[generate.group as string];
  assert.ok(frame.rect[1] > 100, `frame at ${frame.rect}`);
  load([]);
  const alone = insertStarterRecipe([10, 10], { rows: 5 });
  assert.deepEqual(graph().groups[alone.group as string].rect.slice(0, 2), [-30, -50]);
});

test("a tool profile added without a server yet stays as a card and can be removed", () => {
  const { group, nodes } = sqlGroup();
  load(nodes, [group]);
  addRecipeBlock("recipe", "llm", "tool_config");
  const tools = recipeSetup(graph(), graph().groups.recipe).cards.filter((c) => c.kind === "tool_config");
  assert.equal(tools.length, 1);
  assert.equal(tools[0].warning, "Add a tool server");
  removeSetup({ groupId: "recipe", kind: "tool_config", name: tools[0].name });
  assert.equal(
    recipeSetup(graph(), graph().groups.recipe).cards.some((c) => c.kind === "tool_config"),
    false,
  );
});

test("Check flags an empty endpoint and model the way Recipe Studio does, before the server", () => {
  const { group, nodes } = sqlGroup();
  const config = group.config as { model_providers: Obj[]; model_configs: Obj[] };
  // The learning recipe ships both blank; Recipe Studio refuses it until they are set.
  config.model_providers[0].endpoint = "";
  config.model_configs[0].model = "";
  const errors = clientErrors(group, nodes);
  assert.ok(
    errors.some((e) => /endpoint is required/.test(e)),
    errors.join("; "),
  );
  assert.ok(
    errors.some((e) => /model is required/.test(e)),
    errors.join("; "),
  );
});

/** The SQL recipe with an endpoint, and the only columns using the model-less preset switched off. */
function withPresetColumnsOff() {
  const { group, nodes } = sqlGroup();
  const off = new Set(["sql_prompt", "sql", "sql-validator"]);
  const columns = nodes.map((n) => (off.has(columnName(n)) ? { ...n, disabled: true } : n));
  return { group: ready(group), columns, off };
}

test("Check leaves disabled columns and a disabled Recipe Dataset out, like the compiler", () => {
  const { group, columns, off } = withPresetColumnsOff();
  const step = (id: string, disabled: boolean): FlowNode => ({
    ...generateStep(id, [0, 400]),
    params: { rows: 5, mode: "preview" },
    disabled,
  });
  const g = load([...columns, step("old_step", true), step("live_step", false)], [group]);
  assert.equal(groupGenerate(g, "recipe")?.id, "live_step");
  const checked = runColumns(g, "recipe");
  assert.ok(checked.every((n) => !off.has(columnName(n))));
  assert.deepEqual(clientErrors(group, checked), []);
  assert.ok(clientErrors(group, columns).some((e) => /model is required/.test(e)));
});

test("a Hub seed still named by its repo id is picked out for resolving, a reader path is not", () => {
  const seed = (path: string, type = "hf") => ({ seed_config: { source: { seed_type: type, path } } });
  assert.ok(bareHubSeed(seed("unsloth/alpaca-cleaned")));
  assert.equal(bareHubSeed(seed("datasets/unsloth/alpaca-cleaned/**/*.parquet")), null);
  assert.equal(bareHubSeed(seed("unsloth/alpaca-cleaned", "local")), null);
  assert.equal(bareHubSeed(seed("")), null);
  assert.equal(bareHubSeed({}), null);
  // Check asks with what the run sends, token included, so a gated seed checks like it runs.
  const gated = bareHubSeed({
    seed_config: { source: { seed_type: "hf", path: "org/gated", token: "hf_x", subset: "a" } },
  });
  assert.deepEqual(hubSeedInspect(gated ?? {}), {
    dataset_name: "org/gated",
    split: "train",
    subset: "a",
    hf_token: "hf_x",
    preview_size: 1,
  });
});

test("setup links are the flow's own edge type, dashed as references, so they follow the wire style", () => {
  const { group, nodes } = sqlGroup();
  const sql = byName(nodes, "sql");
  const g = load(
    nodes.map((n) => (n.id === sql.id ? { ...n, collapsed: true } : n)),
    [group],
  );
  const edges = setupEdges(recipeSetup(g, group).links, g, { source: "out", target: "in" });
  assert.ok(edges.length > 0);
  for (const e of edges) {
    // flowEdge: Curved or bus-routed with every other wire; the class keeps it a dashed reference.
    assert.equal(e.type, "flowEdge");
    assert.equal(e.className, REFERENCE_EDGE_CLASS);
    assert.deepEqual([e.sourceHandle, e.targetHandle], ["out", "in"]);
    assert.equal(e.selectable, false);
    assert.equal(e.deletable, false);
  }
  assert.ok(!edges.some((e) => e.target === sql.id), "a collapsed column draws no link");
  const css = readSrc("index.css");
  assert.match(css, new RegExp(`\\.${REFERENCE_EDGE_CLASS} \\.react-flow__edge-path[^}]*stroke-dasharray`));
});

test("a note keeps the size it was given whatever its text; long text scrolls inside", () => {
  useDocStore
    .getState()
    .load({ ...doc([]), notes: [{ id: "n1", text: "a long note", rect: [0, 0, 320, 200] }] });
  const note = new deriveRf.RfDeriver().nodes(graph(), new Set(), null).find((n) => n.type === "flowNote");
  assert.deepEqual(note?.style, { width: 320, height: 200 });
});

/** Every pair of overlapping boxes among the current graph's nodes and setup cards. */
function recipeOverlaps(sizes: Map<string, { width: number; height: number }>) {
  const g = graph();
  const boxes = [
    ...g.nodeOrder.map((id) => ({
      id,
      x: g.nodes[id].pos[0],
      y: g.nodes[id].pos[1],
      ...(sizes.get(id) ?? NODE_SIZE),
    })),
    ...setupLayout(g, null, sizes).elements.map(({ card, x, y }) => ({
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

/** `nodes` laid out as the importer does: a 4-wide grid of 320 x 180 cells. */
const onGrid = (nodes: FlowNode[]) =>
  nodes.map((n, i) => ({ ...n, pos: [(i % 4) * 320, Math.floor(i / 4) * 180] as [number, number] }));
const renamedCopy = (n: FlowNode, id: string) => ({
  ...n,
  id,
  params: { column: { ...column(n), name: id } },
});

test("no two nodes of a recipe group overlap after import, after adding setup blocks, or after Tidy up", () => {
  const { group, nodes } = sqlGroup();
  // The Recipe Dataset step at the end of row 0, plus a column left in the step's lane from an older
  // layout (the ticket-router report).
  const placed = onGrid(nodes);
  const step = { ...generateStep("step", [1280, 0]), params: { rows: 5, mode: "preview" } };
  const stray = { ...renamedCopy(placed[0], "stray"), pos: [1280, 460] as [number, number] };
  load([...placed, step, stray], [{ ...group, rect: [-40, -60, 1600, 700] }]);
  // A finished run makes the step tall (Done, a dataset table).
  const sizes = new Map([["step", { width: 260, height: 420 }]]);
  assert.deepEqual(recipeOverlaps(sizes), [], "after import");

  const before = setupLayout(graph(), null, sizes).elements.length;
  for (const type of ["model_provider", "model_config", "tool_config"]) addRecipeBlock("recipe", "llm", type);
  assert.equal(setupLayout(graph(), null, sizes).elements.length, before + 3);
  assert.deepEqual(recipeOverlaps(sizes), [], "after adding setup blocks");

  // A card dragged into the column grid: the next column goes around it.
  const first = setupLayout(graph(), null, sizes).elements[0];
  useDocStore.getState().upsertGroup(withSetupPosition(graph().groups.recipe, first.card.id, [320, 360]));
  const columns = graph().nodeOrder.length;
  addRecipeBlock("recipe", "sampler", "uuid");
  assert.equal(graph().nodeOrder.length, columns + 1);
  assert.deepEqual(recipeOverlaps(sizes), [], "a new column keeps clear of a moved card");

  const g = graph();
  const setup = setupLayoutNodes(g, (id) => sizes.get(id));
  const result = computeLayout({
    direction: "LR",
    nodes: [
      ...g.nodeOrder.map((id) => ({ id, ...(sizes.get(id) ?? { width: 260, height: 140 }) })),
      ...setup.nodes,
    ],
    edges: [
      ...g.edgeOrder.map((id) => ({ source: g.edges[id].source, target: g.edges[id].target })),
      ...setup.edges,
    ],
  });
  assert.ok(Object.keys(result).some(isSetupId), "Tidy up lays out the setup cards too");
  useDocStore.getState().moveNodes(Object.fromEntries(Object.entries(result).filter(([k]) => !isSetupId(k))));
  useDocStore.getState().patchGraph({ groups: withLaidOutSetup(graph(), result) });
  assert.deepEqual(recipeOverlaps(sizes), [], "after Tidy up");
});

test("a starter recipe and a new column both keep clear of setup cards, even dragged ones", () => {
  const { group, nodes } = sqlGroup();
  // Every cell of a two-row frame is full, so a new column needs a new row.
  const placed = onGrid([...nodes, renamedCopy(nodes[0], "extra")]);
  assert.equal(placed.length, 8);
  assert.equal(setupLayout({ ...graph(), groups: {}, groupOrder: [] }, null, new Map()).elements.length, 0);
  load(placed, [{ ...group, rect: [-40, -60, 1280, 400] }]);
  const cardId = setupLayout(graph(), null, new Map()).elements[0].card.id;
  const cardAt = (pos: [number, number]) => {
    useDocStore.getState().upsertGroup(withSetupPosition(graph().groups.recipe, cardId, pos));
    const card = setupLayout(graph(), null, new Map()).elements.find((e) => e.card.id === cardId);
    assert.ok(card);
    return card;
  };
  // Dragged to where the next row would start, and to the empty canvas below the frame.
  const moved = cardAt([0, 360]);
  const { pos } = nextColumnSlot(graph(), "recipe");
  const hit = (p: [number, number], at: { x: number; y: number }) =>
    p[0] < at.x + CARD_SIZE.width && at.x < p[0] + 280 && p[1] < at.y + CARD_SIZE.height && at.y < p[1] + 140;
  assert.ok(!hit(pos, moved), `column at ${pos} sits on the card at ${moved.x},${moved.y}`);
  const below = cardAt([0, 900]);
  const spot = freeSpot(graph(), [below.x + 40, below.y + 60], [600, 400]);
  assert.notDeepEqual(spot, [below.x + 40, below.y + 60], "the starter does not land on the card");
});

test("the run dialogs' check leaves disabled columns out, like the frame's Check", async () => {
  const { group, columns } = withPresetColumnsOff();
  load(columns, [group]);
  // Past the client checks; the server half cannot load under node, which is fine.
  const result = await checkRecipeGroupDetail("recipe");
  assert.ok(!result.errors.some((e) => /model is required/.test(e)), result.errors.join("; "));
});

test("a loaded document seed keeps its files when the group opens in Recipe Studio's forms", () => {
  const { group, nodes } = sqlGroup();
  const withDocs = withConfig(group, {
    seed_config: { source: { seed_type: "unstructured", paths: ["/up/x/a.pdf"] } },
  });
  const seed = ofKind(studioView(withDocs, nodes), "seed") as { resolved_paths?: string[] };
  assert.deepEqual(seed.resolved_paths, ["/up/x/a.pdf"]);
});

test("a wired seed is what the check goes by, not the group's own empty file seed", async () => {
  const { group, nodes } = sqlGroup();
  const shared = withConfig(ready(group, true), {
    seed_config: { source: { seed_type: "unstructured", paths: [] } },
  });
  const loadWith = (wired: boolean) =>
    load(
      [...nodes, generateStep("step"), node("data", "data.hf_dataset", [0, 600])],
      [shared],
      wired ? [edge("seed", "data", "dataset", "step", "seed")] : [],
    );
  loadWith(false);
  assert.ok((await checkRecipeGroupDetail("recipe")).errors.some((e) => /source-data preview/.test(e)));
  loadWith(true);
  assert.deepEqual(await checkRecipeGroupDetail("recipe"), { valid: true, errors: [], rawDetail: null });
});

test("a recipe with a wired seed still fails Check when its columns read each other", async () => {
  const { group, nodes } = sqlGroup();
  const loop = (name: string, reads: string, i: number) =>
    colNode(`loop_${name}`, exprCol(`{{ ${reads} }}`, name), [i * 320, 900]);
  load(
    [
      ...nodes,
      loop("left", "right", 0),
      loop("right", "left", 1),
      generateStep("step"),
      node("data", "data.hf_dataset", [0, 600]),
    ],
    [ready(group, true)],
    [edge("seed", "data", "dataset", "step", "seed")],
  );
  const result = await checkRecipeGroupDetail("recipe");
  assert.equal(result.valid, false);
  assert.ok(
    result.errors.some((e) => /loop: left -> right -> left/.test(e)),
    result.errors.join("; "),
  );
});

test("a GitHub seed added before a repository is entered keeps a form to enter it", () => {
  const { group, nodes } = sqlGroup();
  const bare = { ...group, config: { ...group.config } };
  delete (bare.config as Obj).seed_config;
  load(nodes, [bare]);
  addRecipeBlock("recipe", "seed", "seed_github");
  const after = graph().groups.recipe;
  const source = (after.config as { seed_config?: { source?: { seed_type?: string } } }).seed_config?.source;
  assert.equal(source?.seed_type, "github_repo");
  assert.ok(Object.values(studioView(after, nodes).configs).some((c) => c.kind === "seed"));
});

test("a seed whose selection is mid-edit keeps its last valid config and its card", () => {
  const { group, nodes } = sqlGroup();
  const seedConfig = {
    source: { seed_type: "hf", path: "org/data" },
    selection_strategy: { start: 0, end: 10 },
  };
  const seeded = withConfig(group, { seed_config: seedConfig });
  const view = studioView(seeded, nodes);
  // End cleared to type a new one: the form holds the blank, the recipe keeps the last range.
  const edit = patchEdit(seeded, nodes, view, ofKind(view, "seed").id, { selection_end: "" });
  assert.deepEqual((edit.group?.config as Obj | undefined)?.seed_config, seedConfig);
});

test("renaming a column renames it in the schema transform too", () => {
  const { group, nodes } = sqlGroup();
  const template = '{"text": "{{ sql_prompt }}", "kept": "sql_prompt"}';
  const withTransform = withConfig(group, {
    processors: [{ processor_type: "schema_transform", name: "shape", template }],
  });
  const view = studioView(withTransform, nodes);
  const edit = patchEdit(withTransform, nodes, view, named(view, "sql_prompt").id, { name: "task" });
  const processors = (edit.group?.config as { processors?: { template: unknown }[] }).processors;
  assert.deepEqual(processors?.[0].template, { text: "{{ task }}", kept: "sql_prompt" });
});

test("a pasted or duplicated second Recipe Dataset step comes into its recipe switched off", () => {
  const { group, nodes } = sqlGroup();
  const g = load(
    [...nodes, { ...generateStep("step", [40, 60]), params: {} }],
    [{ ...group, rect: [0, 0, 2000, 1000] }],
  );
  useDocStore.getState().insertFragment(cloneFragment(g, ["step"], [40, 40]));
  const steps = Object.values(graph().nodes).filter((n) => n.type === "recipe.generate");
  assert.equal(steps.length, 2);
  assert.deepEqual(steps.map((n) => !!n.disabled).sort(), [false, true]);
});

test("an edit elsewhere keeps a processor the recipe forms do not show", () => {
  const { group, nodes } = sqlGroup();
  const drop = { processor_type: "drop_columns", name: "drop_helpers", column_names: ["topic"] };
  const withDrop = withConfig(group, { processors: [drop] });
  const view = studioView(withDrop, nodes);
  const edit = patchEdit(withDrop, nodes, view, ofKind(view, "model_provider").id, { name: "renamed" });
  assert.deepEqual((edit.group?.config as { processors?: unknown[] }).processors, [drop]);
});

test("a column duplicated into its recipe gets the next free name", () => {
  const { group, nodes } = sqlGroup();
  load(nodes, [{ ...group, rect: WIDE }]);
  const first = nodes[0];
  useDocStore.getState().insertFragment(cloneFragment(graph(), [first.id], [40, 40]));
  useDocStore.getState().insertFragment(cloneFragment(graph(), [first.id], [80, 80]));
  const name = columnName(first);
  const names = Object.values(graph().nodes)
    .map((n) => (n.params as { column?: { name?: string } }).column?.name)
    .filter((n) => n?.startsWith(name));
  assert.deepEqual(names.sort(), [name, `${name}_2`, `${name}_3`].sort());
});

test("a setup card is found while its name ends in a space", () => {
  const { group, nodes } = sqlGroup();
  const view = studioView(group, nodes);
  const provider = ofKind(view, "model_provider");
  const typing = applyConfigPatch(view.configs, provider.id, { name: `${provider.name} ` });
  const card = (name: string) => ({ groupId: "recipe", kind: "model_provider", name }) as never;
  assert.equal(setupConfig({ ...view, configs: typing }, card(provider.name))?.id, provider.id);
  assert.equal(setupConfig(view, card("nope")), undefined);
});

test("copies of connected columns read each other, not the columns they were copied from", () => {
  const { group, nodes } = sqlGroup();
  load(nodes, [{ ...group, rect: WIDE }]);
  const reader = nodes.find((n) => JSON.stringify(n.params).includes("{{ sql_prompt }}"));
  const source = byName(nodes, "sql_prompt");
  assert.ok(reader);
  const fragment = cloneFragment(graph(), [source.id, reader.id], [40, 40]);
  useDocStore.getState().insertFragment(fragment);
  const text = JSON.stringify(graph().nodes[fragment.nodes[1].id].params);
  assert.match(text, /\{\{ sql_prompt_2 \}\}/);
  assert.doesNotMatch(text, /\{\{ sql_prompt \}\}/);
});

/** The SQL recipe with every preset's inference parameters replaced by `params(preset)`. */
function withPresetParams(params: (preset: Obj) => Obj, extra: Obj = {}) {
  const { group, nodes } = sqlGroup();
  const presets = (group.config.model_configs as Obj[]).map((m) => ({
    ...m,
    inference_parameters: params(m),
  }));
  return { nodes, saved: withConfig(group, { model_configs: presets, ...extra }) };
}
const presetParams = (group: { config?: unknown } | null | undefined) =>
  (group?.config as { model_configs: { inference_parameters: Obj }[] }).model_configs[0].inference_parameters;

test("a column rename reaches a drop processor's list, and an edit keeps a preset's unshown settings", () => {
  const { nodes, saved } = withPresetParams(
    (m) => ({ ...(m.inference_parameters as object), max_parallel_requests: 16 }),
    {
      processors: [{ processor_type: "drop_columns", name: "remove_helpers", column_names: ["sql_prompt"] }],
    },
  );
  const view = studioView(saved, nodes);
  const renamed = patchEdit(saved, nodes, view, named(view, "sql_prompt").id, { name: "task" });
  const drop = (renamed.group?.config as { processors: { column_names: string[] }[] }).processors;
  assert.deepEqual(drop[0].column_names, ["task"]);
  const edited = patchEdit(saved, nodes, view, ofKind(view, "model_provider").id, { api_key: "sk-new" });
  assert.equal(presetParams(edited.group).max_parallel_requests, 16);
});

test("an edited provider or preset takes what its form says, and a renamed preset keeps what the form does not show", () => {
  const { nodes, saved: base } = withPresetParams(() => ({
    temperature: 0.7,
    top_p: 0.9,
    max_parallel_requests: 3,
  }));
  const providers = (base.config.model_providers as Obj[]).map((p) => ({ ...p, is_local: true }));
  const saved = withConfig(base, { model_providers: providers });
  const view = studioView(saved, nodes);
  // Switched to an external endpoint: the saved is_local must not survive.
  const external = patchEdit(saved, nodes, view, ofKind(view, "model_provider").id, {
    is_local: false,
    endpoint: "https://example.com/v1",
  });
  assert.notEqual((external.group?.config as { model_providers: Obj[] }).model_providers[0].is_local, true);
  // Temperature cleared in the form, preset renamed: concurrency stays, temperature goes.
  const preset = ofKind(view, "model_config").id;
  const params = presetParams(
    patchEdit(saved, nodes, view, preset, { name: "writer2", inference_temperature: "" }).group,
  );
  assert.equal(params.max_parallel_requests, 3);
  assert.equal(params.temperature, undefined);
});

test("a copied subcategory and validator follow the copied columns they name", () => {
  const topic = {
    name: "topic",
    column_type: "sampler",
    sampler_type: "subcategory",
    params: { category: "domain", values: { a: ["x"] } },
  };
  const nodes = [
    colNode("c1", category("domain", ["a"]), [0, 0], "r"),
    colNode("c2", topic, [300, 0], "r"),
    colNode("c3", { name: "check", column_type: "validation", target_columns: ["topic"] }, [600, 0], "r"),
  ];
  const g = load(nodes, [
    { id: "r", label: "R", kind: "recipe", rect: [-100, -100, 3000, 1000], config: {} },
  ]);
  const fragment = cloneFragment(g, ["c1", "c2", "c3"], [40, 40]);
  useDocStore.getState().insertFragment(fragment);
  const [, copiedTopic, check] = fragment.nodes.map((n) => column(graph().nodes[n.id]));
  assert.equal(copiedTopic.name, "topic_2");
  assert.equal((copiedTopic.params as { category: string }).category, "domain_2");
  assert.deepEqual(check.target_columns, ["topic_2"]);
});

test("deleting a column takes its nested and filtered {{ references }} out, and a {% %} reader blocks it", () => {
  const { group, nodes } = sqlGroup();
  const prompt = byName(nodes, "sql_prompt");
  const sql = byName(nodes, "sql");
  const nested = withColumn(nodes, "sql", {
    prompt: "Write SQL.\n{{ sql_prompt.first }} {{ sql_prompt | upper }}\nKeep {{ sql_context }}",
  });
  const g = loadWired(nested, group);
  const wire = g.edgeOrder.find((e) => g.edges[e].source === prompt.id && g.edges[e].target === sql.id);
  assert.ok(wire);
  assert.deepEqual(disconnectColumns([wire]), [wire]);
  assert.equal(column(graph().nodes[sql.id]).prompt, "Write SQL.\nKeep {{ sql_context }}");

  // A control tag cannot be rewritten safely: the column stays, and so does every template.
  loadWired(withColumn(nodes, "sql", { prompt: "{% if sql_prompt %}Write SQL{% endif %}" }), group);
  removeNodes([prompt.id]);
  assert.ok(graph().nodes[prompt.id]);
  assert.match(String(column(graph().nodes[sql.id]).prompt), /if sql_prompt/);
});

test("an edit keeps what its form left alone: a preset's temperature distribution and a column's image list", () => {
  const uniform = { distribution_type: "uniform", params: { low: 0.7, high: 1 } };
  const { nodes, saved } = withPresetParams(() => ({ temperature: uniform, max_parallel_requests: 2 }));
  const images = [
    { modality: "image", column_name: "sql_prompt", data_type: "url" },
    { modality: "image", column_name: "sql_context", data_type: "url" },
  ];
  const withImages = withColumn(nodes, "sql", { multi_modal_context: images });
  const view = studioView(saved, withImages);
  const renamed = patchEdit(saved, withImages, view, ofKind(view, "model_config").id, { name: "writer2" });
  assert.deepEqual(presetParams(renamed.group), { temperature: uniform, max_parallel_requests: 2 });

  const dropped = patchEdit(saved, withImages, view, named(view, "sql").id, { drop: true });
  const sql = column(byName(dropped.nodes, "sql"));
  assert.equal(sql.drop, true);
  assert.deepEqual(sql.multi_modal_context, images);
});

test("an open form compares each edit with what it last saved: a toggle back and two renames in a row", () => {
  const { nodes, saved } = withPresetParams((m) => ({
    ...(m.inference_parameters as object),
    max_parallel_requests: 16,
  }));
  loadWired(nodes, saved);
  const opened = viewOf("recipe");
  assert.ok(opened);
  const sql = named(opened, "sql");
  let base = opened;
  let configs = opened.configs;
  for (const drop of [true, false]) {
    configs = applyConfigPatch(configs, sql.id, { drop } as never);
    base = commitConfigs("recipe", base, configs);
  }
  assert.equal(column(graph().nodes[byName(nodes, "sql").id]).drop, false);

  const preset = ofKind(opened, "model_config");
  for (const name of [`${preset.name}X`, `${preset.name}XY`]) {
    configs = applyConfigPatch(configs, preset.id, { name });
    base = commitConfigs("recipe", base, configs);
  }
  const models = graph().groups.recipe.config?.model_configs as {
    alias: string;
    inference_parameters: Obj;
  }[];
  assert.equal(models[0].alias, `${preset.name}XY`);
  assert.equal(models[0].inference_parameters.max_parallel_requests, 16);
});

test("pasted columns are renamed all at once, so one copy's new name is not renamed again", () => {
  load(
    [colNode("have", category("topic", ["a"]), [0, 0])],
    [{ id: "recipe", label: "R", rect: WIDE, kind: "recipe", config: {}, run: { rows: 5 } }],
  );
  useDocStore.getState().insertFragment({
    nodes: [
      colNode("p1", category("topic", ["b"]), [400, 0]),
      colNode("p2", exprCol("{{ topic }}", "topic_2"), [800, 0]),
    ],
    edges: [],
  });
  const pasted = column(graph().nodes.p1);
  const reader = column(graph().nodes.p2);
  assert.equal(pasted.name, "topic_2");
  assert.equal(reader.name, "topic_2_2");
  assert.equal(reader.expr, "{{ topic_2 }}");
});

test("quoted text in a template reads no column, and deleting a {% %} reader itself is not blocked", () => {
  const names = new Set(["topic"]);
  assert.deepEqual(columnReferences(exprCol('{{ "topic" }}'), names), []);
  assert.deepEqual(columnReferences(exprCol("{{ 'topic' ~ x }}"), names), []);
  assert.deepEqual(columnReferences(exprCol("{{ topic }}"), names), ["topic"]);

  const { group, nodes } = sqlGroup();
  const prompt = byName(nodes, "sql_prompt");
  const sql = byName(nodes, "sql");
  const g = loadWired(
    withColumn(nodes, "sql", { prompt: "{% if sql_prompt == 'x' %}yes{% else %}no{% endif %}" }),
    group,
  );
  // What the canvas hands over for a selected node: the node and every wire touching it.
  const edges = g.edgeOrder.filter((id) => g.edges[id].target === sql.id || g.edges[id].source === sql.id);
  removeElementsWithReferences({ nodes: [sql.id], edges, groups: [], notes: [] });
  assert.equal(graph().nodes[sql.id], undefined);
  assert.ok(graph().nodes[prompt.id]);
});

test("an edit to another block keeps the seed exactly as saved, an HF seed's split and subset included", () => {
  const { group, nodes } = sqlGroup();
  const seed = {
    source: { seed_type: "hf", path: "nyu-mll/glue", split: "validation", subset: "sst2" },
    sampling_strategy: "ordered",
  };
  const saved = withConfig(group, { seed_config: seed });
  const view = studioView(saved, nodes);
  const edited = patchEdit(saved, nodes, view, ofKind(view, "model_provider").id, {
    endpoint: "https://example.com/v1",
  });
  assert.deepEqual((edited.group?.config as Obj).seed_config, seed);
});

test("a rename leaves an attribute of the same name alone, even after a spaced dot", () => {
  assert.equal(
    replaceRef(
      "{{ first_name }} / {{ user . first_name }} / {{ user.first_name }}",
      "first_name",
      "given_name",
    ),
    "{{ given_name }} / {{ user . first_name }} / {{ user.first_name }}",
  );
  assert.equal(
    renameRoot("user .\n first_name == first_name", "first_name", "x"),
    "user .\n first_name == x",
  );
});

test("columns reading each other are a loop even when a wired seed defers the server's check", () => {
  assert.deepEqual(columnLoop([exprCol("{{ right }}", "left"), exprCol("{{ left }}", "right")]), [
    "left",
    "right",
    "left",
  ]);
  assert.equal(columnLoop([exprCol("x", "a"), exprCol("{{ a }}", "b")]), null);
});

test("the starter recipe passes Check as made, writing with Chat's loaded model", () => {
  const { group, nodes } = starterRecipeGraph([0, 0], {});
  assert.deepEqual(
    clientErrors(
      group,
      nodes.filter((n: FlowNode) => n.type === COLUMN_TYPE),
    ),
    [],
  );
  // Its card still reads as Chat's model, not the word "local".
  const models = recipeSetup(load(nodes, [group]), group).cards.filter((c) => c.kind === "model_config");
  assert.ok(models.length > 0);
  for (const card of models)
    assert.equal(card.fields.find((f) => f.label === "Model")?.value, "The loaded chat model");
});

test("recipe blocks pasted outside a recipe frame are left out, and a shared tool server edit reaches every profile", () => {
  const { group, nodes } = sqlGroup();
  const g = load(nodes, [{ ...group, rect: WIDE }]);
  const before = g.nodeOrder.length;
  // Far outside the frame.
  useDocStore.getState().insertFragment(cloneFragment(g, [nodes[0].id], [9000, 9000]));
  assert.equal(graph().nodeOrder.length, before);

  const server = {
    id: "s1",
    name: "docs",
    provider_type: "streamable_http",
    endpoint: "https://a.example/mcp",
  };
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
  const configs = {
    c1: { id: "c1", kind: "llm", name: "sql" },
    p1: { id: "p1", kind: "model_provider", name: "local" },
  } as never;
  // One namespace for every block, as the importer reads them, compared trimmed.
  assert.equal(nameTaken(configs, "c1", "local"), true);
  assert.equal(nameTaken(configs, "c1", " local "), true);
  assert.equal(nameTaken(configs, "c1", "sql"), false, "its own name is free");
  assert.equal(nameTaken(configs, "c1", "fresh"), false);
});

test("moving a column wire to another column moves the reference, in one undo step", () => {
  useCatalogStore.getState().setCatalog(catalog);
  const { group, nodes } = sqlGroup();
  const before = loadWired(nodes, group);
  const [prompt, sql, validator] = ["sql_prompt", "sql", "sql-validator"].map((n) => byName(nodes, n).id);
  const id = before.edgeOrder.find(
    (e) => before.edges[e].source === prompt && before.edges[e].target === sql,
  );
  assert.ok(id);
  const old = {
    id,
    source: prompt,
    target: sql,
    sourceHandle: outHandle("out"),
    targetHandle: inHandle("in"),
  };
  const moved = {
    source: prompt,
    sourceHandle: outHandle("out"),
    target: validator,
    targetHandle: inHandle("in"),
  };
  assert.equal(reconnectWire(old as never, moved), null);
  const wires = Object.values(graph().edges).filter((e) => e.source === prompt);
  assert.ok(wires.some((e) => e.target === validator));
  assert.ok(!wires.some((e) => e.target === sql), "the old wire is gone");
  assert.doesNotMatch(columnText(graph().nodes[sql]), /sql_prompt/);
  useDocStore.getState().undo();
  assert.match(columnText(graph().nodes[sql]), /sql_prompt/);
});

test("renaming a column renames it in every image context an AI column keeps", () => {
  const { group, nodes } = sqlGroup();
  const image = (name: string) => ({ modality: "image", column_name: name, data_type: "url" });
  const withImages = withColumn(nodes, "sql", {
    multi_modal_context: [image("sql_prompt"), image("domain")],
  });
  const view = studioView(group, withImages);
  const rename = (from: string, to: string) => {
    const edit = patchEdit(group, withImages, view, named(view, from).id, { name: to });
    const sql = edit.nodes.find((n) => n.id === byName(withImages, "sql").id);
    assert.ok(sql, `${from} renamed in sql`);
    return column(sql).multi_modal_context;
  };
  assert.deepEqual(rename("sql_prompt", "question"), [image("question"), image("domain")]);
  // The second image is not on the form: it is renamed in place.
  assert.deepEqual(rename("domain", "field"), [image("sql_prompt"), image("field")]);
});

test("a recipe block dragged out of its frame stays in its recipe, and the frame grows to hold it", () => {
  const { group, nodes } = sqlGroup();
  const g = load([...nodes, node("free", "input.text")], [group]);
  assert.equal(recipeDrag.ownRecipe(g, nodes[0].id), "recipe");
  assert.equal(recipeDrag.ownRecipe(g, "free"), null);
  const grown = recipeDrag.holdRecipeBlocks(group as never, { [nodes[0].id]: [2000, 2000] }, new Map());
  assert.deepEqual(grown.rect, [0, 0, 2284, 2144]);
  assert.equal(recipeDrag.holdRecipeBlocks(grown, { [nodes[0].id]: [100, 100] }, new Map()), grown);
});

test("a duplicated column skips names the recipe's setup blocks hold, and a copied reader sees its copied image", () => {
  const { group: small, nodes: base } = sqlGroup();
  const presets = (small.config.model_configs as Obj[]).map((m, i) =>
    i === 0 ? { ...m, alias: "domain_2" } : m,
  );
  const group = { ...withConfig(small, { model_configs: presets }), rect: WIDE };
  const images = [{ modality: "image", column_name: "domain", data_type: "url" }];
  const nodes = withColumn(base, "sql_prompt", { multi_modal_context: images });
  load(nodes, [group]);
  const fragment = cloneFragment(
    graph(),
    [byName(nodes, "domain").id, byName(nodes, "sql_prompt").id],
    [40, 40],
  );
  useDocStore.getState().insertFragment(fragment);
  const g = graph();
  // domain_2 is a model preset's alias: the copy takes domain_3.
  assert.equal(columnName(g.nodes[fragment.nodes[0].id]), "domain_3");
  const copy = column(g.nodes[fragment.nodes[1].id]);
  assert.deepEqual(copy.multi_modal_context, [{ ...images[0], column_name: "domain_3" }]);
  const members = Object.values(g.nodes).filter((n) => n.group === "recipe");
  assert.equal(studioView(g.groups.recipe, members).errors.length, 0);
});

test("a formula subtracting two columns reads both, as the backend's import does", () => {
  const net = exprCol("{{gross-discount}}", "net");
  assert.deepEqual(columnReferences(net, new Set(["gross", "discount"])), ["gross", "discount"]);
  // A column whose name has a hyphen is still read whole.
  assert.deepEqual(columnReferences(net, new Set(["gross-discount"])), ["gross-discount"]);
});

test("a loop's own variable is not a column, inside its loop only", () => {
  const names = new Set(["topic", "topics"]);
  const loop = exprCol("{% for topic in ['math', 'science'] %}{{ topic }} {% endfor %}");
  assert.deepEqual(columnReferences(loop, names), []);
  assert.equal(templateStillReads(loop, "topic"), false);
  const outside = exprCol("{{ topic }} {% for topic in topics %}{{ topic }}{% endfor %}");
  assert.deepEqual(columnReferences(outside, names), ["topic", "topics"]);
  assert.deepEqual(columnReferences(exprCol("{{ topic }}{% set topic = 'x' %}{{ topic }}"), names), [
    "topic",
  ]);
  // The right side of a `set` and a loop's iterable read the scope before the name is bound.
  const reassigned = exprCol("{% set topic = topic | upper %}{{ topic }}");
  assert.deepEqual(columnReferences(reassigned, names), ["topic"]);
  assert.equal(templateStillReads(reassigned, "topic"), true);
  assert.deepEqual(columnReferences(exprCol("{% for topic in topic %}{{ topic }}{% endfor %}"), names), [
    "topic",
  ]);
  assert.deepEqual(columnReferences(exprCol("{% set label = 'x' %}{{ label }}"), new Set(["label"])), []);
});

test("resizing a recipe frame or grouping its blocks never takes them out of their recipe", () => {
  const { group, nodes } = sqlGroup();
  load([...nodes, node("free", "input.text", [5000, 5000])], [{ ...group, rect: WIDE }]);
  const store = useDocStore.getState();
  store.resizeGroup({ ...graph().groups.recipe, rect: [-100, -100, 400, 200] });
  assert.ok(nodes.every((n) => graph().nodes[n.id].group === "recipe"));
  store.groupNodes({ id: "plain", label: "Group", rect: [0, 0, 10, 10] }, [nodes[0].id, "free"]);
  assert.equal(graph().nodes[nodes[0].id].group, "recipe");
  assert.equal(graph().nodes.free.group, "plain");
});

// Template renames of `topic` to `subject`: [test name, [template, expected][]].
const renameCases: [string, [string, string][]][] = [
  [
    "a rename leaves the template's own variable of that name alone",
    [
      [
        '{{ topic }} / {% set topic = "fixed" %}{{ topic | upper }}',
        '{{ subject }} / {% set topic = "fixed" %}{{ topic | upper }}',
      ],
      ["{% set topic = topic | upper %}{{ topic }}", "{% set topic = subject | upper %}{{ topic }}"],
      [
        "{% for t in topic %}{{ t }}{% endfor %}{% if topic %}x{% endif %}",
        "{% for t in subject %}{{ t }}{% endfor %}{% if subject %}x{% endif %}",
      ],
    ],
  ],
  [
    "a set inside a loop ends with it",
    [
      [
        '{% for item in [1] %}{% set topic = "inside" %}{{ topic }}{% endfor %}{{ topic }}',
        '{% for item in [1] %}{% set topic = "inside" %}{{ topic }}{% endfor %}{{ subject }}',
      ],
    ],
  ],
  [
    "Jinja whitespace control still closes a loop",
    [
      [
        "{% for topic in items -%}{{ topic | upper }}{%- endfor %}{{ topic }}",
        "{% for topic in items -%}{{ topic | upper }}{%- endfor %}{{ subject }}",
      ],
    ],
  ],
  [
    "setting a namespace's attribute binds no variable of that name",
    [
      [
        "{% set ns = namespace() %}{% set ns.topic = 'local' %}{{ topic }}",
        "{% set ns = namespace() %}{% set ns.topic = 'local' %}{{ subject }}",
      ],
    ],
  ],
  [
    "a loop's else branch reads the enclosing scope, an if's else inside the loop does not",
    [
      [
        "{% for topic in [] %}{{ topic }}{%- else -%}{{ topic }}{% endfor %}",
        "{% for topic in [] %}{{ topic }}{%- else -%}{{ subject }}{% endfor %}",
      ],
      [
        "{% for topic in xs %}{% if x %}a{% else %}{{ topic }}{% endif %}{{ topic }}{% endfor %}{{ topic }}",
        "{% for topic in xs %}{% if x %}a{% else %}{{ topic }}{% endif %}{{ topic }}{% endfor %}{{ subject }}",
      ],
    ],
  ],
];
for (const [name, cases] of renameCases)
  test(name, () => {
    for (const [template, expected] of cases)
      assert.equal(replaceRef(template, "topic", "subject"), expected);
  });

test("a delete leaves the template's own variable of that name alone", () => {
  const reader = exprCol("{{ topic }} / {% for topic in ['one', 'two'] %}{{ topic }}{% endfor %}", "r");
  assert.equal(
    withoutReference(reader, "topic").expr,
    " / {% for topic in ['one', 'two'] %}{{ topic }}{% endfor %}",
  );
});

test("editing a sampler keeps the settings its form does not show, and a rename reaches a numeric sampler's conditions", () => {
  const { group, nodes } = sqlGroup();
  const price = {
    name: "price",
    column_type: "sampler",
    sampler_type: "uniform",
    params: { low: 0, high: 100, decimal_places: 2 },
    conditional_params: { "tier == 'premium'": { sampler_type: "uniform", low: 100, high: 200 } },
  };
  const all = [
    ...nodes,
    colNode("tier", category("tier", ["basic", "premium"]), [0, 400]),
    colNode("price", price, [320, 400]),
  ];
  const view = studioView(group, all);
  assert.deepEqual(view.errors, []);
  const edited = patchEdit(group, all, view, named(view, "price").id, { high: "200" });
  assert.deepEqual(column(byName(edited.nodes, "price")).params, { low: 0, high: 200, decimal_places: 2 });
  const renamed = patchEdit(group, all, view, named(view, "tier").id, { name: "plan" });
  assert.deepEqual(Object.keys(column(byName(renamed.nodes, "price")).conditional_params as object), [
    "plan == 'premium'",
  ]);
});

test("a loop's filter reads its own variable: a rename or a reference never reaches it", () => {
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
  assert.deepEqual(
    columnReferences(exprCol("{% for topic in [0, 1] if topic > 0 %}x{% endfor %}"), names),
    [],
  );
  assert.deepEqual(columnReferences(exprCol("{% for t in items if t > limit %}{% endfor %}"), names), [
    "items",
    "limit",
  ]);
});

test("a rename reaches an override's parent column", () => {
  const { group } = sqlGroup();
  const sub = (values: object) => ({ sampler_type: "subcategory", category: "domain", values });
  const subColumn = {
    name: "sub",
    column_type: "sampler",
    sampler_type: "subcategory",
    params: sub({ science: ["physics"], art: ["paint"] }),
    conditional_params: { "domain == 'science'": sub({ science: ["chem"], art: ["clay"] }) },
  };
  const nodes = [
    colNode("domain", category("domain", ["science", "art"]), [0, 0]),
    colNode("sub", subColumn, [320, 0]),
  ];
  const view = studioView({ ...group, config: { ...group.config } }, nodes);
  assert.deepEqual(view.errors, []);
  const edit = patchEdit(group, nodes, view, named(view, "domain").id, { name: "area" });
  assert.deepEqual(column(byName(edit.nodes, "sub")).conditional_params, {
    "area == 'science'": { ...sub({ science: ["chem"], art: ["clay"] }), category: "area" },
  });
});

test("a seed is not added while another block holds the name seed, which every seed opens as", () => {
  const { group, nodes } = sqlGroup();
  load(withColumn(nodes, "domain", { name: "seed" }), [group]);
  addRecipeBlock("recipe", "seed", "seed_local");
  const g = graph();
  assert.equal(g.groups.recipe.config?.seed_config, undefined);
  const view = studioView(
    g.groups.recipe,
    Object.values(g.nodes).filter((n) => n.group === "recipe"),
  );
  assert.deepEqual(view.errors, []);
});

test("a rename reaches a transform's Jinja but not a quoted key, and a paste renames readers only after their own recipe's copies", () => {
  const { group, nodes } = sqlGroup();
  const template = { text: "{{ domain }}", detail: '{{ record["domain"] }}' };
  const withTransform = withConfig(group, {
    processors: [{ processor_type: "schema_transform", name: "t", template }],
  });
  const view = studioView(withTransform, nodes);
  const edit = patchEdit(withTransform, nodes, view, named(view, "domain").id, { name: "area" });
  const processors = (edit.group?.config as { processors: { template: unknown }[] }).processors;
  assert.deepEqual(processors[0].template, { text: "{{ area }}", detail: '{{ record["domain"] }}' });

  const make = (g: string, value: string, y: number) => [
    colNode(`${g}_t`, category("topic", [value]), [0, y], g),
    colNode(`${g}_q`, { ...(exprCol("{{ topic }}", "question") as object), dtype: "str" }, [320, y], g),
  ];
  const frame = (id: string, y: number) => ({ ...group, id, rect: [-100, y - 100, 900, 300] as Rect });
  const dest = { ...frame("dest", 1600), rect: [-100, 1400, 4000, 1000] as Rect };
  const g = load(
    [...make("a", "science", 0), ...make("b", "history", 400)],
    [frame("a", 0), frame("b", 400), dest],
  );
  const fragment = cloneFragment(g, ["a_t", "a_q", "b_t", "b_q"], [0, 1600]);
  useDocStore.getState().insertFragment(fragment);
  const [, aq, , bq] = fragment.nodes.map((n) => column(graph().nodes[n.id]));
  assert.equal(aq.expr, "{{ topic }}");
  assert.equal(bq.expr, "{{ topic_2 }}");
});

test("a recipe's model presets are in the download check", () => {
  const groups = {
    r: { config: { model_configs: [{ alias: "m", model: "unsloth/Gemma-GGUF", gguf_variant: "Q4_K_M" }] } },
  };
  const col = { type: "recipe.column", group: "r", params: { column: { model_alias: "m" } } };
  const step = { type: "recipe.generate", group: "r", params: {} };
  assert.equal(modelSignature({ c: col, s: step }, groups), "unsloth/gemma-gguf:q4_k_m");
  // Off until its Recipe Dataset step runs, or while no enabled column uses the preset.
  assert.equal(modelSignature({ c: col, s: { ...step, disabled: true } }, groups), "");
  assert.equal(modelSignature({ c: { ...col, disabled: true }, s: step }, groups), "");
  assert.equal(modelSignature({}), "");
});

test("with, macro and call blocks bind their own names, so a rename or a reference scan skips them", () => {
  const rename = (t: string) => replaceRef(t, "topic", "area");
  const refs = (expr: string, names = new Set(["topic", "words"])) =>
    columnReferences(exprCol(expr, "result"), names);
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
  assert.deepEqual(refs("{% with topic = 'x' %}{{ topic }}{% endwith %}{{ words }}"), ["words"]);
  assert.deepEqual(refs("{% macro show(topic) %}{{ topic }}{% endmacro %}{{ show(words) }}"), ["words"]);
  // A macro's own name is the template's from its declaration on, not a same-named column.
  assert.equal(
    replaceRef(
      "{{ format }}{% macro format(value) %}{{ value | upper }}{% endmacro %}{{ format(topic) }}",
      "format",
      "display_format",
    ),
    "{{ display_format }}{% macro format(value) %}{{ value | upper }}{% endmacro %}{{ format(topic) }}",
  );
  // A macro's parameter defaults read the column; its parameter names do not.
  assert.equal(
    rename("{% macro show(value=topic, n) %}{{ value }}{% endmacro %}{{ show() }}"),
    "{% macro show(value=area, n) %}{{ value }}{% endmacro %}{{ show() }}",
  );
  assert.deepEqual(refs("{% macro show(value=topic) %}{{ value }}{% endmacro %}{{ show() }}"), ["topic"]);
  // A `set` inside an `if` may not run: the read after it can be the column, so both are renamed.
  assert.equal(
    rename('{% if u %}{% set topic = "o" %}{{ topic }}{% endif %}{{ topic }}'),
    '{% if u %}{% set area = "o" %}{{ area }}{% endif %}{{ area }}',
  );
  assert.deepEqual(refs('{% if u %}{% set topic = "o" %}{% endif %}{{ topic }}'), ["topic"]);
  assert.equal(rename("{% set topic = topic %}{{ topic }}"), "{% set topic = area %}{{ topic }}");
  // A raw block is literal text: no rename, no reference, no removal inside it.
  assert.equal(
    rename("{%- raw -%}{{ topic | upper }}{% endraw %} {{ topic }}"),
    "{%- raw -%}{{ topic | upper }}{% endraw %} {{ area }}",
  );
  assert.deepEqual(refs("{% raw %}{{ topic }}{% endraw %}{{ words }}"), ["words"]);
  assert.equal(
    withoutReference(exprCol("{% raw %}{{ topic }}{% endraw %}\n{{ topic }}", "result"), "topic").expr,
    "{% raw %}{{ topic }}{% endraw %}",
  );
  assert.deepEqual(
    refs(
      "{% macro format(value) %}{{ value }}{% endmacro %}{{ format(topic) }}",
      new Set(["topic", "format"]),
    ),
    ["topic"],
  );
});

test("a duplicated reader gets the wire to the column it reads", () => {
  const { group, nodes } = sqlGroup();
  loadWired(nodes, group, WIDE);
  const source = byName(nodes, "sql_prompt");
  const fragment = cloneFragment(graph(), [byName(nodes, "sql").id], [40, 40]);
  useDocStore.getState().insertFragment(fragment);
  const copy = fragment.nodes[0].id;
  assert.ok(Object.values(graph().edges).some((e) => e.source === source.id && e.target === copy));
  removeNodes([source.id]);
  assert.doesNotMatch(columnText(graph().nodes[copy]), /\{\{ sql_prompt \}\}/);
});

const jinjaTestCases: [string, string[], string[]][] = [
  ["{{ value is string }}", ["value", "string", "number"], ["value"]],
  ["{{ value is not number }}", ["value", "string", "number"], ["value"]],
  ["{{ words | sort(reverse=true) }}", ["words", "reverse"], ["words"]],
  ["{{ words | sort(attribute=reverse) }}", ["words", "reverse"], ["words", "reverse"]],
  ["{{ reverse == words }}", ["words", "reverse"], ["reverse", "words"]],
];
for (const [expr, names, reads] of jinjaTestCases)
  test(`a Jinja test or keyword argument is not a column: ${expr} reads ${reads.join(", ")}`, () => {
    assert.deepEqual(columnReferences(exprCol(expr, "result"), new Set(names)), reads);
  });

for (const design of [true, false])
  test(`a ${design ? "recipe column's" : "runnable block's"} toolbar ${design ? "has no" : "has"} Run and Regenerate`, () => {
    const type = design ? COLUMN_TYPE : "input.text";
    assert.equal(!!catalog.nodes.find((s) => s.type === type)?.design, design);
    const table = moduleStubs.stubEveryImport({
      react: fakeReact({ runEffects: false }).react,
      "react/jsx-runtime": moduleStubs.stubJsxRuntime(),
      "@xyflow/react": {
        NodeToolbar: "NodeToolbar",
        Position: { Top: "top", Bottom: "bottom" },
        useReactFlow: () => ({ deleteElements: () => {} }),
        useStore: () => false,
      },
      "./canvas-rules": { toolbarNodeId: () => "n1" },
      "../stores/ui-store": selectorStore({ selectedNodes: ["n1"], inspectorNodeId: null }),
      "../stores/doc-store": { useDocStore: selectorStore({ graph: { nodes: { n1: { id: "n1", type } } } }) },
      "../stores/catalog-store": {
        useCatalogStore: selectorStore({ specs: Object.fromEntries(catalog.nodes.map((s) => [s.type, s])) }),
      },
      "../stores/runtime-store": { useRuntimeStore: selectorStore({ nodes: {} }) },
    });
    (table["../stores/ui-store"] as Obj).useUiStore = table["../stores/ui-store"];
    const { SelectedNodeToolbar } = moduleStubs.loadWithStubs<{
      SelectedNodeToolbar: (p: { readOnly: boolean }) => unknown;
    }>(new URL("../src/features/flows/editor/node-toolbar.tsx", import.meta.url), table);
    const ids = moduleStubs
      .elementsOf(SelectedNodeToolbar({ readOnly: false }))
      .map((e) => e.props["data-testid"])
      .filter(Boolean);
    assert.ok(ids.includes("node-toolbar-settings"), ids.join(","));
    assert.equal(ids.includes("node-toolbar-run"), !design);
    assert.equal(ids.includes("node-toolbar-reroll"), !design);
  });

test("a block dropped in a recipe frame's setup-card space joins that frame", async () => {
  const table = moduleStubs.stubEveryImport({
    react: fakeReact({ runEffects: false }).react,
    "react/jsx-runtime": moduleStubs.stubJsxRuntime(),
    "../stores/doc-store": { useDocStore },
    "./super-cards": await import("../src/features/flows/editor/super-cards.ts"),
    "./derive-rf": deriveRf,
    "./recipe-drag": recipeDrag,
  });
  type Pos = Record<string, [number, number]>;
  type Size = { width: number; height: number };
  const { groupMembership } = moduleStubs.loadWithStubs<{
    groupMembership: (
      p: Pos,
      sizes: Map<string, Size>,
      heights?: Record<string, number>,
    ) => Record<string, string | null>;
  }>(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), table);
  const { group, nodes } = sqlGroup();
  load([...nodes, node("free", "input.text", [0, 0])], [{ ...group, rect: [0, 0, 1000, 300] }]);
  // Below the saved height, inside the height the frame is drawn at with its setup cards.
  const drop: Pos = { free: [100, 500] };
  assert.deepEqual(groupMembership(drop, new Map()), { free: null });
  assert.deepEqual(groupMembership(drop, new Map(), { recipe: 800 }), { free: "recipe" });
  assert.deepEqual(groupMembership({ free: [100, 900] }, new Map(), { recipe: 800 }), { free: null });
});

async function renderProcessorsForm() {
  const harness = fakeReact({ runEffects: false });
  const table = moduleStubs.stubEveryImport({
    react: harness.react,
    "react/jsx-runtime": moduleStubs.stubJsxRuntime(),
    "../stores/doc-store": {
      ...(await import("../src/features/flows/stores/doc-store.ts")),
      useDocStore: Object.assign(
        <T>(select: (s: ReturnType<typeof useDocStore.getState>) => T) => select(useDocStore.getState()),
        { getState: useDocStore.getState },
      ),
    },
    "./recipe-actions": await import("../src/features/flows/recipe/recipe-actions.ts"),
    "./recipe-adapter": await import("../src/features/flows/recipe/recipe-adapter.ts"),
    "./recipe-blocks": await import("../src/features/flows/recipe/recipe-blocks.ts"),
    "./recipe-columns": await import("../src/features/flows/recipe/recipe-columns.ts"),
  });
  const { ProcessorsForm } = moduleStubs.loadWithStubs<{
    ProcessorsForm: (p: { groupId: string }) => unknown;
  }>(new URL("../src/features/flows/recipe/recipe-forms.tsx", import.meta.url), table);
  const Textarea = (table["@/components/ui/textarea"] as Obj).Textarea;
  const template = (groupId: string) => {
    const area = moduleStubs
      .elementsOf(harness.render(ProcessorsForm, { groupId }))
      .find((e) => e.type === Textarea);
    assert.ok(area, `a template box for ${groupId}`);
    return {
      value: area.props.value as string,
      type: (text: string) =>
        (area.props.onChange as (e: { target: { value: string } }) => void)({ target: { value: text } }),
    };
  };
  const saved = (groupId: string) =>
    (graph().groups[groupId].config as { processors: { template: unknown }[] }).processors[0].template;
  const transform = (template: object) => ({ processor_type: "schema_transform", name: "shape", template });
  const a = sqlGroup();
  const b = sqlGroup();
  load(
    [...a.nodes, ...b.nodes.map((n) => ({ ...n, id: `b_${n.id}`, group: "recipe_b" }))],
    [
      withConfig(a.group, { processors: [transform({ a: "{{ sql }}" })] }),
      { ...withConfig(b.group, { processors: [transform({ b: "{{ sql }}" })] }), id: "recipe_b" },
    ],
  );
  return { template, saved };
}

test("the schema transform form edits the group it shows now, after a switch or Undo", async () => {
  const { template, saved } = await renderProcessorsForm();
  assert.deepEqual(JSON.parse(template("recipe").value), { a: "{{ sql }}" });
  template("recipe").type('{"a": "{{ sql_prompt }}"}');
  assert.deepEqual(saved("recipe"), { a: "{{ sql_prompt }}" });
  assert.deepEqual(saved("recipe_b"), { b: "{{ sql }}" });
  // Undo: the form shows the document's template again, and the next edit starts from it.
  useDocStore.getState().undo();
  assert.deepEqual(JSON.parse(template("recipe").value), { a: "{{ sql }}" });
  // Switched to another group: it shows and edits that group.
  assert.deepEqual(JSON.parse(template("recipe_b").value), { b: "{{ sql }}" });
  template("recipe_b").type('{"b": 1}');
  assert.deepEqual(saved("recipe_b"), { b: 1 });
  assert.deepEqual(saved("recipe"), { a: "{{ sql }}" });
});

test("a schema transform template that is not JSON yet stays a draft in its own group's form", async () => {
  const { template, saved } = await renderProcessorsForm();
  template("recipe").type('{"a": ');
  assert.equal(template("recipe").value, '{"a": ');
  assert.deepEqual(saved("recipe"), { a: "{{ sql }}" }, "the recipe keeps its last valid template");
  // The draft belongs to the group it was typed in.
  assert.deepEqual(JSON.parse(template("recipe_b").value), { b: "{{ sql }}" });
});
