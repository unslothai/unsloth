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
const { COLUMN_TYPE } = await import(
  "../src/features/flows/recipe/recipe-columns.ts"
);
const { runWithSettings, settingsFromRun } = await import(
  "../src/features/flows/recipe/recipe-run-settings.ts"
);
const { parseRecipeJson, recipeFileName } = await import(
  "../src/features/flows/recipe/recipe-io.ts"
);
const { buildDialogOptions } = await import(
  "../src/features/recipe-studio/utils/recipe-studio-view.ts"
);

type Payload = {
  recipe: Record<string, unknown> & { columns: Record<string, unknown>[] };
  ui?: Record<string, unknown>;
};

function learning(name: string): Payload {
  return JSON.parse(
    readFileSync(
      new URL(
        `../src/features/data-recipes/learning-recipes/${name}`,
        import.meta.url,
      ),
      "utf8",
    ),
  ) as Payload;
}

function group(payload: Payload) {
  const { columns, ...config } = structuredClone(payload.recipe);
  const nodes = columns.map((column, i) => ({
    id: `recipe_col_${i + 1}`,
    type: COLUMN_TYPE,
    v: 1,
    params: { column },
    pos: [i * 320, 0] as [number, number],
    label: String(column.name),
    group: "recipe",
  }));
  return {
    group: {
      id: "recipe",
      label: "Recipe",
      rect: [0, 0, 100, 100] as [number, number, number, number],
      kind: "recipe" as const,
      config,
      run: { rows: 5 },
      ui: structuredClone(payload.ui ?? {}),
    },
    nodes,
  };
}

function withTools(): Payload {
  const payload = learning("text-to-sql.json");
  payload.recipe.mcp_providers = [
    {
      name: "docs",
      provider_type: "streamable_http",
      endpoint: "https://mcp.example/mcp",
    },
  ];
  payload.recipe.tool_configs = [
    { tool_alias: "web", providers: ["docs"], max_tool_call_turns: 3 },
  ];
  payload.recipe.columns[4].tool_alias = "web";
  payload.ui = {
    nodes: [
      {
        id: "web",
        x: 0,
        y: 0,
        node_type: "tool_config",
        tools_by_provider: { docs: ["search"] },
      },
    ],
  };
  return payload;
}

test("a tool profile opens in Recipe Studio's dialog and the AI columns can pick it", () => {
  const { group: g, nodes } = group(withTools());
  const view = studioView(g, nodes);
  const tool = Object.values(view.configs).find((c) => c.kind === "tool_config");
  assert.ok(tool);
  assert.equal(tool.name, "web");
  assert.deepEqual(
    (tool as { fetched_tools_by_provider?: unknown }).fetched_tools_by_provider,
    { docs: ["search"] },
  );
  assert.deepEqual(
    buildDialogOptions(Object.values(view.configs)).toolProfileAliases,
    ["web"],
  );
});

test("editing a tool profile keeps it, its servers and its tool list on the group", () => {
  const { group: g, nodes } = group(withTools());
  const view = studioView(g, nodes);
  const tool = Object.values(view.configs).find((c) => c.kind === "tool_config");
  assert.ok(tool);
  const configs = applyConfigPatch(view.configs, tool.id, {
    max_tool_call_turns: "7",
  } as never);
  const edit = commitStudioEdit(g, nodes, view, configs);
  const config = edit.group?.config as Record<string, unknown>;
  assert.deepEqual(
    (config.mcp_providers as { name: string }[]).map((p) => p.name),
    ["docs"],
  );
  const tools = config.tool_configs as Record<string, unknown>[];
  assert.equal(tools[0].tool_alias, "web");
  assert.equal(tools[0].max_tool_call_turns, 7);
  const uiNodes = edit.group?.ui?.nodes as Record<string, unknown>[];
  assert.deepEqual(
    uiNodes.map((n) => n.node_type),
    ["tool_config"],
    "only tool profile nodes, never the canvas layout",
  );
});

test("a GitHub seed edited from Flows lands in the group's seed config", () => {
  const payload = learning("github-support-bot.json");
  const { group: g, nodes } = group(payload);
  const view = studioView(g, nodes);
  const seed = Object.values(view.configs).find((c) => c.kind === "seed");
  assert.ok(seed);
  assert.equal(
    (seed as { seed_source_type?: string }).seed_source_type,
    "github_repo",
  );
  const configs = applyConfigPatch(view.configs, seed.id, {
    github_repo_slug: "unslothai/unsloth\nunslothai/unsloth-zoo",
  } as never);
  const edit = commitStudioEdit(g, nodes, view, configs);
  const source = (edit.group?.config?.seed_config as Record<string, unknown>)
    .source as Record<string, unknown>;
  assert.equal(source.seed_type, "github_repo");
  assert.deepEqual(source.repos, ["unslothai/unsloth", "unslothai/unsloth-zoo"]);
});

test("an uploaded CSV seed keeps its path, file name and preview rows", () => {
  const payload = learning("text-to-sql.json");
  const { group: g, nodes } = group(payload);
  const view = studioView(g, nodes);
  // What the seed dialog writes after an upload is inspected.
  const add = {
    id: "seed-new",
    kind: "seed",
    name: "seed",
    seed_source_type: "local",
    hf_path: "/srv/uploads/seed.csv",
    local_file_name: "seed.csv",
    seed_columns: ["topic"],
    seed_preview_rows: [{ topic: "joins" }],
    sampling_strategy: "ordered",
    selection_type: "none",
  };
  const configs = { ...view.configs, [add.id]: add as never };
  const edit = commitStudioEdit(g, nodes, view, configs);
  const seed = edit.group?.config?.seed_config as Record<string, unknown>;
  assert.deepEqual(seed.source, {
    seed_type: "local",
    path: "/srv/uploads/seed.csv",
  });
  assert.equal(edit.group?.ui?.local_file_name, "seed.csv");
  assert.deepEqual(edit.group?.ui?.seed_preview_rows, [{ topic: "joins" }]);
  const reopened = studioView(edit.group as never, nodes);
  const again = Object.values(reopened.configs).find((c) => c.kind === "seed");
  assert.equal((again as { hf_path?: string }).hf_path, "/srv/uploads/seed.csv");
});

test("run settings round-trip through the group's run like Recipe Studio's dialog", () => {
  const defaults = settingsFromRun({ rows: 5 });
  assert.equal(defaults.batchEnabled, false);
  assert.equal(defaults.nonInferenceWorkers, 4);
  const run = runWithSettings(
    { rows: 5, output_formats: ["jsonl"] },
    {
      ...defaults,
      batchEnabled: true,
      batchSize: 250,
      mergeBatches: true,
      llmParallelRequests: 16,
      nonInferenceWorkers: 8,
      shutdownErrorRate: 0.25,
    },
    "nightly",
  );
  assert.deepEqual(run.output_formats, ["jsonl"]);
  assert.equal((run.run_config as Record<string, unknown>).buffer_size, 250);
  assert.equal(run.merge_batches, true);
  assert.equal(run.llm_parallel_requests, 16);
  assert.equal(run.run_name, "nightly");
  const back = settingsFromRun(run);
  assert.equal(back.batchEnabled, true);
  assert.equal(back.batchSize, 250);
  assert.equal(back.mergeBatches, true);
  assert.equal(back.llmParallelRequests, 16);
  assert.equal(back.shutdownErrorRate, 0.25);
  const off = runWithSettings(
    run,
    { ...back, batchEnabled: false, llmParallelRequests: null },
    "",
  );
  assert.equal("buffer_size" in (off.run_config as object), false);
  assert.equal(off.merge_batches, false);
  assert.equal("llm_parallel_requests" in off, false);
  assert.equal("run_name" in off, false);
});

test("recipe JSON is told apart from a flow and from junk", () => {
  const sql = learning("text-to-sql.json");
  assert.ok("payload" in parseRecipeJson(JSON.stringify(sql)));
  assert.ok("payload" in parseRecipeJson(JSON.stringify(sql.recipe)));
  assert.match(
    (parseRecipeJson('{"format": "unsloth-flow", "nodes": []}') as { error: string })
      .error,
    /flow/,
  );
  assert.ok("error" in parseRecipeJson("{nope"));
  assert.ok("error" in parseRecipeJson('{"recipe": {}}'));
  assert.equal(recipeFileName("Text to SQL"), "text-to-sql.recipe.json");
  assert.equal(recipeFileName(""), "recipe.recipe.json");
});
