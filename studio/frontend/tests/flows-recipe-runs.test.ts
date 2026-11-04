// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { runsAreLive, savedRunDataset, selectedRun, toRunRecords, visibleRuns } = await import(
  "../src/features/flows/recipe/recipe-runs-state.ts"
);

// The shape `core/flows/nodes/data.py` RunHistory writes: no Recipe Studio-only fields.
const backend = (id: string, createdAt: number, status: string, kind = "full") => ({
  id,
  recipeId: "flow:f1:gen",
  jobId: `job-${id}`,
  kind,
  status,
  rows: 20,
  createdAt,
  finishedAt: status === "completed" ? createdAt + 5 : null,
  recipeSignature: "",
  dataset: [{ sql: "SELECT 1" }],
  datasetTotal: 20,
  artifact_path: status === "completed" ? "/x/recipes/run" : null,
  lastEventId: 3,
});

test("backend run records come out complete and newest first, running ones on top", () => {
  const records = toRunRecords([
    backend("a", 1, "completed"),
    backend("b", 3, "completed", "preview"),
    backend("c", 2, "active"),
  ]);
  assert.deepEqual(
    records.map((r) => r.id),
    ["c", "b", "a"],
  );
  const done = records[2];
  assert.deepEqual(done.completed_columns, []);
  assert.deepEqual(done.log_lines, []);
  assert.equal(done.datasetPage, 1);
  assert.equal(done.datasetPageSize, 20);
  assert.equal(done.batch, null);
  assert.equal(done.run_name, null);
  // What Recipe Studio's executions view needs to offer Publish and Download.
  assert.equal(done.jobId, "job-a");
  assert.equal(done.artifact_path, "/x/recipes/run");
});

test("Preview shows the newest run alone, Runs shows all of them", () => {
  const records = toRunRecords([
    backend("a", 1, "completed"),
    backend("b", 3, "error"),
    backend("c", 2, "active"),
  ]);
  assert.deepEqual(
    visibleRuns(records, "preview").map((r) => r.id),
    ["b"],
  );
  assert.equal(visibleRuns(records, "runs").length, 3);
  assert.deepEqual(visibleRuns([], "preview"), []);
});

test("the picked run stays selected while it is shown, else the first one is", () => {
  const records = toRunRecords([backend("a", 1, "completed"), backend("b", 2, "completed")]);
  assert.equal(selectedRun(records, "a"), "a");
  assert.equal(selectedRun(records, "gone"), "b");
  assert.equal(selectedRun(records, null), "b");
  assert.equal(selectedRun([], "a"), null);
});

test("runs refresh while the flow or a recipe job is still going", () => {
  const done = toRunRecords([backend("a", 1, "completed")]);
  assert.equal(runsAreLive(done, false), false);
  assert.equal(runsAreLive(done, true), true);
  assert.equal(runsAreLive(toRunRecords([backend("b", 1, "pending")]), false), true);
});

test("Preview and Runs open from the recipe panel and the run drawer; the frame runs the recipe", () => {
  const read = (path: string) =>
    readFileSync(new URL(`../src/features/flows/${path}`, import.meta.url), "utf8");
  for (const file of ["recipe/recipe-panels.tsx", "editor/run-drawer.tsx"])
    assert.match(read(file), /setRecipeRuns\(/, file);
  assert.match(read("editor/frame-nodes.tsx"), /runRecipeGroup\(/);
  assert.match(
    read("editor/flows-editor-page.tsx"),
    /lazy\(\(\) => import\("\.\.\/recipe\/recipe-runs"\)\)/,
  );
});

test("an older run pages from its saved parquet once its recipe job is gone", () => {
  assert.deepEqual(savedRunDataset({ artifact_path: "/home/u/.unsloth/studio/recipes/run-7" }), {
    kind: "dataset",
    source: "recipe",
    ref: "recipes/run-7/parquet-files",
  });
  assert.equal(savedRunDataset({ artifact_path: "C:\\x\\recipes\\run-8\\" })?.ref, "recipes/run-8/parquet-files");
  assert.equal(savedRunDataset({ artifact_path: null }), null);
  // A schema transform's run pages its own output, never the job's untransformed rows.
  const transformed = toRunRecords([
    { ...backend("t", 1, "completed"), artifact_path: "/x/recipes/run-9", output_ref: "recipes/run-9/processors-files/chat" },
  ])[0];
  assert.equal(savedRunDataset(transformed)?.ref, "recipes/run-9/processors-files/chat");
  assert.match(
    readFileSync(new URL("../src/features/flows/recipe/recipe-runs.tsx", import.meta.url), "utf8"),
    /record\.output_ref\s*\?\s*Promise\.reject/,
  );
  assert.match(readFileSync(new URL("../src/features/flows/recipe/recipe-runs.tsx", import.meta.url), "utf8"), /datasetRows\(saved, offset, size\)/);
  assert.match(readFileSync(new URL("../src/features/flows/recipe/recipe-runs.tsx", import.meta.url), "utf8"), /Math\.min\(read\.total, read\.view_rows\)/);
});

test("a schema transform's run downloads its own rows and is never published as the job's", () => {
  const runs = readFileSync(new URL("../src/features/flows/recipe/recipe-runs.tsx", import.meta.url), "utf8");
  assert.match(runs, /downloadDataset=\{downloadRun\}/);
  assert.match(runs, /if \(!record\.output_ref\) return downloadExecutionDataset\(record\);/);
  assert.match(runs, /datasetExport\(saved\)/);
  const view = readFileSync(
    new URL("../src/features/recipe-studio/components/executions/executions-view.tsx", import.meta.url),
    "utf8",
  );
  assert.match(view, /downloadDataset\(selectedExecution\)/);
  assert.match(view, /!selectedExecution\.output_ref/);
});

test("Bring in my Data Recipes first moves recipes still saved in this browser", () => {
  const gallery = readFileSync(new URL("../src/features/flows/gallery/flows-gallery-page.tsx", import.meta.url), "utf8");
  const bring = gallery.slice(gallery.indexOf("const bringRecipes"));
  assert.ok(bring.indexOf("await importLegacyRecipes()") > 0);
  assert.ok(bring.indexOf("await importLegacyRecipes()") < bring.indexOf("await importAllRecipes()"));
});
