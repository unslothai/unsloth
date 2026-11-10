// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readFlowsSrc } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { readRunPage, runsAreLive, savedRunDataset, selectedRun, toRunRecords, visibleRuns } = await import(
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

test("the Runs view stays a lazy chunk of the editor page", () => {
  assert.match(
    readFlowsSrc("editor/flows-editor-page.tsx"),
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
});

/** Page readers that log what they were asked; the job reader fails when `jobGone`. */
function pageReaders(jobGone: boolean) {
  const calls: string[] = [];
  return {
    calls,
    read: {
      job: async (jobId: string, page: { limit: number; offset: number }) => {
        calls.push(`job:${jobId}:${page.offset}:${page.limit}`);
        if (jobGone) throw new Error("job gone");
        return { dataset: [{ from: "job" }], total: 20 };
      },
      saved: async (value: { ref: string }, offset: number, limit: number) => {
        calls.push(`saved:${value.ref}:${offset}:${limit}`);
        return { rows: [{ from: "saved" }], total: 500, view_rows: 100 };
      },
    },
  };
}

test("a run page reads the live job while it is there", async () => {
  const { calls, read } = pageReaders(false);
  const page = await readRunPage(toRunRecords([backend("a", 1, "completed")])[0], 20, 10, read);
  assert.deepEqual(page, { dataset: [{ from: "job" }], total: 20 });
  assert.deepEqual(calls, ["job:job-a:20:10"]);
});

test("a run page falls back to the saved parquet, capped at the reader's view window", async () => {
  const { calls, read } = pageReaders(true);
  const page = await readRunPage(toRunRecords([backend("a", 1, "completed")])[0], 20, 10, read);
  assert.deepEqual(page, { dataset: [{ from: "saved" }], total: 100 });
  assert.deepEqual(calls, ["job:job-a:20:10", "saved:recipes/run/parquet-files:20:10"]);
});

test("a schema transform's run page never reads the job's untransformed rows", async () => {
  const { calls, read } = pageReaders(false);
  const record = { ...toRunRecords([backend("t", 1, "completed")])[0], output_ref: "recipes/run-9/processors-files/chat" };
  const page = await readRunPage(record, 0, 20, read);
  assert.deepEqual(page.dataset, [{ from: "saved" }]);
  assert.deepEqual(calls, ["saved:recipes/run-9/processors-files/chat:0:20"]);
});

test("a run page with no live job and nothing saved fails", async () => {
  const { read } = pageReaders(true);
  const record = toRunRecords([{ ...backend("p", 1, "active"), artifact_path: null }])[0];
  await assert.rejects(readRunPage(record, 0, 20, read), /job gone/);
});

const stubs = await import("./helpers/module-stubs.ts");
const { flushMicrotasks } = await import("./helpers/flows.ts");
const { fakeReact } = await import("./helpers/fake-react.ts");
const runsState = await import("../src/features/flows/recipe/recipe-runs-state.ts");
const executionHelpers = await import("../src/features/recipe-studio/executions/execution-helpers.ts");
const executionsViewHelpers = await import(
  "../src/features/recipe-studio/components/executions/executions-view-helpers.ts"
);

function loadDownloadRun() {
  const calls: unknown[][] = [];
  const table = stubs.stubEveryImport({
    react: fakeReact({ runEffects: false }).react,
    "react/jsx-runtime": stubs.stubJsxRuntime(),
    "./recipe-runs-state": runsState,
    "@/features/recipe-studio/components/executions/download-execution-dataset": {
      downloadExecutionDataset: async (record: { id: string }) => {
        calls.push(["job export", record.id]);
        return "started";
      },
    },
    "@/lib/native-files": {
      downloadFile: async (body: unknown, name: string, type: string) => {
        calls.push(["file", body, name, type]);
      },
    },
    "../api": {
      datasetExport: async (value: { ref: string }) => `export of ${value.ref}`,
    },
  });
  const mod = stubs.loadWithStubs<{ downloadRun: (r: unknown) => Promise<string> }>(
    new URL("../src/features/flows/recipe/recipe-runs.tsx", import.meta.url),
    table,
  );
  return { downloadRun: mod.downloadRun, calls };
}

const TRANSFORM_REF = "recipes/run-9/processors-files/chat";
const downloadCases: {
  name: string;
  record: () => Record<string, unknown>;
  outcome: string;
  calls: unknown[][];
}[] = [
  {
    name: "a plain run downloads the job's own export",
    record: () => toRunRecords([backend("a", 1, "completed")])[0],
    outcome: "started",
    calls: [["job export", "a"]],
  },
  {
    name: "a schema transform's full run downloads its own saved output, not the job's",
    record: () => ({ ...toRunRecords([backend("t", 1, "completed")])[0], output_ref: TRANSFORM_REF, run_name: "my run" }),
    outcome: "saved",
    calls: [["file", `export of ${TRANSFORM_REF}`, "my-run.jsonl", "application/x-ndjson"]],
  },
  {
    name: "a schema transform's preview downloads the transformed rows it holds",
    record: () => ({
      ...toRunRecords([backend("p", 1, "completed", "preview")])[0],
      output_ref: TRANSFORM_REF,
      dataset: [{ text: "a" }, { text: "b" }],
    }),
    outcome: "saved",
    calls: [["file", '{"text":"a"}\n{"text":"b"}\n', "p.jsonl", "application/x-ndjson"]],
  },
];
for (const c of downloadCases)
  test(`download: ${c.name}`, async () => {
    const { downloadRun, calls } = loadDownloadRun();
    assert.equal(await downloadRun(c.record()), c.outcome);
    assert.deepEqual(calls, c.calls);
  });

function renderExecutions(record: Record<string, unknown>) {
  const downloads: unknown[] = [];
  const table = stubs.stubEveryImport({
    react: fakeReact({ runEffects: false }).react,
    "react/jsx-runtime": stubs.stubJsxRuntime(),
    "../../executions/execution-helpers": executionHelpers,
    "./executions-view-helpers": executionsViewHelpers,
  });
  const { ExecutionsView } = stubs.loadWithStubs<{ ExecutionsView: (p: unknown) => unknown }>(
    new URL("../src/features/recipe-studio/components/executions/executions-view.tsx", import.meta.url),
    table,
  );
  const tree = ExecutionsView({
    executions: [record],
    selectedExecutionId: record.id,
    currentSignature: "",
    onSelectExecution: () => {},
    onCancelExecution: () => {},
    onLoadDatasetPage: () => {},
    downloadDataset: async (execution: unknown) => {
      downloads.push(execution);
      return "saved";
    },
  });
  const dialogType = (table["./publish-execution-dialog"] as Record<string, unknown>).PublishExecutionDialog;
  const elements = stubs.elementsOf(tree);
  return {
    downloads,
    publishable: elements.find((e) => e.type === dialogType)?.props.execution ?? null,
    download: elements.find((e) => e.props["aria-label"] === "Download dataset"),
  };
}

test("a finished run can be published; a schema transform's run cannot", () => {
  const plain = toRunRecords([backend("a", 1, "completed")])[0];
  assert.equal(renderExecutions(plain).publishable, plain);
  const transformed = { ...plain, output_ref: TRANSFORM_REF };
  assert.equal(renderExecutions(transformed).publishable, null);
});

test("the executions view's Download hands the run to the Flows downloader", () => {
  const transformed = { ...toRunRecords([backend("t", 1, "completed")])[0], output_ref: TRANSFORM_REF };
  const view = renderExecutions(transformed);
  assert.ok(view.download, "Download is offered");
  (view.download.props.onClick as () => void)();
  assert.deepEqual(view.downloads, [transformed]);
});

test("Bring in my Data Recipes first moves recipes still saved in this browser", async () => {
  const order: string[] = [];
  let legacyDone!: () => void;
  const table = stubs.stubEveryImport({
    react: fakeReact({ runEffects: false }).react,
    "react/jsx-runtime": stubs.stubJsxRuntime(),
    sonner: { toast: { info: () => {}, error: () => {}, success: () => {} } },
    "@/features/data-recipes/data/legacy-import": {
      importLegacyRecipes: () => {
        order.push("legacy");
        return new Promise<void>((resolve) => {
          legacyDone = resolve;
        });
      },
    },
    "../api": {
      importAllRecipes: async () => {
        order.push("all");
        return { created: [], skipped: [], failed: [] };
      },
    },
  });
  const { FlowsGalleryPage } = stubs.loadWithStubs<{ FlowsGalleryPage: () => unknown }>(
    new URL("../src/features/flows/gallery/flows-gallery-page.tsx", import.meta.url),
    table,
  );
  const button = stubs
    .elementsOf(FlowsGalleryPage())
    .find((e) => e.props["data-testid"] === "flows-bring-recipes");
  assert.ok(button);
  (button.props.onClick as () => void)();
  await flushMicrotasks();
  assert.deepEqual(order, ["legacy"], "the server import waits for the browser's recipes");
  legacyDone();
  await flushMicrotasks();
  assert.deepEqual(order, ["legacy", "all"]);
});
