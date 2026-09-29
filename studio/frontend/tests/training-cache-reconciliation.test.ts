// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  DatasetCacheRejectionTracker,
  createDatasetCacheUsabilityIdentity,
  datasetCacheUsabilityIdentitiesEqual,
} = await import("../src/features/training/lib/dataset-cache-rejection.ts");
const { buildCachedInventoryRow } = await import(
  "../src/features/hub/inventory/view-models.ts"
);

const RECONCILIATION_PATH = fileURLToPath(
  new URL(
    "../src/features/studio/hooks/use-training-cache-reconciliation.ts",
    import.meta.url,
  ),
);

const STORE_PATH = fileURLToPath(
  new URL(
    "../src/features/training/stores/training-config-store.ts",
    import.meta.url,
  ),
);

const DATASETS_API_PATH = fileURLToPath(
  new URL("../src/features/training/api/datasets-api.ts", import.meta.url),
);

const START_FRESH_PATH = fileURLToPath(
  new URL(
    "../src/features/training/lib/start-fresh-training-run.ts",
    import.meta.url,
  ),
);

const PREVIEW_DIALOG_PATH = fileURLToPath(
  new URL(
    "../src/features/studio/sections/dataset-preview-dialog.tsx",
    import.meta.url,
  ),
);

const DATASET_SELECTOR_PATH = fileURLToPath(
  new URL(
    "../src/features/dataset-picker/components/dataset-selector.tsx",
    import.meta.url,
  ),
);

function parseSource(path: string): ts.SourceFile {
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TS,
  );
}

const reconciliationSource = parseSource(RECONCILIATION_PATH);

test("processed dataset load paths stay separate from management paths", () => {
  const row = buildCachedInventoryRow(
    {
      repo_id: "Org/Data",
      size_bytes: 256,
      cache_path: "/cache/datasets--Org--Data",
      load_cache_path: "/processed/Org___Data",
    },
    "unknown",
  );

  assert.equal(row.cachePath, "/cache/datasets--Org--Data");
  assert.equal(row.loadCachePath, "/processed/Org___Data");

  const selector = readFileSync(DATASET_SELECTOR_PATH, "utf8");
  const reconciliation = readFileSync(RECONCILIATION_PATH, "utf8");
  assert.match(
    selector,
    /localPath: cached\?\.loadCachePath \?\? cached\?\.cachePath \?\? null/,
  );
  assert.match(
    reconciliation,
    /return reference\.load_cache_path \?\? reference\.cache_path/,
  );
});

function findCall(
  root: ts.Node,
  predicate: (call: ts.CallExpression) => boolean,
): ts.CallExpression | null {
  let found: ts.CallExpression | null = null;
  const visit = (node: ts.Node): void => {
    if (found) {
      return;
    }
    if (ts.isCallExpression(node) && predicate(node)) {
      found = node;
      return;
    }
    node.forEachChild(visit);
  };
  root.forEachChild(visit);
  return found;
}

test("dataset cache reconciliation reruns for usability inputs but not cache status", () => {
  const datasetFetch = findCall(
    reconciliationSource,
    (call) =>
      call.expression.getText(reconciliationSource) ===
        "fetchInventorySource" &&
      ts.isStringLiteral(call.arguments[0]) &&
      call.arguments[0].text === "cachedDatasets",
  );
  assert.ok(datasetFetch, "cached dataset inventory fetch not found");

  const datasetEffect = findCall(
    reconciliationSource,
    (call) =>
      call.expression.getText(reconciliationSource) === "useEffect" &&
      call.arguments[0] != null &&
      datasetFetch.pos >= call.arguments[0].pos &&
      datasetFetch.end <= call.arguments[0].end,
  );
  assert.ok(datasetEffect, "cached dataset reconciliation effect not found");

  const dependencies = datasetEffect.arguments[1];
  assert.ok(
    dependencies && ts.isArrayLiteralExpression(dependencies),
    "dataset reconciliation dependencies not found",
  );
  assert.deepEqual(
    dependencies.elements.map((element) =>
      element.getText(reconciliationSource),
    ),
    [
      "inventoryVersion",
      "cachedDatasetRows",
      "datasetSource",
      "dataset",
      "datasetSubset",
      "datasetSplit",
      "datasetStreaming",
    ],
  );

  const options = datasetFetch.arguments[1];
  assert.ok(
    options && ts.isObjectLiteralExpression(options),
    "cached dataset inventory options not found",
  );
  assert.deepEqual(
    options.properties.map((property) =>
      property.getText(reconciliationSource),
    ),
    ["inventoryVersion"],
  );

  const stateRead = findCall(
    datasetEffect.arguments[0],
    (call) =>
      call.expression.getText(reconciliationSource) ===
      "useTrainingConfigStore.getState",
  );
  assert.ok(
    stateRead,
    "dataset cache state is not snapshotted inside the effect",
  );
});

function usabilityIdentity(
  overrides: Partial<{
    dataset: string;
    cachePath: string | null;
    subset: string | null;
    split: string | null;
    streaming: boolean;
  }> = {},
) {
  return createDatasetCacheUsabilityIdentity({
    dataset: "Org/Data",
    cachePath: "/cache/datasets--Org--Data",
    subset: "default",
    split: "train",
    streaming: false,
    ...overrides,
  });
}

const cachedRow = {
  cachePath: "/cache/datasets--Org--Data",
  sizeBytes: 128,
  partial: false,
  partialTransport: null,
};

function rejectCache(
  tracker: InstanceType<typeof DatasetCacheRejectionTracker>,
  identity: ReturnType<typeof usabilityIdentity>,
): void {
  if (!tracker.rejectValidation(tracker.beginValidation(identity))) {
    throw new Error("current cache validation was unexpectedly stale");
  }
}

test("an unchanged rejected inventory row stays rejected across reconciliation runs", () => {
  const tracker = new DatasetCacheRejectionTracker();
  const identity = usabilityIdentity();

  tracker.observe(identity, cachedRow);
  rejectCache(tracker, identity);

  assert.equal(tracker.shouldPromote(identity, cachedRow), false);
  assert.equal(tracker.shouldPromote(identity, cachedRow), false);
});

test("a relevant inventory identity change makes a rejected cache retryable", () => {
  const tracker = new DatasetCacheRejectionTracker();
  const identity = usabilityIdentity();

  tracker.observe(identity, cachedRow);
  rejectCache(tracker, identity);
  assert.equal(tracker.shouldPromote(identity, cachedRow), false);
  assert.equal(
    tracker.shouldPromote(identity, { ...cachedRow, sizeBytes: 256 }),
    true,
  );

  rejectCache(tracker, identity);
  const replacement = usabilityIdentity({
    cachePath: "/other-cache/datasets--Org--Data",
  });
  assert.equal(
    tracker.shouldPromote(replacement, {
      ...cachedRow,
      cachePath: "/other-cache/datasets--Org--Data",
    }),
    true,
  );
  assert.equal(tracker.shouldPromote(identity, cachedRow), true);
});

test("cache rejection is scoped to subset, split, and streaming mode", () => {
  const tracker = new DatasetCacheRejectionTracker();
  const rejected = usabilityIdentity();

  tracker.observe(rejected, cachedRow);
  rejectCache(tracker, rejected);

  assert.equal(
    tracker.shouldPromote(
      usabilityIdentity({ split: "validation" }),
      cachedRow,
    ),
    true,
  );
  assert.equal(
    tracker.shouldPromote(usabilityIdentity({ subset: "english" }), cachedRow),
    true,
  );
  assert.equal(
    tracker.shouldPromote(usabilityIdentity({ streaming: true }), cachedRow),
    true,
  );
  assert.equal(tracker.shouldPromote(rejected, cachedRow), false);

  tracker.reset("org/data");
  assert.equal(tracker.shouldPromote(rejected, cachedRow), true);
});

test("a cache request without a path binds rejection to the observed inventory row", () => {
  const tracker = new DatasetCacheRejectionTracker();
  const unpinned = usabilityIdentity({ cachePath: null });

  tracker.observe(usabilityIdentity(), cachedRow);
  rejectCache(tracker, unpinned);

  assert.equal(tracker.shouldPromote(usabilityIdentity(), cachedRow), false);
  assert.equal(
    tracker.shouldPromote(
      usabilityIdentity({ cachePath: "/replacement/datasets--Org--Data" }),
      {
        ...cachedRow,
        cachePath: "/replacement/datasets--Org--Data",
      },
    ),
    true,
  );
});

test("an explicit reset invalidates an older cache validation result", () => {
  const tracker = new DatasetCacheRejectionTracker();
  const identity = usabilityIdentity();
  const staleValidation = tracker.beginValidation(identity);

  tracker.reset("org/data");

  assert.equal(tracker.isValidationCurrent(staleValidation), false);
  assert.equal(tracker.rejectValidation(staleValidation), false);
  assert.equal(tracker.shouldPromote(identity, cachedRow), true);
});

test("a material inventory change invalidates an older cache validation result", () => {
  const tracker = new DatasetCacheRejectionTracker();
  const identity = usabilityIdentity();

  tracker.observe(identity, cachedRow);
  const staleValidation = tracker.beginValidation(identity);
  tracker.observe(identity, { ...cachedRow, sizeBytes: 256 });

  assert.equal(tracker.isValidationCurrent(staleValidation), false);
  assert.equal(tracker.rejectValidation(staleValidation), false);
  assert.equal(
    tracker.shouldPromote(identity, { ...cachedRow, sizeBytes: 256 }),
    true,
  );
});

test("usability identity equality supports exact stale-response guards", () => {
  const expected = usabilityIdentity();
  assert.equal(
    datasetCacheUsabilityIdentitiesEqual(
      expected,
      usabilityIdentity({ dataset: "org/data" }),
    ),
    true,
  );
  assert.equal(
    datasetCacheUsabilityIdentitiesEqual(
      expected,
      usabilityIdentity({ split: "validation" }),
    ),
    false,
  );
  assert.equal(
    datasetCacheUsabilityIdentitiesEqual(
      expected,
      usabilityIdentity({ cachePath: "/cache/other" }),
    ),
    false,
  );
});

test("dataset checks pass cancellation through to the request", () => {
  const storeSource = parseSource(STORE_PATH);
  const datasetCheck = findCall(
    storeSource,
    (call) => call.expression.getText(storeSource) === "checkDatasetFormat",
  );
  assert.ok(datasetCheck, "dataset format check not found");
  const checkOptions = datasetCheck.arguments[0];
  assert.ok(
    checkOptions && ts.isObjectLiteralExpression(checkOptions),
    "dataset format check options not found",
  );
  assert.ok(
    checkOptions.properties.some(
      (property) =>
        property.getText(storeSource) === "signal: controller.signal",
    ),
    "dataset format check does not receive the controller signal",
  );

  const apiSource = parseSource(DATASETS_API_PATH);
  const request = findCall(
    apiSource,
    (call) =>
      call.expression.getText(apiSource) === "authFetch" &&
      ts.isStringLiteral(call.arguments[0]) &&
      call.arguments[0].text === "/api/hub/datasets/check-format",
  );
  assert.ok(request, "dataset format request not found");
  const requestOptions = request.arguments[1];
  assert.ok(
    requestOptions && ts.isObjectLiteralExpression(requestOptions),
    "dataset format request options not found",
  );
  assert.ok(
    requestOptions.properties.some(
      (property) => property.getText(apiSource) === "signal",
    ),
    "dataset format request does not forward the signal",
  );
});

test("deleted local dataset handling covers background, preview, and start checks", () => {
  for (const path of [STORE_PATH, START_FRESH_PATH, PREVIEW_DIALOG_PATH]) {
    const source = parseSource(path);
    assert.ok(
      findCall(
        source,
        (call) => call.expression.getText(source) === "clearDeletedDataset",
      ),
      `${path} does not clear a backend-reported deleted dataset`,
    );
  }

  const previewSource = parseSource(PREVIEW_DIALOG_PATH);
  const previewCheck = findCall(
    previewSource,
    (call) => call.expression.getText(previewSource) === "checkDatasetFormat",
  );
  assert.ok(previewCheck, "preview dataset format check not found");
  const previewOptions = previewCheck.arguments[0];
  assert.ok(
    previewOptions && ts.isObjectLiteralExpression(previewOptions),
    "preview dataset format options not found",
  );
  assert.ok(
    previewOptions.properties.some(
      (property) =>
        property.getText(previewSource) === "signal: controller.signal",
    ),
    "preview dataset format request is not cancelled when superseded",
  );
});
