// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

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

test("dataset cache reconciliation only reruns for external inventory inputs", () => {
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
    ["inventoryVersion", "datasetSource", "dataset"],
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
