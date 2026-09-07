// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

function source(path: string) {
  return ts.createSourceFile(
    path,
    readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
}

function expression(file: ts.SourceFile, name: string) {
  let result: ts.Expression | undefined;
  function visit(node: ts.Node) {
    if (
      (ts.isVariableDeclaration(node) || ts.isPropertyAssignment(node)) &&
      node.name.getText(file) === name &&
      node.initializer
    )
      result = node.initializer;
    ts.forEachChild(node, visit);
  }
  visit(file);
  assert.ok(result, name);
  return result.getText(file);
}

for (const activeCache of [false, true, undefined]) {
  test(`Update actions respect cache destination (${activeCache})`, () => {
    const picker = source(
      "features/model-picker/components/model-selector/pickers.tsx",
    );
    const update = new Function(
      "c",
      "updateGgufVariant",
      `return (${expression(picker, "onUpdate")});`,
    )({ active_cache: activeCache, repo_id: "Org/Model" }, () => undefined);
    assert.equal(
      typeof update === "function",
      activeCache !== false,
      "picker Update targets a different copy",
    );
    const card = source("features/hub/catalog/gguf-download-card.tsx");
    const available = new Function(
      "activeCache",
      "selected",
      `return (${expression(card, "updateAvailable")});`,
    )(activeCache, { downloaded: true, update_available: true });
    assert.equal(
      available,
      activeCache !== false,
      "Hub Update targets a different copy",
    );
    const local = source("features/hub/catalog/local-on-device-card.tsx");
    const canUpdate = new Function(
      "activeCache",
      "online",
      "source",
      "repoId",
      "isActive",
      "isLoading",
      "updateJobActive",
      "updateAvailable",
      `return (${expression(local, "canUpdate")});`,
    )(activeCache, true, "hf_cache", "Org/Model", false, false, false, true);
    assert.equal(
      canUpdate,
      activeCache !== false,
      "local card Update targets a different copy",
    );
  });
}

test("an open update dialog cannot update an inactive copy", () => {
  for (const name of ["gguf-download-card", "local-on-device-card"]) {
    const file = source(`features/hub/catalog/${name}.tsx`);
    const confirm = expression(file, "handleConfirmUpdate");
    const callback = new Function(
      "useCallback",
      "activeCache",
      "updateTarget",
      "updateTargetVariant",
      "repoId",
      "setUpdateTarget",
      "setUpdateOpen",
      "downloadManager",
      `return (${confirm});`,
    )(
      (fn: unknown) => fn,
      false,
      "Q8_0",
      "Q8_0",
      "Org/Model",
      () => assert.fail("stale confirmation ran"),
      () => assert.fail("stale confirmation ran"),
      { requestStart: () => assert.fail("wrong destination") },
    );
    callback();
  }
});
