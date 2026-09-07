// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { modelIdsMatchForPicker } from "../src/features/model-picker/components/model-selector/row-identity.ts";
import { cachedGgufRowKey } from "../src/features/model-picker/components/model-selector/sole-quant-cache.ts";

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

test("Hub variant queries use the selected snapshot while file actions keep their root", () => {
  const card = source("features/hub/catalog/gguf-download-card.tsx");
  const resolve = new Function(
    "repoId",
    "loadId",
    "cachePath",
    `return (${expression(card, "localVariantPath")});`,
  );
  assert.equal(
    resolve(
      "Org/Model",
      "/hub/models--Org--Model/snapshots/old",
      "/hub/models--Org--Model",
    ),
    "/hub/models--Org--Model/snapshots/old",
  );
  assert.equal(
    resolve("Org/Model", "Org/Model", "/hub/models--Org--Model"),
    "/hub/models--Org--Model",
  );
  assert.equal(resolve("Org/Model", "Org/Model", null), null);
  assert.match(
    source("features/hub/catalog/model-inspector.tsx").text,
    /loadId=\{model\.resource\.runId\}/,
  );
  assert.match(
    source("features/hub/catalog/download-section.tsx").text,
    /loadId=\{loadId\}/,
  );
  assert.match(card.text, /cachePath=\{cachePath\}/);
});

test("an inactive resident copy does not lock updates to the active copy", () => {
  const picker = source(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  const disabled = new Function(
    "loadedModelId",
    "c",
    "copyMatches",
    `return (${expression(picker, "updateDisabled")});`,
  );
  assert.equal(disabled("Org/Model", { repo_id: "Org/Model" }, false), false);
  assert.equal(disabled("Org/Model", { repo_id: "Org/Model" }, true), true);
  assert.equal(disabled("Other/Model", { repo_id: "Org/Model" }, true), false);
});

test("keyboard focus follows the selected cache copy and visible pinned quant", () => {
  const picker = source(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  const resolve = new Function(
    "useMemo",
    "value",
    "hubOptionKeys",
    "visibleCachedGguf",
    "pinnedRows",
    "selectedLoadId",
    "selectedGgufVariant",
    "activeGgufVariant",
    "cachedGgufRowKey",
    "makeModelOptionKey",
    "modelIdsMatchForPicker",
    `return (${expression(picker, "selectedHubOptionKey")});`,
  );
  const rows = [
    { repo_id: "Org/Model", inventory_id: "copy1", load_id: "/first/snapshot" },
    {
      repo_id: "Org/Model",
      inventory_id: "copy2",
      load_id: "/second/snapshot",
    },
  ];
  const keys = [
    "downloaded-gguf::unrelated",
    "downloaded-gguf::copy1",
    "downloaded-gguf::copy2",
  ];
  const pinned = {
    key: "Org/Model::Q8_0",
    entry: { repoId: "Org/Model", quant: "Q8_0", loadId: "/second/snapshot" },
  };
  const select = (options: string[]) =>
    resolve(
      (fn: () => unknown) => fn(),
      "Org/Model",
      options,
      rows,
      [pinned],
      "/second/snapshot",
      "Q8_0",
      null,
      cachedGgufRowKey,
      (section: string, id: string) => `${section}::${id}`,
      modelIdsMatchForPicker,
    );
  assert.equal(select(keys), "downloaded-gguf::copy2");
  assert.equal(
    select(["pinned-quant::Org/Model::Q8_0", ...keys]),
    "pinned-quant::Org/Model::Q8_0",
  );
  assert.equal(select(["search-hf::Org/Model"]), "search-hf::Org/Model");
});

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
