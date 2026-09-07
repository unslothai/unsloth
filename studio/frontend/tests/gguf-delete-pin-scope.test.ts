// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { pinKey, pinnedQuantEntries } from "../src/features/model-picker/components/model-selector/pinned-models.ts";
import { ggufVariantsMatchForPicker, modelIdsMatchForPicker } from "../src/features/model-picker/components/model-selector/row-identity.ts";

function readSource(path: string) {
  const text = readFileSync(new URL(path, import.meta.url), "utf8");
  return ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
}

function compile(expression: string, context: Record<string, unknown>) {
  const js = ts.transpileModule(`const callback = ${expression};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText;
  return new Function(...Object.keys(context), js + "; return callback;")(...Object.values(context));
}

const helper = readSource("../src/features/model-picker/components/model-selector/reconcile-gguf-pins.ts");
function helperFunction(name: string) {
  const declaration = helper.statements.find(
    (node) => ts.isFunctionDeclaration(node) && node.name?.text === name,
  );
  assert.ok(declaration, name);
  return declaration.getText(helper).replace("export ", "");
}
const isChatGgufTask = compile(helperFunction("isChatGgufTask"), {});
const callbacks: { name: string; body: string; wholeRepo: boolean }[] = [];
for (const path of [
  "../src/features/model-picker/components/model-selector/pickers.tsx",
  "../src/features/hub/catalog/models-catalog-rows.tsx",
]) {
  const source = readSource(path);
  function visit(node: ts.Node) {
    if (ts.isPropertyAssignment(node) && node.name.getText(source) === "onConfirm") {
      const body = node.initializer.getText(source);
      if (body.includes("reconcileGgufPinsAfterDelete")) {
        const name = path.includes("models-catalog") ? "catalog repo"
          : body.includes("onDeleteVariant(v.quant)") ? "expander quant"
          : body.includes("entry.repoId") ? "pinned quant"
          : body.includes("variant.quant") ? "sole quant" : "picker repo";
        callbacks.push({ name, body, wholeRepo: name.endsWith("repo") });
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}
assert.equal(callbacks.length, 5);

for (const task of [null, "text-generation", "image-text-to-text", "text-to-image", "text-to-video", "image-to-video", "text-to-speech", "automatic-speech-recognition"]) {
  const chat = task === null || task === "text-generation" || task === "image-text-to-text";
  for (const { name, body, wholeRepo } of callbacks) {
    test(`${name} keeps ${task ?? "unknown chat"} deletion pin scope`, async () => {
      const repoId = "Org/Model";
      const deleted = "Q6_K";
      const surviving = "Q8_0";
      let pinned = [repoId, pinKey(repoId, deleted), pinKey(repoId, surviving)];
      const togglePinned = (repo: string, quant?: string) => {
        const key = pinKey(repo, quant);
        pinned = pinned.includes(key) ? pinned.filter((item) => item !== key) : [...pinned, key];
      };
      const unpinRepo = () => { pinned = []; };
      const usePinnedModelsStore = { getState: () => ({ pinned, togglePinned, unpinRepo }) };
      const reconcile = compile(helperFunction("reconcileGgufPinsAfterDelete"), {
        fetchCachedGgufInventory: async () => ({ cached: [{ repo_id: repoId }], scan_confirmed: true }),
        // A media quant survives only in the inactive copy. Whole-row controls
        // also retain a second copy, where media previously cleared all repo pins.
        listGgufVariants: async () => ({ variants: chat || wholeRepo ? [{ quant: surviving, downloaded: true, partial: false }] : [] }),
        usePinnedModelsStore, pinnedQuantEntries, modelIdsMatchForPicker, ggufVariantsMatchForPicker,
      });
      let reconciliations = 0;
      const callback = compile(body, {
        repoId, hfToken: undefined, pipelineTag: task,
        v: { quant: deleted }, entry: { repoId, quant: deleted },
        c: { repo_id: repoId, task, cache_path: "/inactive/cache" },
        variant: { quant: deleted }, isPinned: true, pinnedKeys: [...pinned],
        row: { kind: "cache", isGguf: true, pipelineTag: task, cachePath: "/inactive/cache" },
        deletableRepoId: repoId, isDataset: false,
        diffusionTaskById: new Map([[repoId.toLowerCase(), task]]),
        mediaPageForTask: (value: string | null) => value && !["text-generation", "image-text-to-text"].includes(value) ? "images" : null,
        isChatGgufTask, usePinnedModelsStore, pinKey, togglePinned, togglePinnedQuant: togglePinned, unpinRepo,
        reconcileGgufPinsAfterDelete: async (...args: unknown[]) => { reconciliations++; await reconcile(...args); },
        onDeleteVariant: async () => {}, deleteCachedModel: async () => {}, deleteCachedDataset: async () => {},
        refreshCachedLists: () => {}, prunePinnedQuantValidation: () => {}, setRefreshKey: () => {},
      });
      await callback();
      assert.deepEqual(pinned, !chat && wholeRepo ? [] : [repoId, pinKey(repoId, surviving)]);
      assert.equal(reconciliations, chat ? 1 : 0);
    });
  }
}
