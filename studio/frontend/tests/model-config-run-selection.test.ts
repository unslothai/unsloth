// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

import { isExternalModelId } from "../src/features/chat/external-providers.ts";
import { isLocalModelPath } from "../src/features/chat/utils/model-download-staging.ts";
import type {
  ModelPickTarget,
  ModelSelectorChangeMeta,
} from "../src/features/model-picker/components/model-selector/types.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
const { modelConfigTarget } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

function readSource(relativePath: string): ts.SourceFile {
  return ts.createSourceFile(
    relativePath,
    readFileSync(new URL(relativePath, import.meta.url), "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
}

function findNode<T extends ts.Node>(
  root: ts.Node,
  predicate: (node: ts.Node) => node is T,
): T {
  let found: T | undefined;
  function visit(node: ts.Node): void {
    if (predicate(node)) {
      found = node;
    }
    if (!found) {
      ts.forEachChild(node, visit);
    }
  }
  visit(root);
  assert.ok(found, "Expected source node was not found");
  return found;
}

// Execute the shipped callback and persistence guard without mounting the UI.
const selector = readSource(
  "../src/features/model-picker/components/model-selector.tsx",
);
const configPage = findNode(
  selector,
  (node): node is ts.JsxSelfClosingElement =>
    ts.isJsxSelfClosingElement(node) &&
    node.tagName.getText() === "ModelConfigPage",
);
const onRun = findNode(
  configPage,
  (node): node is ts.JsxAttribute =>
    ts.isJsxAttribute(node) && node.name.getText() === "onRun",
);
assert.ok(onRun.initializer && ts.isJsxExpression(onRun.initializer));
assert.ok(onRun.initializer.expression);
const runConfiguredModel = new Function(
  "visibleConfigTarget",
  "onSelect",
  "config",
  `return (${onRun.initializer.expression.getText()})(config, false);`,
) as (
  target: ModelPickTarget,
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void,
  config: typeof DEFAULT_PER_MODEL_CONFIG,
) => void;

const runtime = readSource(
  "../src/features/chat/hooks/use-chat-model-runtime.ts",
);
const declarations = [
  "modelId",
  "loadPath",
  "ggufVariant",
  "nativePathToken",
  "indexedLocalPick",
].map((name) => {
  const declaration = findNode(
    runtime,
    (node): node is ts.VariableDeclaration =>
      ts.isVariableDeclaration(node) && node.name.getText() === name,
  );
  return `const ${declaration.getText()};`;
});
const recordCall = findNode(
  runtime,
  (node): node is ts.CallExpression =>
    ts.isCallExpression(node) &&
    node.expression.getText() === "recordLastLocalModelLoad",
);
const recordGuard = recordCall.parent.parent.parent;
assert.ok(ts.isIfStatement(recordGuard));
type Selection = ModelSelectorChangeMeta & { id: string };
type RememberedModel = {
  id: string;
  kind: "gguf" | "model";
  ggufVariant: string | null;
};
const finishLoad = new Function(
  "selection",
  "loadResponse",
  "isGguf",
  "isLora",
  "recordLastLocalModelLoad",
  "isLocalModelPath",
  "isExternalModelId",
  `${declarations.join("\n")}\n${recordGuard.getText()}\nreturn loadPath;`,
) as (
  selection: Selection,
  response: { is_gguf: boolean; is_lora: boolean },
  isGguf: boolean,
  isLora: boolean,
  record: (model: RememberedModel) => void,
  isLocalPath: typeof isLocalModelPath,
  isExternalId: typeof isExternalModelId,
) => string;

const cases = [
  {
    name: "secondary GGUF cache",
    id: "Org/Model-GGUF",
    loadId: "/cache/models--Org--Model-GGUF/snapshots/revision",
    isGguf: true,
    ggufVariant: "Q4_K_M",
  },
  {
    name: "Windows GGUF cache",
    id: "Org/Model-GGUF",
    loadId: String.raw`C:\Cache\models--Org--Model-GGUF\snapshots\revision`,
    isGguf: true,
    ggufVariant: "Q8_0",
  },
  {
    name: "secondary safetensors cache",
    id: "Org/Model",
    loadId: "/cache/models--Org--Model/snapshots/revision",
    isGguf: false,
  },
  { name: "active Hub cache", id: "Org/Model", isGguf: false },
  {
    name: "indexed local GGUF",
    id: "/models/model.gguf",
    isGguf: true,
    source: "local" as const,
  },
  {
    name: "native file picker",
    id: "/models/picked.gguf",
    isGguf: true,
    source: "local" as const,
    nativePathToken: "signed-path-token",
  },
];

for (const entry of cases) {
  test(`configuration Run preserves loading and remembering for ${entry.name}`, () => {
    const { name: _name, id, ...fields } = entry;
    const meta: ModelSelectorChangeMeta = {
      source: "hub",
      isLora: false,
      isDownloaded: true,
      ...fields,
    };
    const target = modelConfigTarget(id, meta);
    const config = { ...DEFAULT_PER_MODEL_CONFIG, maxSeqLength: 8192 };
    const calls: Selection[] = [];
    runConfiguredModel(
      target,
      (selectedId, selectedMeta) => {
        calls.push({ id: selectedId, ...selectedMeta });
      },
      config,
    );

    assert.deepEqual(calls, [
      { id, ...meta, config, isDiffusion: false, forceReload: true },
    ]);
    const remembered: RememberedModel[] = [];
    const loadPath = finishLoad(
      calls[0],
      { is_gguf: entry.isGguf, is_lora: false },
      entry.isGguf,
      false,
      (model) => remembered.push(model),
      isLocalModelPath,
      isExternalModelId,
    );
    assert.equal(loadPath, meta.loadId ?? id);
    assert.deepEqual(
      remembered,
      meta.nativePathToken
        ? []
        : [
            {
              id,
              kind: entry.isGguf ? "gguf" : "model",
              ggufVariant: meta.ggufVariant ?? null,
            },
          ],
    );
  });
}
