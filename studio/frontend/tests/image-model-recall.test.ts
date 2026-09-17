// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import {
  matchesRememberedModel,
  readImageModel,
  rememberImageModel,
} from "../src/features/images/image-model-recall.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { storage } = installLocalStorageFake();

test("recalling a quantized model carries the selected adapters into its load", async () => {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile(
    "images-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const names = new Set([
    "bakedLorasFor",
    "currentLoadAdvanced",
    "handleGenerateWithRecall",
  ]);
  const declarations: string[] = [];
  function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && names.has(node.name.getText(tree)))
      declarations.push(`const ${node.getText(tree)};`);
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.equal(declarations.length, names.size);
  const { outputText } = ts.transpileModule(
    declarations.join("\n") +
      "\nreturn { handleGenerateWithRecall, currentLoadAdvanced };",
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  for (const quant of ["int8", "fp8"]) {
    const model = {
      repoId: "unsloth/model-GGUF",
      kind: "gguf",
      filename: "model-Q8_0.gguf",
    };
    const loads: unknown[][] = [];
    const scope = {
      useCallback: (fn: unknown) => fn,
      lastLoad: { current: null },
      status: { loaded: false, repo_id: null },
      loras: [
        { id: " org/style ", weight: 0.7 },
        { id: "", weight: 1 },
        { id: "org/off", weight: 0 },
      ],
      cpuOffload: false,
      speedMode: "auto",
      transformerQuant: quant,
      textEncoderQuant: "int8",
      attentionBackend: "auto",
      memoryMode: "auto",
      transformerCache: "auto",
      selectedGpu: "auto",
      gpuChoices: [],
      busy: null,
      imagePresets: { hydrated: true },
      prompt: "a teapot",
      rememberedModel: model,
      pendingRecalledGeneration: { current: null },
      loadSeq: { current: 3 },
      workflow: "txt2img",
      handleGenerate: () => {
        throw new Error("generation must wait for the load");
      },
      handleLoad: async (...args: unknown[]) => {
        loads.push(args);
        return true;
      },
    };
    const callbacks = new Function(...Object.keys(scope), outputText)(
      ...Object.values(scope),
    );
    await callbacks.handleGenerateWithRecall();
    assert.equal(loads.length, 1);
    assert.equal(loads[0][0], model.repoId);
    assert.deepEqual(loads[0][1], {
      kind: model.kind,
      filename: model.filename,
    });
    const advanced = loads[0][2] as
      | {
          loras?: unknown;
          transformer_quant?: string;
          text_encoder_quant?: string;
        }
      | undefined;
    assert.deepEqual(advanced?.loras, [{ id: "org/style", weight: 0.7 }]);
    assert.equal(advanced?.transformer_quant, quant);
    assert.equal(advanced?.text_encoder_quant, "int8");
    assert.equal(
      callbacks.currentLoadAdvanced("org/different-model").loras,
      undefined,
    );
  }
});

test("recall keeps the exact GGUF artifact", () => {
  const model = {
    repoId: "unsloth/model-GGUF",
    kind: "gguf" as const,
    filename: "model-Q4.gguf",
  };
  rememberImageModel(model);
  assert.deepEqual(readImageModel(), model);
  assert.equal(
    matchesRememberedModel(model, {
      loaded: true,
      repo_id: model.repoId,
      model_kind: "gguf",
      gguf_filename: "model-Q8.gguf",
    }),
    false,
  );
  assert.equal(
    matchesRememberedModel(model, {
      loaded: true,
      repo_id: model.repoId,
      model_kind: "gguf",
      gguf_filename: model.filename,
    }),
    true,
  );
});

test("invalid or incomplete stored targets cannot trigger automatic loading", () => {
  for (const value of [
    "broken",
    "null",
    "{}",
    '{"repoId":"org/model","kind":"gguf"}',
    '{"repoId":"org/model","kind":"unknown"}',
  ]) {
    storage.setItem("unsloth:images:last-model", value);
    assert.equal(readImageModel(), null);
  }
});

test("a new model pick cancels recalled generation and retires the staged load", () => {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile(
    "images-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let declaration: ts.VariableDeclaration | undefined;
  function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "beginPick") {
      assert.equal(declaration, undefined, "beginPick must have one owner");
      declaration = node;
    }
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.ok(declaration);
  const { outputText } = ts.transpileModule(
    `const ${declaration.getText(tree)}; beginPick();`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  const scope = {
    useCallback: (fn: unknown) => fn,
    pendingRecalledGeneration: { current: { model: "previous" } },
    pickSeq: { current: 4 },
    pendingStagedLoad: { current: { token: 4 } },
    pendingLoadEntries: { current: ["previous"] },
    stagedLoadDeferred: { current: true },
    stagedQuantRevert: { current: { prev: "Q8_0" } },
  };
  new Function(...Object.keys(scope), outputText)(...Object.values(scope));
  assert.equal(scope.pendingRecalledGeneration.current, null);
  assert.equal(scope.pickSeq.current, 5);
  assert.equal(scope.pendingStagedLoad.current, null);
  assert.equal(scope.pendingLoadEntries.current, null);
  assert.equal(scope.stagedLoadDeferred.current, false);
  assert.equal(scope.stagedQuantRevert.current, null);
});
