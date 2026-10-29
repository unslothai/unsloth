// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Model params use Studio's model picker and keep the saved {repo, variant} shape.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { choiceFromPick, isTrainableRepo, modelChoiceOf } from "../src/features/flows/editor/model-choice.ts";

test("values saved before the picker still load", () => {
  assert.equal(modelChoiceOf(null), null);
  assert.equal(modelChoiceOf(undefined), null);
  assert.deepEqual(modelChoiceOf({ repo: "unsloth/Qwen3-4B-GGUF", variant: "Q4_K_M" }), {
    repo: "unsloth/Qwen3-4B-GGUF",
    variant: "Q4_K_M",
  });
  assert.deepEqual(modelChoiceOf({ repo: "unsloth/Qwen3-4B" }), { repo: "unsloth/Qwen3-4B", variant: null });
  assert.deepEqual(modelChoiceOf({ repo: "", variant: null }), { repo: "", variant: null });
  assert.deepEqual(modelChoiceOf("unsloth/Qwen3-4B"), { repo: "unsloth/Qwen3-4B", variant: null });
});

test("a pick stores the repo plus the GGUF variant when the picker returns one", () => {
  assert.deepEqual(choiceFromPick("unsloth/Qwen3-4B-GGUF", { ggufVariant: "Q4_K_M" }), {
    repo: "unsloth/Qwen3-4B-GGUF",
    variant: "Q4_K_M",
  });
  assert.deepEqual(choiceFromPick("unsloth/Qwen3-4B", {}), { repo: "unsloth/Qwen3-4B", variant: null });
});

test("the model field renders Studio's ModelSelector, keeps the loaded-model checkbox, and has no datalist", () => {
  const fields = readSrc("features/flows/editor/param-fields.tsx");
  const field = fields.slice(fields.indexOf("function ModelField"), fields.indexOf("export const DOCUMENT_ACCEPT"));
  assert.match(field, /Use the loaded \{spec\.model_kind \?\? ""\} model/);
  assert.match(field, /<FlowModelPicker/);
  assert.doesNotMatch(field, /datalist|Hugging Face repo/);
  const picker = readSrc("features/flows/editor/model-picker-field.tsx");
  assert.match(picker, /import \{ type ModelOption, ModelSelector \} from "@\/features\/model-picker";/);
  assert.match(picker, /loaded=\{false\}/, "a pick never loads a model");
});

test("Fine-tune's base model keeps its plain string and uses the same picker", () => {
  const fields = readSrc("features/flows/editor/param-fields.tsx");
  assert.match(fields, /spec\.id === "base_model"[\s\S]{0,200}<FlowModelPicker[\s\S]{0,200}onChange=\{\(choice\) => onChange\(choice\.repo\)\}/);
});

test("Fine-tune's base model picker leaves GGUF repos out", () => {
  assert.equal(isTrainableRepo("unsloth/Qwen3-4B"), true);
  assert.equal(isTrainableRepo("unsloth/Qwen3-4B-GGUF"), false);
  const fields = readSrc("features/flows/editor/param-fields.tsx");
  assert.match(fields, /spec\.id === "base_model"[\s\S]{0,120}kind="train"/);
  const picker = readSrc("features/flows/editor/model-picker-field.tsx");
  assert.match(picker, /train: \{[^}]*rowFilter: \(row\) => isTrainableRepo\(row\.id\)/);
});

test("a select param the doc never stored shows its default, not an empty box", () => {
  const fields = readSrc("features/flows/editor/param-fields.tsx");
  const select = fields.slice(fields.indexOf('case "select":'), fields.indexOf('case "seed":'));
  assert.match(select, /spec\.default \?\? spec\.options\?\.\[0\]/);
});
