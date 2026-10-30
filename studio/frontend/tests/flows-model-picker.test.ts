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
  // `loaded` only marks residency; a pick just stores the choice.
  assert.match(picker, /loaded=\{resident\}/);
  assert.match(picker, /onValueChange=\{\(id, meta\) => onChange\(choiceFromPick\(id, meta\)\)\}/, "a pick never loads a model");
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

test("Fine-tune's Recommended leads with the Train page's own picks", () => {
  const picker = readSrc("features/flows/editor/model-picker-field.tsx");
  assert.match(picker, /priorityTrainingModels\(deviceType\)/);
  assert.match(picker, /if \(kind === "train"\)\s*setModels\(trainModels\(deviceType, all\)\)/);
});

test("Fine-tune's Recommended shows the Train page's picks as their own group, not as GGUF rows", () => {
  const picker = readSrc("features/flows/editor/model-picker-field.tsx");
  assert.match(picker, /moreModels=\{\s*kind === "train" \? \{ label: "For fine-tuning", models \} : undefined\s*\}/);
  // Unmarked rows count as GGUF in the "More" group, so the priority ids must say they aren't.
  assert.match(picker, /priorityTrainingModels\(deviceType\)\.map\(\(id\) => \(\{\s*id,\s*name: id,\s*isGguf: false,/);
});

test("a loaded model the picker's rowFilter rejects is not pinned at the top of Recommended", () => {
  const pickers = readSrc("features/model-picker/components/model-selector/pickers.tsx");
  const loaded = pickers.slice(pickers.indexOf("const loadedRows = useMemo"), pickers.indexOf("const loadedRowIds"));
  assert.match(loaded, /!rowFilter \|\| rowFilter\(\{ id: m\.id \}\)/);
  assert.match(loaded, /\[isChatPicker, section, debouncedQuery, loadedModels, rowFilter\]/);
});

test("a Flows model field's closed trigger reads like Chat's: ghost, name with format and quant, check when resident", () => {
  const picker = readSrc("features/flows/editor/model-picker-field.tsx");
  assert.match(picker, /variant="ghost"/);
  assert.doesNotMatch(picker, /variant="outline"/);
  assert.match(picker, /loaded=\{resident\}\s*loadedMark="check"/);
  assert.match(picker, /activeGgufVariant=\{choice\.variant \?\? null\}/);
  const selector = readSrc("features/model-picker/components/model-selector.tsx");
  assert.match(selector, /loadedMark === "check" \? \(\s*<HugeiconsIcon\s*icon=\{CheckmarkCircle02Icon\}/);
  // Every other picker keeps the dot unless it asks.
  assert.match(selector, /loadedMark = "dot",/);
});
