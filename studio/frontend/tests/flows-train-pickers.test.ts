// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fine-tune's base model and the Hugging Face Dataset node use the Train page's own pickers.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { hfDatasetChoice } from "../src/features/flows/editor/dataset-choice.ts";

test("Fine-tune's base model opens the Train page's picker and stores a plain string", () => {
  const fields = readSrc("features/flows/editor/param-fields.tsx");
  assert.match(
    fields,
    /spec\.id === "base_model"[\s\S]{0,200}<FlowTrainModelPicker\s*value=\{String\(value \?\? ""\)\}\s*onChange=\{\(model\) => onChange\(model\)\}/,
  );
  const picker = readSrc("features/flows/editor/model-picker-field.tsx");
  assert.match(picker, /import \{ TrainModelSelector \} from "@\/features\/train-model-picker";/);
  assert.match(picker, /<TrainModelSelector value=\{value\} onValueChange=\{onChange\} \/>/);
  // The generic picker no longer has a train kind.
  assert.doesNotMatch(picker, /kind === "train"|train: \{/);
});

test("TrainModelSelector in controlled mode reads the value and reports picks instead of the store", () => {
  const src = readSrc("features/train-model-picker/components/train-model-selector.tsx");
  assert.match(src, /const controlled = onValueChange !== undefined;/);
  assert.match(src, /const selectedModel = controlled \? value \|\| null : store\.selectedModel;/);
  assert.match(src, /\? \(model\) => model && onValueChange\(model\)\s*: store\.selectTrainingModel;/);
});

test("the Hugging Face Dataset node renders the Train page's dataset picker and subset/split dropdowns", () => {
  const inspector = readSrc("features/flows/editor/inspector.tsx");
  assert.match(
    inspector,
    /spec\.type === HF_DATASET_NODE \? \(\s*<FlowHfDatasetFields params=\{node\.params\} onChange=\{setParams\} \/>/,
  );
  const field = readSrc("features/flows/editor/dataset-picker-field.tsx");
  assert.match(field, /export const HF_DATASET_NODE = "data\.hf_dataset";/);
  assert.match(field, /import \{ DatasetSelector \} from "@\/features\/dataset-picker";/);
  assert.match(field, /import \{ HfDatasetSubsetSplitSelectors \} from "@\/features\/training";/);
  // A new repo clears the subset so the dropdowns pick its default again.
  assert.match(field, /onValueChange=\{\(next\) => onChange\(\{ repo: next, subset: "" \}\)\}/);
  assert.match(field, /hideEvalSplit=\{true\}/);
});

test("DatasetSelector in controlled mode is Hub-only and leaves the Train page's store alone", () => {
  const src = readSrc("features/dataset-picker/components/dataset-selector.tsx");
  assert.match(src, /const dataset = controlled \? value \|\| null : storeDataset;/);
  assert.match(src, /const datasetSource = controlled \? "huggingface" : storeDatasetSource;/);
  assert.match(src, /\.filter\(\(d\) => !d\.partial && !controlled\)/);
  assert.match(src, /onValueChange \? \(id\) => id && onValueChange\(id\) : storeSelectHfDataset/);
});

test("the subset/split selectors can drop the eval split, in both the dropdown and manual forms", () => {
  const src = readSrc("features/training/components/hf-dataset-subset-split-selectors.tsx");
  assert.equal(src.match(/\{!hideEvalSplit && \(/g)?.length, 2);
  assert.match(src, /hideEvalSplit = false,/);
});

test("the node's params map onto the picker's values, with an unset split running as train", () => {
  assert.deepEqual(hfDatasetChoice({}), { repo: null, subset: null, split: "train" });
  assert.deepEqual(hfDatasetChoice({ repo: " openai/gsm8k ", subset: "main", split: "test" }), {
    repo: "openai/gsm8k",
    subset: "main",
    split: "test",
  });
  assert.deepEqual(hfDatasetChoice({ repo: "", subset: "", split: "" }), { repo: null, subset: null, split: null });
});
