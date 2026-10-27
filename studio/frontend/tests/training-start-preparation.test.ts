// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pins parsing of `_monitor_tqdm`'s `f"{desc} {pct}% ({n:,}/{total:,})"` (backend worker.py).

import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyPreparation,
  parsePreparationProgress,
  resolvePreparationMessage,
  shouldShowPreparationStatus,
} from "../src/features/studio/preparation-progress.ts";

test("a preparation step routes to the resource row it belongs to", () => {
  assert.equal(classifyPreparation('Tokenizing ["text"]'), "dataset");
  assert.equal(classifyPreparation("Loading dataset"), "dataset");
  assert.equal(classifyPreparation("Map"), "dataset");
  assert.equal(classifyPreparation("Unsloth: Formatting dataset"), "dataset");
  assert.equal(classifyPreparation("Loading checkpoint shards"), "model");
  assert.equal(classifyPreparation("Loading model"), "model");
  // "tokenizer" is model setup; only "tokenizing" is dataset work.
  assert.equal(classifyPreparation("Loading tokenizer"), "model");
  assert.equal(classifyPreparation("Configuring training"), "model");
});

test("every status the worker sends reaches a row", () => {
  // Swept from the `_send_status`/`status_message` literals in studio/backend/core/training.
  const resources = {
    modelName: "Qwen/Qwen3.5-0.8B-Base",
    datasetName: "ryanmarten/OpenThoughts-1k-sample",
  };
  const datasetSteps = [
    "Loading dataset...",
    "Loading and formatting dataset...",
    "Loading cached dataset: ryanmarten/OpenThoughts-1k-sample...",
    "Downloading dataset: ryanmarten/OpenThoughts-1k-sample...",
    "Downloading dataset from S3...",
    "Downloaded ryanmarten/OpenThoughts-1k-sample (1,000 rows)",
    "Streaming dataset: ryanmarten/OpenThoughts-1k-sample...",
    "Formatting dataset (chatml)...",
    "Formatting VLM dataset...",
    "Dataset ready (1,000 samples, chatml format)",
    "Sliced dataset to 500 rows (indices 0-500)",
    "Using 1024 of 192523 rows (max_steps run)",
    "Loaded 1000 samples from local files",
    "Encoding audio with SNAC...",
    'Tokenizing ["text"] (num_proc=4) 15% (32,000/207,865)',
  ];
  const audioSteps = [
    // Loaded only to preprocess the dataset, so they belong to its row.
    "Loading SNAC codec model...",
    "Loading BiCodec tokenizer...",
    "Loading OuteTTS AudioProcessor...",
    "Loading Whisper model for word timings...",
    "Encoding audio with BiCodec... 100/1000",
    "Preprocessing CSM... 5/100",
  ];
  const modelSteps = [
    "Importing Unsloth...",
    "Detecting model type...",
    "Loading Qwen/Qwen3.5-0.8B-Base...",
    "Loading model...",
    "Configuring training...",
    "Configuring LoRA adapters...",
    "Preparing model for full finetuning...",
    "Full finetuning mode - no LoRA adapters",
    "Initializing MLX training...",
    "Loading MLX libraries...",
    "Starting training...",
    "Saving model...",
  ];
  for (const message of [...datasetSteps, ...audioSteps]) {
    const { title } = parsePreparationProgress(message, "Preparing");
    assert.equal(classifyPreparation(title, resources), "dataset", message);
  }
  for (const message of modelSteps) {
    const { title } = parsePreparationProgress(message, "Preparing");
    assert.equal(classifyPreparation(title, resources), "model", message);
  }
});

test("a step naming only a repo id routes by that name", () => {
  // `Loading <repo_id>...` carries no word the patterns match.
  const resources = {
    modelName: "Qwen/Qwen3.5-0.8B-Base",
    datasetName: "ryanmarten/OpenThoughts-1k-sample",
  };
  assert.equal(
    classifyPreparation("Loading Qwen/Qwen3.5-0.8B-Base", resources),
    "model",
  );
  assert.equal(
    classifyPreparation("Loading ryanmarten/OpenThoughts-1k-sample", resources),
    "dataset",
  );
  assert.equal(
    classifyPreparation("Loading qwen/qwen3.5-0.8b-base", resources),
    "model",
  );
  assert.equal(classifyPreparation("Loading checkpoint shards", {}), "model");
});

test("every tqdm description the dataset work emits reaches the dataset row", () => {
  // Swept from the `desc =` literals under studio/backend, forwarded verbatim by `_monitor_tqdm`.
  const resources = {
    modelName: "Qwen/Qwen3-0.6B",
    datasetName: "ryanmarten/OpenThoughts-1k-sample",
  };
  const descriptions = [
    "Applying chat template to chatml 15% (32,000/207,865)",
    "Applying chat template to sharegpt 15% (32,000/207,865)",
    "Converting VLM samples 10% (100/1,000)",
    "Converting ShareGPT+image 5% (50/1,000)",
  ];
  for (const message of descriptions) {
    const { title } = parsePreparationProgress(message, "Preparing");
    assert.equal(classifyPreparation(title, resources), "dataset", message);
  }
  assert.equal(classifyPreparation("Loading checkpoint shards", resources), "model");
  assert.equal(classifyPreparation("Loading tokenizer", resources), "model");
});

test("an id shared by both repos routes by wording, not by the tie-break", () => {
  // The Hub allows one owner/name as both repo types, so the id decides nothing.
  const resources = { modelName: "org/foo", datasetName: "org/foo" };
  assert.equal(classifyPreparation("Loading org/foo", resources), "model");
  assert.equal(classifyPreparation("Tokenizing org/foo", resources), "dataset");
  assert.equal(classifyPreparation("Loading checkpoint shards", resources), "model");
  const distinct = { modelName: "org/foo-base", datasetName: "org/foo" };
  assert.equal(classifyPreparation("Loading org/foo-base", distinct), "model");
  assert.equal(classifyPreparation("Loading org/foo", distinct), "dataset");
});

test("the preparation row covers the gap up to the first step", () => {
  assert.equal(shouldShowPreparationStatus("finalizing", 0, false), false);
  assert.equal(shouldShowPreparationStatus("completed", 0, false), false);
  assert.equal(shouldShowPreparationStatus("configuring", 0, false), true);
  assert.equal(shouldShowPreparationStatus("loading_dataset", 0, false), true);
  assert.equal(shouldShowPreparationStatus("idle", 0, true), true);
  // `training` is reported once the trainer is built, with dataset mapping still ahead.
  assert.equal(shouldShowPreparationStatus("training", 0, false), true);
  assert.equal(shouldShowPreparationStatus("training", 1, false), false);
});

test("the fallback covers only the window before the worker reports", () => {
  assert.equal(resolvePreparationMessage("   ", "Preparing"), "Preparing");
  // "Downloading dataset: ..." is a real setup step, not a stale line.
  assert.equal(
    resolvePreparationMessage("Downloading dataset from S3...", "Preparing"),
    "Downloading dataset from S3...",
  );
  assert.equal(
    resolvePreparationMessage('Tokenizing ["text"] 15% (1/2)', "Preparing"),
    'Tokenizing ["text"] 15% (1/2)',
  );
});

test("a counted message draws a determinate bar from the worker's own percent", () => {
  assert.deepEqual(
    parsePreparationProgress(
      'Tokenizing ["text"] (num_proc=4) 15% (32,000/207,865)',
      "Preparing",
    ),
    {
      title: 'Tokenizing ["text"]',
      detail: "32,000 / 207,865",
      percent: 15,
    },
  );
  // The worker truncates; reuse its number so the bar matches the log line.
  assert.equal(
    parsePreparationProgress("Filter (num_proc=4) 7% (16,000/207,865)", "Preparing")
      .percent,
    7,
  );
});

test("the audio loops report bare counts and still draw a bar", () => {
  // These carry no percent, so the tqdm shape misses them.
  assert.deepEqual(
    parsePreparationProgress("Encoding audio... 100/1000", "Preparing"),
    { title: "Encoding audio", detail: "100 / 1000", percent: 10 },
  );
  assert.deepEqual(
    parsePreparationProgress("Processing train audio... 1,500/12,000", "Preparing"),
    { title: "Processing train audio", detail: "1,500 / 12,000", percent: 12 },
  );
});

test("an uncounted message stays indeterminate", () => {
  assert.deepEqual(parsePreparationProgress("Loading model...", "Preparing"), {
    title: "Loading model",
    detail: null,
    percent: null,
  });
  assert.deepEqual(
    parsePreparationProgress("Unsloth: Formatting dataset…", "Preparing"),
    { title: "Formatting dataset", detail: null, percent: null },
  );
  assert.deepEqual(parsePreparationProgress("", "Preparing"), {
    title: "Preparing",
    detail: null,
    percent: null,
  });
});

test("counts that cannot describe a bar do not draw one", () => {
  assert.deepEqual(parsePreparationProgress("Filter 100% (10/0)", "Preparing"), {
    title: "Filter",
    detail: null,
    percent: null,
  });
  assert.deepEqual(parsePreparationProgress("Filter 100% (11/10)", "Preparing"), {
    title: "Filter",
    detail: null,
    percent: null,
  });
});

test("a trainer's own start line stays on the model row whatever it trains", () => {
  // These name a codec only because it names the run, so they must not go to the dataset row.
  for (const message of [
    "Starting SNAC training...",
    "Starting Whisper training...",
    "Starting CSM training...",
    "Starting embedding training...",
    "Starting training...",
    "Initializing MLX training...",
    "Queued MLX training setup",
  ]) {
    const { title } = parsePreparationProgress(message, "Preparing");
    assert.equal(classifyPreparation(title), "model", message);
  }
});

test("reloading the eval split is dataset work", () => {
  const { title } = parsePreparationProgress(
    "Cached eval split unavailable; reloading train and eval from the Hub...",
    "Preparing",
  );
  assert.equal(classifyPreparation(title), "dataset");
});

test("one resource id being a prefix of the other does not steal the row", () => {
  // `Loading org/foo-base` contains `org/foo`, so a bare `includes` misroutes it.
  const resources = { modelName: "org/foo-base", datasetName: "org/foo" };
  assert.equal(classifyPreparation("Loading org/foo-base", resources), "model");
  assert.equal(classifyPreparation("Loading org/foo", resources), "dataset");
  const swapped = { modelName: "org/foo", datasetName: "org/foo-sample" };
  assert.equal(classifyPreparation("Loading org/foo-sample", swapped), "dataset");
  assert.equal(classifyPreparation("Loading org/foo", swapped), "model");
  assert.equal(classifyPreparation("Loading org/foo-base...", resources), "model");
  assert.equal(
    classifyPreparation("Downloading dataset: org/foo (1,000 rows)", resources),
    "dataset",
  );
});
