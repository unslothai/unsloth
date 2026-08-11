// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Between the last download and the first training step the overlay sat on a static
// line while tokenizing ran for minutes. These pin the parsing of the worker's tqdm
// status messages, whose exact shape is `f"{desc} {pct}% ({n:,}/{total:,})"` in
// `_monitor_tqdm` (studio/backend/core/training/worker.py).

import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyPreparation,
  parsePreparationProgress,
  resolvePreparationMessage,
  shouldShowPreparationStatus,
} from "../src/features/studio/preparation-progress.ts";

test("a preparation step routes to the resource row it belongs to", () => {
  // Dataset work always names itself, so everything else belongs to the model row.
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
  // Fifteen of these reached no row at all, and the three download lines were being
  // replaced by a generic "Preparing" before they got that far.
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
    "Loaded 1000 samples from local files",
    "Encoding audio with SNAC...",
    'Tokenizing ["text"] (num_proc=4) 15% (32,000/207,865)',
  ];
  const audioSteps = [
    // loaded only to preprocess the dataset, so they belong to its row; routing them to the
    // model made the display flip between rows partway through one encoding pass.
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
  // The worker reports `Loading <repo_id>...`, which carries no word the patterns match, so
  // the row stayed empty through the whole model load.
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
  // Case folded, since the message echoes whatever casing the config carries.
  assert.equal(
    classifyPreparation("Loading qwen/qwen3.5-0.8b-base", resources),
    "model",
  );
  // Unset resources fall through to the patterns rather than matching everything.
  assert.equal(classifyPreparation("Loading checkpoint shards", {}), "model");
});

test("the preparation row covers the gap up to the first step", () => {
  assert.equal(shouldShowPreparationStatus("finalizing", 0, false), false);
  assert.equal(shouldShowPreparationStatus("completed", 0, false), false);
  assert.equal(shouldShowPreparationStatus("configuring", 0, false), true);
  assert.equal(shouldShowPreparationStatus("loading_dataset", 0, false), true);
  assert.equal(shouldShowPreparationStatus("idle", 0, true), true);
  // The worker reports `training` as soon as the trainer is built, with dataset
  // mapping still ahead of it, so the row stays until a step lands.
  assert.equal(shouldShowPreparationStatus("training", 0, false), true);
  assert.equal(shouldShowPreparationStatus("training", 1, false), false);
});

test("the fallback covers only the window before the worker reports", () => {
  assert.equal(resolvePreparationMessage("   ", "Preparing"), "Preparing");
  // "Downloading dataset: ..." is a real step of the dataset's setup, not a stale line to
  // discard: dropping every message starting with "download" hid three of them.
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
  // 16,000/207,865 is 7.7%, and the worker truncates. Taking its number rather than
  // recomputing keeps the bar and the log line above it showing the same figure.
  assert.equal(
    parsePreparationProgress("Filter (num_proc=4) 7% (16,000/207,865)", "Preparing")
      .percent,
    7,
  );
});

test("the audio loops report bare counts and still draw a bar", () => {
  // `Encoding audio... {i}/{n}` and friends carry no percent, so the tqdm shape misses them
  // and a long preprocessing pass swept indeterminately with the counts already in hand.
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
  // A zero total, and a bar whose `n` overran `total` after a restart.
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
  // `Starting SNAC training...` and `Starting Whisper training...` name a codec only
  // because it names the run. Matching `snac`/`whisper` sent them to the dataset row while
  // the sibling `Starting CSM training...` went to the model row, so the same event landed
  // in different places depending on which family was selected.
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
  // Says "eval split" rather than "dataset", so no pattern caught it and a dataset reload
  // was reported on the model row.
  const { title } = parsePreparationProgress(
    "Cached eval split unavailable; reloading train and eval from the Hub...",
    "Preparing",
  );
  assert.equal(classifyPreparation(title), "dataset");
});
