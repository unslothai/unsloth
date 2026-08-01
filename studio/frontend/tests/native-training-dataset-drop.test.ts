// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
  classifyNativeTrainingDatasetDrop,
  isTrainingDatasetUploadPath,
  nativeDropPositionHitsBounds,
  nativePathFilename,
} from "../src/features/training/lib/native-dataset-drop.ts";

test("classifies supported desktop training drops", () => {
  assert.deepEqual(
    classifyNativeTrainingDatasetDrop([String.raw`C:\data\train.JSONL`]),
    {
      kind: "dataset",
      path: String.raw`C:\data\train.JSONL`,
      filename: "train.JSONL",
    },
  );
  assert.equal(
    classifyNativeTrainingDatasetDrop(["/data/source.pdf"]).kind,
    "document",
  );
  assert.equal(
    classifyNativeTrainingDatasetDrop(["/data/train.zip"]).kind,
    "unsupported",
  );
  assert.equal(
    classifyNativeTrainingDatasetDrop(["/data/a.csv", "/data/b.csv"]).kind,
    "multiple",
  );
});

test("distinguishes uploaded files from recipe output directories", () => {
  for (const path of [
    "/datasets/uploads/train.JSONL",
    String.raw`C:\datasets\uploads\train.parquet`,
  ]) {
    assert.equal(isTrainingDatasetUploadPath(path), true, path);
  }
  for (const path of [
    "/datasets/recipes/recipe_support/parquet-files",
    String.raw`C:\datasets\recipes\recipe_support\parquet-files`,
  ]) {
    assert.equal(isTrainingDatasetUploadPath(path), false, path);
  }
});

test("truncates native dataset filenames without splitting Unicode characters", () => {
  const filename = nativePathFilename(`/data/${"a".repeat(159)}💡.jsonl`);
  assert.equal(Array.from(filename).length, 160);
  assert.equal(filename.endsWith("💡"), true);
});

test("classifies native drops before truncating long display filenames", () => {
  const datasetPath = `C:\\data\\${"a".repeat(170)}.JSONL`;
  const documentPath = `/data/${"b".repeat(170)}.pdf`;

  assert.deepEqual(classifyNativeTrainingDatasetDrop([datasetPath]), {
    kind: "dataset",
    path: datasetPath,
    filename: "a".repeat(160),
  });
  assert.deepEqual(classifyNativeTrainingDatasetDrop([documentPath]), {
    kind: "document",
    filename: "b".repeat(160),
  });
});

test("hit testing converts native physical coordinates to CSS pixels", () => {
  const bounds = { left: 100, right: 300, top: 50, bottom: 150 };
  assert.equal(
    nativeDropPositionHitsBounds({ x: 400, y: 200 }, 2, bounds),
    true,
  );
  assert.equal(
    nativeDropPositionHitsBounds({ x: 700, y: 200 }, 2, bounds),
    false,
  );
});

test("native dataset drops track runtime window scale changes", () => {
  const source = readFileSync(
    new URL(
      "../src/features/studio/sections/use-dataset-uploads.ts",
      import.meta.url,
    ),
    "utf8",
  );

  assert.equal(source.includes("currentWindow.onScaleChanged("), true);
  assert.equal(source.includes("scaleFactor = payload.scaleFactor"), true);
  assert.equal(source.includes("stopScaleChanged?.()"), true);
});

test("frontend, backend, and Rust accept the same native dataset extensions", () => {
  const backendSource = readFileSync(
    new URL("../../backend/hub/services/datasets/local.py", import.meta.url),
    "utf8",
  );
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const backend = [
    ...(backendSource
      .match(/LOCAL_UPLOAD_EXTS\s*=\s*\{([^}]+)\}/s)?.[1]
      .matchAll(/"(\.[^"]+)"/g) ?? []),
  ]
    .map((match) => match[1])
    .sort();
  const rust = [
    ...(rustSource
      .match(/TRAINING_DATASET_EXTS[^=]*=\s*&\[([^\]]+)\]/s)?.[1]
      .matchAll(/"([^"]+)"/g) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();

  assert.deepEqual([...TRAINING_DATASET_UPLOAD_EXTENSIONS].sort(), backend);
  assert.deepEqual([...TRAINING_DATASET_UPLOAD_EXTENSIONS].sort(), rust);
});
