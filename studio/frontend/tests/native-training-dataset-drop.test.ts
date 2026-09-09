// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  TRAINING_DATASET_UPLOAD_ACCEPT,
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
  classifyNativeTrainingDatasetDrop,
  isTrainingDatasetUploadPath,
  nativeDropPositionHitsBounds,
  nativePathFilename,
} from "../src/features/training/lib/native-dataset-drop.ts";

const BACKEND_DATASET_EXTENSIONS_PATTERN =
  /LOCAL_UPLOAD_EXTS\s*=\s*\{([^}]+)\}/s;
const BACKEND_DOCUMENT_EXTENSIONS_PATTERN =
  /UNSTRUCTURED_ALLOWED_EXTS\s*=\s*\{([^}]+)\}/s;
const RECIPE_DOCUMENT_EXTENSIONS_PATTERN =
  /ACCEPTED_EXTENSIONS\s*=\s*\[([^\]]+)\]/s;
const RUST_DATASET_EXTENSIONS_PATTERN =
  /TRAINING_DATASET_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const DOTTED_EXTENSION_PATTERN = /"(\.[^"]+)"/g;
const UNDOTTED_EXTENSION_PATTERN = /"([^"]+)"/g;

function extractLiteralExtensions(source: string, pattern: RegExp): string[] {
  const literal = pattern.exec(source)?.[1];
  if (literal === undefined) {
    throw new Error(`Extension declaration did not match ${pattern.source}`);
  }
  return [...literal.matchAll(DOTTED_EXTENSION_PATTERN)]
    .map((match) => match[1])
    .sort();
}

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

test("redirects Markdown documents across native path formats", () => {
  for (const path of [
    String.raw`C:\Users\trainer\Documents\NOTES.MD`,
    "/home/trainer/documents/notes.md",
    "/Users/trainer/Documents/NOTES.MD",
  ]) {
    const dropped = classifyNativeTrainingDatasetDrop([path]);
    assert.equal(dropped.kind, "document", path);
    assert.equal(
      dropped.kind === "document" ? dropped.filename.toLowerCase() : null,
      "notes.md",
      path,
    );
  }
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

// Only WebView2 reports a device-pixel drop position; macOS and GTK already
// report CSS pixels at 100% zoom, so scaling those by the monitor factor halved
// every hit test on a HiDPI display and the zone never matched.
test("hit testing scales a Windows drop position to CSS pixels", () => {
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64)" },
    configurable: true,
  });
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

test("hit testing takes a macOS drop position as-is", () => {
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X)" },
    configurable: true,
  });
  const bounds = { left: 100, right: 300, top: 50, bottom: 150 };
  assert.equal(
    nativeDropPositionHitsBounds({ x: 200, y: 100 }, 2, bounds),
    true,
  );
  assert.equal(
    nativeDropPositionHitsBounds({ x: 400, y: 200 }, 2, bounds),
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
      .match(BACKEND_DATASET_EXTENSIONS_PATTERN)?.[1]
      .matchAll(DOTTED_EXTENSION_PATTERN) ?? []),
  ]
    .map((match) => match[1])
    .sort();
  const rust = [
    ...(rustSource
      .match(RUST_DATASET_EXTENSIONS_PATTERN)?.[1]
      .matchAll(UNDOTTED_EXTENSION_PATTERN) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();

  assert.deepEqual([...TRAINING_DATASET_UPLOAD_EXTENSIONS].sort(), backend);
  assert.deepEqual([...TRAINING_DATASET_UPLOAD_EXTENSIONS].sort(), rust);
  assert.equal(
    TRAINING_DATASET_UPLOAD_ACCEPT,
    TRAINING_DATASET_UPLOAD_EXTENSIONS.join(","),
  );
});

test("training document redirects match Data Recipes", () => {
  const backendSource = readFileSync(
    new URL("../../backend/routes/data_recipe/seed.py", import.meta.url),
    "utf8",
  );
  const recipeSource = readFileSync(
    new URL(
      "../src/features/recipe-studio/dialogs/seed/unstructured-drop-zone.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const backend = extractLiteralExtensions(
    backendSource,
    BACKEND_DOCUMENT_EXTENSIONS_PATTERN,
  );
  const recipe = extractLiteralExtensions(
    recipeSource,
    RECIPE_DOCUMENT_EXTENSIONS_PATTERN,
  );
  const training = [...TRAINING_DOCUMENT_REDIRECT_EXTENSIONS].sort();

  assert.deepEqual(training, [".docx", ".md", ".pdf", ".txt"]);
  assert.deepEqual(training, backend);
  assert.deepEqual(training, recipe);
});
