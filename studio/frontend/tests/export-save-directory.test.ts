// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRelativeSaveDirectory,
  saveDirectoryField,
} from "../src/features/export/lib/save-directory.ts";

const DEFAULT = "qwen3-4b-GGUF";

test("the save folder input shows the default until the user edits it", () => {
  assert.deepEqual(saveDirectoryField(null, DEFAULT), {
    inputValue: DEFAULT,
    saveDirectory: DEFAULT,
  });
});

test("a cleared save folder input stays empty and exports to the default", () => {
  assert.deepEqual(saveDirectoryField("", DEFAULT), {
    inputValue: "",
    saveDirectory: DEFAULT,
  });
});

test("a trailing space survives typing and is trimmed from the export path", () => {
  assert.deepEqual(saveDirectoryField("D:\\My ", DEFAULT), {
    inputValue: "D:\\My ",
    saveDirectory: "D:\\My",
  });
});

test("a pasted absolute path is used as typed", () => {
  assert.deepEqual(saveDirectoryField("/Volumes/ext/exports", DEFAULT), {
    inputValue: "/Volumes/ext/exports",
    saveDirectory: "/Volumes/ext/exports",
  });
});

const checkpointDir = (
  method: "lora" | "merged" | "gguf",
  run: string,
  checkpoint: string,
) => buildRelativeSaveDirectory(method, "checkpoint", "", run, checkpoint);

test("a run's LoRA and merged exports get separate default folders", () => {
  const lora = checkpointDir("lora", "run_A", "run_A");
  const merged = checkpointDir("merged", "run_A", "run_A");
  assert.notEqual(lora, merged);
  assert.ok(lora.startsWith("run_A/"));
  assert.ok(merged.startsWith("run_A/"));
});

test("the same checkpoint of two runs gets separate GGUF default folders", () => {
  assert.notEqual(
    checkpointDir("gguf", "run_A", "checkpoint-100"),
    checkpointDir("gguf", "run_B", "checkpoint-100"),
  );
});

test("a run's final checkpoint keeps its GGUF default folder", () => {
  assert.equal(checkpointDir("gguf", "run_A", "run_A"), "run_A-GGUF");
});
