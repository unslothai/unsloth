// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { isDiffusionTask } from "../src/features/model-picker/components/model-selector/model-capabilities.ts";
import { readSrc } from "./helpers/kit.ts";

const pickerSource = readSrc("features/model-picker/components/model-selector/pickers.tsx");

function taskList(name: string): string[] {
  const match = pickerSource.match(new RegExp(`export const ${name} = \\[([\\s\\S]*?)\\] as const;`));
  assert.ok(match, name);
  return [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
}

test("Chat excludes both downloaded FLUX formats while keeping text and VLMs", () => {
  const cached = [
    { id: "unsloth/flux-3-image-klein-pup-GGUF", task: "text-to-image" },
    { id: "unsloth/flux-3-image-klein-FP8", task: "text-to-image" },
    { id: "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF", task: "text-generation" },
    { id: "unsloth/Qwen3.5-2B", task: "image-text-to-text" },
  ];
  assert.deepEqual(cached.filter((row) => !isDiffusionTask(row.task)).map((row) => row.id), [
    "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF",
    "unsloth/Qwen3.5-2B",
  ]);
});

test("image, editing and video generation tasks stay out of Chat", () => {
  for (const task of [
    "text-to-image", "image-to-image", "image-text-to-image", "inpainting",
    "text-to-video", "image-to-video", "image-text-to-video", "image-diffusion-unsupported",
  ]) {
    assert.equal(isDiffusionTask(task), true, task);
  }
});

test("unknown metadata and existing audio routing retain their behavior", () => {
  for (const task of [undefined, null, "text-generation", "image-text-to-text", "text-to-speech"]) {
    assert.equal(isDiffusionTask(task), false, String(task));
  }
});

test("every task the Images and Video pickers own is kept out of Chat", () => {
  const owned = [...taskList("IMAGE_GEN_TASKS"), ...taskList("VIDEO_GEN_TASKS")];
  assert.ok(owned.length >= 6);
  for (const task of owned) assert.equal(isDiffusionTask(task), true, task);
});

test("the unfiltered task gate drops diffusion rows and the filtered branch is untouched", () => {
  const start = pickerSource.indexOf("function passesTaskGate(");
  assert.ok(start >= 0);
  const body = pickerSource.slice(start, pickerSource.indexOf("\n}\n", start));
  const filtered = body.indexOf("if (filter) {");
  assert.ok(filtered >= 0);
  assert.match(body, /taskMatchesFilter\(repoTask, filter\)/);
  assert.match(body.slice(body.lastIndexOf("}")), /return !isDiffusionTask\(repoTask\);/);
});

test("all five On Device lists go through the task gate", () => {
  for (const list of [
    "sortedCachedGguf",
    "sortedCachedModels",
    "sortedLmStudio",
    "sortedLocalDir",
    "sortedCustomFolderModels",
  ]) {
    const start = pickerSource.indexOf(`const ${list} = useMemo(`);
    assert.ok(start >= 0, list);
    const end = pickerSource.indexOf("const ", start + 20);
    assert.match(pickerSource.slice(start, end), /passesTaskGate\(/, list);
  }
});
