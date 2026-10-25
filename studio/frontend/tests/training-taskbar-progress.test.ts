// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { taskbarProgressFor } = await import(
  "../src/features/training/hooks/use-training-taskbar-progress.ts"
);

test("a training run fills the taskbar bar with its rounded percent", () => {
  assert.deepEqual(
    taskbarProgressFor({
      phase: "training",
      isTrainingRunning: true,
      progressPercent: 41.6,
    }),
    { status: "Normal", progress: 42 },
  );
});

test("work with no percent yet, or none left, shows a busy bar", () => {
  for (const phase of [
    "downloading_model",
    "loading_model",
    "configuring",
  ] as const) {
    assert.deepEqual(
      taskbarProgressFor({
        phase,
        isTrainingRunning: false,
        progressPercent: 0,
      }),
      { status: "Indeterminate" },
    );
  }
  assert.deepEqual(
    taskbarProgressFor({
      phase: "training",
      isTrainingRunning: true,
      progressPercent: 0,
    }),
    { status: "Indeterminate" },
  );
  assert.deepEqual(
    taskbarProgressFor({
      phase: "finalizing",
      isTrainingRunning: true,
      progressPercent: 100,
    }),
    { status: "Indeterminate" },
  );
});

test("a failed run turns the bar red, even before its first step", () => {
  assert.deepEqual(
    taskbarProgressFor({
      phase: "error",
      isTrainingRunning: false,
      progressPercent: 0,
    }),
    { status: "Error", progress: 100 },
  );
});

test("idle, completed and stopped runs clear the bar", () => {
  for (const phase of ["idle", "completed", "stopped"] as const) {
    assert.deepEqual(
      taskbarProgressFor({
        phase,
        isTrainingRunning: false,
        progressPercent: 100,
      }),
      { status: "None" },
    );
  }
});
