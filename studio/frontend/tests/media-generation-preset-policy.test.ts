// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  chainDynamicDefaultRollback,
  closestDurationIndex,
  closestResolutionIndex,
  getBuiltinVariantName,
  mergeUntouchedParams,
  normalizeCustomPresets,
  reapplyTargetFromStatus,
} from "../src/features/generation-presets/preset-policy.ts";

test("pending model defaults preserve only fields the user edited", () => {
  const baseline = { negativePrompt: "", width: 768, steps: 8 };
  assert.deepEqual(
    mergeUntouchedParams(
      baseline,
      { ...baseline, negativePrompt: "keep this", width: 1024 },
      { negativePrompt: "", width: 1280, steps: 40 },
    ),
    { negativePrompt: "keep this", width: 1024, steps: 40 },
  );
});

test("dynamic default rollbacks unwind newest-first to the original baseline", () => {
  const calls: string[] = [];
  const previous = () => {
    calls.push("previous");
    return true;
  };
  const next = () => {
    calls.push("next");
    return true;
  };
  const rollback = chainDynamicDefaultRollback<number>(previous, next);
  assert.equal(rollback((current: number) => current), true);
  assert.deepEqual(calls, ["next", "previous"]);
});

test("a stale newer rollback cannot continue into an older transaction", () => {
  let previousCalled = false;
  const rollback = chainDynamicDefaultRollback<number>(
    () => {
      previousCalled = true;
      return true;
    },
    () => false,
  );
  assert.equal(rollback((current: number) => current), false);
  assert.equal(previousCalled, false);
});

test("resident media status reconstructs every reloadable model kind", () => {
  assert.deepEqual(
    reapplyTargetFromStatus({
      loaded: true,
      repo_id: "unsloth/image-pipeline",
      model_kind: "pipeline",
    }),
    { repoId: "unsloth/image-pipeline", kind: "pipeline" },
  );
  assert.deepEqual(
    reapplyTargetFromStatus({
      loaded: true,
      repo_id: "unsloth/image-gguf",
      model_kind: "gguf",
      gguf_filename: "image-Q4_K_M.gguf",
    }),
    {
      repoId: "unsloth/image-gguf",
      kind: "gguf",
      filename: "image-Q4_K_M.gguf",
    },
  );
  assert.deepEqual(
    reapplyTargetFromStatus({
      loaded: true,
      repo_id: "/models/video",
      model_kind: "single_file",
      gguf_filename: "video.safetensors",
    }),
    {
      repoId: "/models/video",
      kind: "single_file",
      filename: "video.safetensors",
    },
  );
});

test("resident checkpoint status without an exact filename is not reloadable", () => {
  assert.equal(
    reapplyTargetFromStatus({
      loaded: true,
      repo_id: "unsloth/image-gguf",
      model_kind: "gguf",
    }),
    null,
  );
  assert.equal(
    reapplyTargetFromStatus({
      loaded: false,
      repo_id: "unsloth/image-pipeline",
      model_kind: "pipeline",
    }),
    null,
  );
});

test("saving over Default creates a protected custom variant", () => {
  const used = new Set(["Default", "Default 1", "Portrait"]);
  assert.equal(getBuiltinVariantName(used), "Default 2");
});

test("hydration protects built-in names and drops empty names", () => {
  const normalized = normalizeCustomPresets([
    { name: " Default ", params: { steps: 10 } },
    { name: "", params: { steps: 20 } },
    { name: "Portrait", params: { steps: 30 } },
  ]);
  assert.deepEqual(
    normalized.map((preset) => preset.name),
    ["Default 1", "Portrait"],
  );
});

test("video resolution mapping prioritizes aspect before area", () => {
  const options: Array<[number, number]> = [
    [1024, 1024],
    [1216, 704],
    [704, 1216],
  ];
  assert.equal(closestResolutionIndex(options, 1920, 1080), 1);
  assert.equal(closestResolutionIndex(options, 1080, 1920), 2);
});

test("video duration mapping uses the closest supported temporal lattice", () => {
  const options = [
    { seconds: 1.04 },
    { seconds: 2.08 },
    { seconds: 3.12 },
    { seconds: 5.2 },
  ];
  assert.equal(closestDurationIndex(options, 4.9), 3);
  assert.equal(closestDurationIndex(options, 2.4), 1);
});
