// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  chainDynamicDefaultRollback,
  closestDurationIndex,
  closestResolutionIndex,
  getBuiltinVariantName,
  imageLoadConfigFromStatus,
  mergeUntouchedParams,
  reapplyTargetFromStatus,
  residentLoadConfigIsKnown,
  videoLoadConfigFromStatus,
} from "../src/features/generation-presets/preset-policy.ts";

const explicitControl = (
  requested: string | boolean,
  value: string | boolean | null = requested,
) => ({
  value,
  requested,
  source: "explicit" as const,
  status: "applied" as const,
  reason: "",
});

const automaticControl = (value: string | boolean | null = "off") => ({
  value,
  requested: null,
  source: "auto" as const,
  status: "applied" as const,
  reason: "",
});

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
  const rollback = chainDynamicDefaultRollback(previous, next);
  assert.equal(rollback(), true);
  assert.deepEqual(calls, ["next", "previous"]);
});

test("a stale newer rollback cannot continue into an older transaction", () => {
  let previousCalled = false;
  const rollback = chainDynamicDefaultRollback(
    () => {
      previousCalled = true;
      return true;
    },
    () => false,
  );
  assert.equal(rollback(), false);
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

test("resident image status reconstructs every load option Reapply submits", () => {
  assert.deepEqual(
    imageLoadConfigFromStatus({
      resolved: {
        speed_mode: explicitControl("max"),
        transformer_quant: explicitControl("off"),
        attention_backend: explicitControl("flash3", "_native_flash3"),
        memory_mode: explicitControl("low_vram", "sequential"),
        transformer_cache: explicitControl("fbcache"),
        cpu_offload: explicitControl(true),
      },
    }),
    {
      speedMode: "max",
      transformerQuant: "none",
      attentionBackend: "flash3",
      memoryMode: "low_vram",
      transformerCache: "fbcache",
      cpuOffload: true,
    },
  );
});

test("resident video status retains explicit speed and step-cache choices", () => {
  const auto = automaticControl();
  assert.deepEqual(
    videoLoadConfigFromStatus({
      resolved: {
        speed_mode: explicitControl("eager"),
        transformer_quant: auto,
        attention_backend: auto,
        memory_mode: auto,
        transformer_cache: explicitControl("off"),
      },
    }),
    {
      speedMode: "eager",
      transformerQuant: "auto",
      attentionBackend: "auto",
      memoryMode: "auto",
      transformerCache: "off",
    },
  );
});

test("automatic resident offload does not become an explicit legacy flag", () => {
  const auto = automaticControl();
  const config = imageLoadConfigFromStatus({
    resolved: {
      speed_mode: auto,
      transformer_quant: auto,
      attention_backend: auto,
      memory_mode: auto,
      transformer_cache: auto,
      cpu_offload: automaticControl(true),
    },
  });
  assert.equal(config?.cpuOffload, false);
});

test("saving over Default creates a protected custom variant", () => {
  const used = new Set(["Default", "Default 1", "Portrait"]);
  assert.equal(getBuiltinVariantName(used), "Default 2");
});

test("video resolution mapping prioritizes aspect before area", () => {
  const options: [number, number][] = [
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

test("a build reporting no resolved record has no load options to preserve", () => {
  // The native sd.cpp engine reports no resolved record at all, so Reapply cannot be replacing an
  // explicit choice it never read: offering it there matches a page-initiated load of the same build.
  assert.equal(residentLoadConfigIsKnown({}, null), true);
  assert.equal(residentLoadConfigIsKnown({ resolved: null }, null), true);
  // A diffusers build names them, so an incomplete parse must keep Reapply off.
  assert.equal(
    residentLoadConfigIsKnown(
      { resolved: { memory_mode: automaticControl("balanced") } },
      null,
    ),
    false,
  );
  assert.equal(
    residentLoadConfigIsKnown(
      { resolved: { memory_mode: automaticControl("balanced") } },
      { memoryMode: "balanced" },
    ),
    true,
  );
});
