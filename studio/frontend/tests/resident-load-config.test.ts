// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  imageLoadConfigFromStatus,
  reapplyTargetFromStatus,
  videoLoadConfigFromStatus,
} from "../src/features/resident-load/resident-load-config.ts";

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

test("the native engine's resident load configuration cannot be read back from status", () => {
  // sd.cpp consumes cpu_offload, memory_mode and speed_mode, but collapses the first two into
  // offload flags and never reports memory_mode, so status cannot say what a reload should submit.
  // Reapply stays unavailable for a resident native build rather than replacing it with local
  // defaults: the parse must answer null, not a config assembled from what little status names.
  assert.equal(videoLoadConfigFromStatus({}), null);
  assert.equal(videoLoadConfigFromStatus({ resolved: null }), null);
  assert.equal(
    imageLoadConfigFromStatus({
      resolved: { cpu_offload: explicitControl(true) },
    }),
    null,
  );
});
