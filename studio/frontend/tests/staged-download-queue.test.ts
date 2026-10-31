// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";
registerBundlerResolver();
const { queuedStagedEntries, publishStagedQueue, useStagedDownloadQueues } =
  await import("../src/features/hub/download-manager/staged-download-queue.ts");

const encoder = {
  repoId: "encoder",
  files: ["text_encoder/model.safetensors"],
  bytes: 100,
  checkpoint: false,
};
const decoder = {
  repoId: "decoder",
  files: ["vae/model.safetensors"],
  bytes: 20,
  checkpoint: false,
};
const job = {
  kind: "model" as const,
  repoId: "encoder",
  variant: "@diffusion",
  state: "running" as const,
  scopedFiles: encoder.files,
};

test("image/video and audio stages stay visible while only the active transfer is hidden", () => {
  const queues = {
    image: { scopeId: "diffusion", entries: [encoder, decoder] },
    audio: { scopeId: "audio", entries: [encoder] },
  };
  assert.deepEqual(
    queuedStagedEntries(queues, { job }).map((e) => e.repoId),
    ["decoder", "encoder"],
  );
  assert.equal(queuedStagedEntries(queues, {}).length, 3);
});

test("another file set, scope, or dataset transfer cannot hide a pending model stage", () => {
  const queues = { image: { scopeId: "diffusion", entries: [encoder] } };
  for (const other of [
    { ...job, variant: "@hub-assets" },
    { ...job, scopedFiles: ["other.safetensors"] },
    { ...job, kind: "dataset" as const },
  ]) {
    assert.equal(queuedStagedEntries(queues, { other }).length, 1);
  }
  assert.equal(
    queuedStagedEntries(queues, { job: { ...job, state: "cancelling" } })
      .length,
    0,
  );
  assert.equal(
    queuedStagedEntries(queues, { job: { ...job, state: "complete" } }).length,
    1,
  );
});

test("GGUF audio stages match their variant without requiring scoped files", () => {
  const queues = {
    audio: {
      scopeId: "audio",
      entries: [{ ...encoder, ggufVariant: "Q4_K_M" }],
    },
  };
  assert.equal(
    queuedStagedEntries(queues, {
      job: { ...job, variant: "Q4_K_M", scopedFiles: undefined },
    }).length,
    0,
  );
  assert.equal(
    queuedStagedEntries(queues, { job: { ...job, variant: "Q8_0" } }).length,
    1,
  );
});

test("replacing, advancing, and retiring one owner preserves other queues in the same scope", () => {
  publishStagedQueue("images", {
    scopeId: "diffusion",
    entries: [encoder, decoder],
  });
  publishStagedQueue("video", { scopeId: "diffusion", entries: [encoder] });
  publishStagedQueue("images", { scopeId: "diffusion", entries: [decoder] });
  assert.equal(
    useStagedDownloadQueues.getState().queues.images.entries.length,
    1,
  );
  publishStagedQueue("images", null);
  assert.deepEqual(Object.keys(useStagedDownloadQueues.getState().queues), [
    "video",
  ]);
  publishStagedQueue("video", { scopeId: "diffusion", entries: [] });
  assert.deepEqual(useStagedDownloadQueues.getState().queues, {});
});
