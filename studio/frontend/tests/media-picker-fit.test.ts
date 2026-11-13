// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { curatedPickerFit, mediaGgufPickerStatus } from "../src/features/model-picker/components/model-selector/media-picker-fit.ts";
import { IMAGE_CATALOG, VIDEO_CATALOG, AUDIO_CATALOG, pickDefaultArtifact, groupForRepoId } from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { curatedBudget, curatedBudgetText } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

const id = "unsloth/Z-Image-Turbo";

test("discrete GPU media can fit combined memory and warn about offloading", () => {
  const fit = curatedPickerFit(id, IMAGE_CATALOG, { gpuGb: 12, systemRamGb: 32 });
  assert.equal(fit?.fits, true);
  assert.equal(fit?.status, "offload");
  assert.equal(fit?.device, "GPU + RAM");
  assert.equal(fit?.deviceGb, 44);
});

test("only insufficient combined capacity gets an exceeds verdict", () => {
  const fit = curatedPickerFit(id, IMAGE_CATALOG, { gpuGb: 12, systemRamGb: 8 });
  assert.equal(fit?.fits, false);
  assert.equal(fit?.status, "exceeds");
  const budget = fit && curatedBudget(fit);
  assert.ok(budget);
  assert.match(curatedBudgetText(19, 12, budget), /20GB combined GPU and available RAM/);
});

test("Spark and other shared pools do not count RAM twice", () => {
  const fit = curatedPickerFit(id, IMAGE_CATALOG, { gpuGb: 16, systemRamGb: 32, hostPooledMemory: true });
  assert.equal(fit?.fits, false);
  assert.equal(fit?.deviceGb, 16);
  assert.equal(fit?.status, "exceeds");
});

test("resident models need no offload warning", () => {
  assert.equal(curatedPickerFit(id, IMAGE_CATALOG, { gpuGb: 48, systemRamGb: 32 })?.status, null);
});

test("a host with no GPU keeps the RAM verdict, not an offload warning", () => {
  const fit = curatedPickerFit(id, IMAGE_CATALOG, { gpuGb: 0, systemRamGb: 64 });
  assert.equal(fit?.fits, true);
  assert.equal(fit?.status, null);
  assert.equal(fit?.device, "RAM");
  assert.equal(curatedPickerFit(id, IMAGE_CATALOG, { gpuGb: 0, systemRamGb: 16 })?.status, "exceeds");
});

test("video uses the same combined-memory rule", () => {
  assert.equal(curatedPickerFit("Wan-AI/Wan2.1-T2V-14B-Diffusers", VIDEO_CATALOG, { gpuGb: 24, systemRamGb: 64 })?.status, "offload");
});

test("media GGUF offload and combined shortage have distinct display states", () => {
  assert.equal(mediaGgufPickerStatus("partial"), "offload");
  assert.equal(mediaGgufPickerStatus("oom"), "exceeds");
  assert.equal(mediaGgufPickerStatus("fits"), "fits");
});

test("audio and Auto artifact routing retain their existing fit policies", () => {
  assert.equal(curatedPickerFit("unsloth/whisper-large-v3", AUDIO_CATALOG, { gpuGb: 1, systemRamGb: 5 })?.fits, false);
  const group = groupForRepoId(id, IMAGE_CATALOG);
  assert.ok(group);
  assert.equal(pickDefaultArtifact(group, { gpuGb: 8, systemRamGb: 32, isDownloaded: () => false }).format, "gguf");
});

test("host RAM cannot stand in for a measured tier's GPU floor", () => {
  const H3 = "MiniMaxAI/MiniMax-H3";
  for (const gpuGb of [4, 24]) {
    const fit = curatedPickerFit(H3, VIDEO_CATALOG, { gpuGb, systemRamGb: 256 });
    assert.equal(fit?.fits, false, `${gpuGb} GB GPU`);
    assert.equal(fit?.status, "exceeds", `${gpuGb} GB GPU`);
  }
  assert.equal(curatedPickerFit(H3, VIDEO_CATALOG, { gpuGb: 80, systemRamGb: 256 })?.status, "offload");
});

test("measured discrete offload tiers cannot invent a second pool on shared hosts", () => {
  const fit = curatedPickerFit("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, { gpuGb: 4, systemRamGb: 64, hostPooledMemory: true });
  assert.equal(fit?.fits, false);
  assert.equal(fit?.status, "exceeds");
});
