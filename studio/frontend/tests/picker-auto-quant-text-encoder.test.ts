// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  IMAGE_CATALOG,
  artifactForRepoId,
  catalogGroupFitsDevice,
  curatedArtifactFit,
  groupForRepoId,
  pickDefaultArtifact,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

const KLEIN = "black-forest-labs/flux-3-image-klein";
const KLEIN_BASE = "black-forest-labs/flux-3-image-klein-base";
const QWEN_21 = "unsloth/Qwen-Image-2.1";
const QWEN_21_TURBO = "Qwen/Qwen-Image-2.1-Turbo";
const KONTEXT = "unsloth/FLUX.1-Kontext-dev";
const notDownloaded = () => false;
const GIB = 1024 ** 3;

const card = (gpuGb: number, denseQuantSchemes: readonly string[] = ["int8", "fp8"]) => ({
  gpuGb,
  systemRamGb: 64,
  denseQuantSchemes,
});

const pick = (id: string, gpuGb: number, schemes?: readonly string[]) => {
  const group = groupForRepoId(id, IMAGE_CATALOG);
  assert.ok(group, id);
  return pickDefaultArtifact(group, { ...card(gpuGb, schemes), isDownloaded: notDownloaded }).label;
};

const sizeOf = (id: string, gpuGb: number, schemes?: readonly string[]) => {
  const fit = curatedArtifactFit(id, IMAGE_CATALOG, card(gpuGb, schemes));
  assert.ok(fit?.sizeGb !== undefined, id);
  return fit.sizeGb;
};

test("each auto-quantised row states the hosted encoder file the backend loads", () => {
  // Hub file sizes (bytes): the bf16 encoder shards against the file te_prequant_sources resolves for the
  // family's te_quant_auto on a CUDA bf16 host.
  for (const [id, dense, quantised] of [
    [KLEIN, 8_875_719_344, 5_247_076_320],
    [KLEIN_BASE, 8_875_719_344, 5_247_076_320],
    [QWEN_21, 17_534_340_584, 9_349_769_248],
    [QWEN_21_TURBO, 17_534_340_584, 9_349_769_248],
    [KONTEXT, 9_524_648_584, 6_163_991_880],
  ] as const) {
    const te = artifactForRepoId(id, IMAGE_CATALOG)?.artifact.autoQuantTextEncoderGb;
    assert.ok(te, id);
    assert.ok(Math.abs(te.denseGb - dense / GIB) < 0.01, `${id} dense`);
    assert.ok(Math.abs(te.quantisedGb - quantised / GIB) < 0.01, `${id} quantised`);
  }
});

test("rows whose encoder an auto load keeps bf16 carry no quantised companion", () => {
  // No te_quant_auto (ComfyUI keeps the encoder bf16), or an auto int8 file that is not hosted, so the backend
  // loads the dense encoder.
  for (const id of [
    "unsloth/Z-Image-Turbo",
    "unsloth/FLUX.1-schnell",
    "unsloth/FLUX.1-dev",
    "black-forest-labs/FLUX.1-Krea-dev",
    "unsloth/Qwen-Image",
    "unsloth/Qwen-Image-2512",
    "unsloth/Qwen-Image-Edit-2511",
    "krea/Krea-2-Turbo",
    "Alpha-VLLM/Lumina-Image-2.0",
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    "HiDream-ai/HiDream-I1-Full",
  ]) {
    assert.equal(
      artifactForRepoId(id, IMAGE_CATALOG)?.artifact.autoQuantTextEncoderGb,
      undefined,
      id,
    );
  }
});

test("the dense row is sized with the quantised encoder, not the bf16 one", () => {
  // Klein: 6.0 int8 DiT + (18.92 - 10.33 bf16 DiT) companions, less the 3.38 GiB the int8 encoder saves.
  assert.ok(Math.abs(sizeOf(KLEIN, 24) - 11.2) < 0.02, String(sizeOf(KLEIN, 24)));
  assert.ok(Math.abs(sizeOf(KLEIN_BASE, 24) - 10.61) < 0.02, String(sizeOf(KLEIN_BASE, 24)));
  assert.ok(Math.abs(sizeOf(QWEN_21, 32) - 19.39) < 0.02, String(sizeOf(QWEN_21, 32)));
  assert.ok(Math.abs(sizeOf(QWEN_21_TURBO, 32) - 19.39) < 0.02);
  // Kontext has no hosted DiT, so the bf16 DiT stays and only T5-XXL drops to fp8: 32 - 3.13.
  assert.ok(Math.abs(sizeOf(KONTEXT, 32) - 28.87) < 0.02, String(sizeOf(KONTEXT, 32)));
});

test("a card the dense row fits with its quantised encoder picks it over the GGUF", () => {
  assert.equal(pick(KLEIN_BASE, 16), "BF16");
  assert.equal(pick(KLEIN_BASE, 16, ["fp8", "int8"]), "BF16");
  assert.equal(pick(QWEN_21, 32), "BF16");
  assert.equal(pick(QWEN_21_TURBO, 32), "BF16");
  // Still too big: unchanged.
  assert.equal(pick(QWEN_21, 24), "GGUF");
  assert.equal(pick(KLEIN, 12), "GGUF");
  const kleinBase = groupForRepoId(KLEIN_BASE, IMAGE_CATALOG);
  assert.ok(kleinBase);
  assert.equal(catalogGroupFitsDevice(kleinBase, card(16), notDownloaded), true);
});

test("a host that reports no dense scheme keeps the bf16 sizing", () => {
  // No dense scheme = not a CUDA bf16 card the backend's encoder cast runs on, so nothing is assumed.
  for (const [id, dense] of [
    [KLEIN, 18.92],
    [QWEN_21, 33],
    [KONTEXT, 32],
  ] as const) {
    assert.equal(curatedArtifactFit(id, IMAGE_CATALOG, card(80, []))?.sizeGb, dense, id);
  }
  assert.equal(pick(QWEN_21, 32, []), "GGUF");
});
