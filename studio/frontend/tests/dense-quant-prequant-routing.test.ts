// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A dense-quant host does not download the bf16 shards of an official image pipeline. The backend
// resolves the auto load to the hosted pre-quantised transformer (unsloth/<Model>-FP8, holding
// <Model>-FP8.pt and <Model>-INT8.pt), so the row's download plan reports that artifact. The
// picker has to agree on two things or it misreports the load it is about to start:
//
//   1. the NAME says the precision that will run, from the host's own scheme list, and
//   2. the FIT is judged against the quantised resident size, so a card that only fits the
//      quantised form is routed to the official row rather than down the quant ladder.
//
// Both read `dense_quant_schemes`, which older backends do not send; absent is [] everywhere,
// and there nothing below changes.

import assert from "node:assert/strict";
import test from "node:test";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  artifactForRepoId,
  catalogToModelOptions,
  curatedArtifactFitsDevice,
  curatedDisplayNameFor,
  curatedRowLabelFor,
  groupForRepoId,
  pickDefaultArtifact,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

const Z_TURBO = "Tongyi-MAI/Z-Image-Turbo";
const QWEN_IMAGE = "Qwen/Qwen-Image";
const H3 = "MiniMaxAI/MiniMax-H3";
const notDownloaded = () => false;

/** A card with plenty of host RAM behind it: these pipelines are placed wholly on the card, so the
 *  RAM is never the deciding number and only the GPU figure varies between cases. */
const onCard = (gpuGb: number, denseQuantSchemes?: readonly string[]) => ({
  gpuGb,
  systemRamGb: 128,
  ...(denseQuantSchemes ? { denseQuantSchemes } : {}),
});

test("the catalog states the hosted checkpoint each official row would be fetched as", () => {
  for (const [id, repo, fp8, int8] of [
    [Z_TURBO, "unsloth/Z-Image-Turbo-FP8", 5.86, 5.86],
    [QWEN_IMAGE, "unsloth/Qwen-Image-FP8", 19.06, 31.73],
    [
      "black-forest-labs/FLUX.1-schnell",
      "unsloth/FLUX.1-schnell-FP8",
      11.09,
      14.13,
    ],
    ["krea/Krea-2-Turbo", "unsloth/Krea-2-Turbo-FP8", 11.95, 12.19],
  ] as const) {
    const hit = artifactForRepoId(id, IMAGE_CATALOG);
    assert.ok(hit, id);
    assert.equal(hit.artifact.prequantRepo, repo, id);
    assert.equal(hit.artifact.prequantSizeGb?.fp8, fp8, id);
    assert.equal(hit.artifact.prequantSizeGb?.int8, int8, id);
  }
});

test("the fit verdict is the quantised resident size on a host that runs a scheme", () => {
  // 30 GB dense wants 42.9 GB of card under the 70% rule; 24.4 GB pre-quantised wants 34.9.
  assert.equal(
    curatedArtifactFitsDevice(Z_TURBO, IMAGE_CATALOG, onCard(40)),
    false,
  );
  assert.equal(
    curatedArtifactFitsDevice(Z_TURBO, IMAGE_CATALOG, onCard(40, ["fp8"])),
    true,
  );
  assert.equal(
    curatedArtifactFitsDevice(Z_TURBO, IMAGE_CATALOG, onCard(40, ["int8"])),
    true,
  );
  // Still sized, not waved through: below the quantised form's own floor the answer is no.
  assert.equal(
    curatedArtifactFitsDevice(Z_TURBO, IMAGE_CATALOG, onCard(24, ["fp8"])),
    false,
  );
});

test("the two schemes are sized apart, since they are different artifacts", () => {
  // Qwen-Image is 19.06 GB at fp8 and 31.73 at int8, so one fits a 64 GB card and one does not.
  assert.equal(
    curatedArtifactFitsDevice(QWEN_IMAGE, IMAGE_CATALOG, onCard(64, ["fp8"])),
    true,
  );
  assert.equal(
    curatedArtifactFitsDevice(QWEN_IMAGE, IMAGE_CATALOG, onCard(64, ["int8"])),
    false,
  );
});

test("a host with no scheme, and a row with no hosted checkpoint, are unchanged", () => {
  // An older backend sends no list at all; a capable host may send an empty one. Both keep the
  // dense rule rather than claiming a download that was never resolved.
  assert.equal(
    curatedArtifactFitsDevice(Z_TURBO, IMAGE_CATALOG, onCard(40, [])),
    false,
  );
  // FLUX.1-dev has no hosted artifact in the catalog and SDXL is a UNet the quantiser skips.
  for (const id of ["black-forest-labs/FLUX.1-dev", "stabilityai/sdxl-turbo"]) {
    assert.equal(
      curatedArtifactFitsDevice(id, IMAGE_CATALOG, onCard(40, ["fp8"])),
      curatedArtifactFitsDevice(id, IMAGE_CATALOG, onCard(40)),
      id,
    );
  }
});

test("the router sends a card that only fits the quantised form to the official row", () => {
  const group = groupForRepoId(Z_TURBO, IMAGE_CATALOG);
  assert.ok(group);
  // 40 GB card, no scheme: the bf16 row does not fit, so the quant ladder takes it.
  assert.equal(
    pickDefaultArtifact(group, { ...onCard(40), isDownloaded: notDownloaded })
      .format,
    "bnb-4bit",
  );
  assert.equal(
    pickDefaultArtifact(group, {
      ...onCard(40, ["fp8"]),
      isDownloaded: notDownloaded,
    }).repoId,
    Z_TURBO,
  );
  // Below the quantised floor the ladder is still the right answer.
  assert.equal(
    pickDefaultArtifact(group, {
      ...onCard(24, ["fp8"]),
      isDownloaded: notDownloaded,
    }).format,
    "bnb-4bit",
  );
});

test("the row names the scheme the host runs", () => {
  assert.equal(
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", ["fp8"])?.name,
    "Z-Image-Turbo (Fast FP8)",
  );
  // An Ampere card runs int8, and says so.
  assert.equal(
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", ["int8"])?.name,
    "Z-Image-Turbo (Fast INT8)",
  );
  // A capable host that names no scheme keeps the bare ordering qualifier.
  assert.equal(
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", [])?.name,
    "Z-Image-Turbo (Fast)",
  );
  assert.equal(
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant")?.name,
    "Z-Image-Turbo (Fast)",
  );
  // The trigger reads the same as the row, open or closed.
  assert.equal(
    curatedDisplayNameFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", ["fp8"]),
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", ["fp8"])?.name,
  );
  assert.equal(
    catalogToModelOptions(IMAGE_CATALOG, "dense-quant", ["int8"]).find(
      (option) => option.id === Z_TURBO,
    )?.name,
    "Z-Image-Turbo (Fast INT8)",
  );
});

// H3's pipeline row read "Fast FP8" before it was flattened to "Fast". It is back, and now
// data-driven, so it cannot say FP8 on a host that would run INT8.
test("the H3 pipeline row names its precision again", () => {
  assert.deepEqual(
    curatedRowLabelFor(H3, VIDEO_CATALOG, "dense-quant", ["fp8"]),
    {
      name: "MiniMax H3 (Fast FP8)",
      tags: ["BF16"],
    },
  );
  assert.deepEqual(
    curatedRowLabelFor(H3, VIDEO_CATALOG, "dense-quant", ["int8"]),
    {
      name: "MiniMax H3 (Fast INT8)",
      tags: ["BF16"],
    },
  );
  assert.equal(
    curatedRowLabelFor(
      "unsloth/MiniMax-H3-GGUF",
      VIDEO_CATALOG,
      "dense-quant",
      ["fp8"],
    )?.name,
    "MiniMax-H3-GGUF (Slow)",
  );
});

test("the scheme reaches the name and never the chip", () => {
  // Chips describe the artifact as published, so they are identical on every host: the stored
  // precision is what tells two rows of one group apart, and the host cannot rewrite it.
  for (const schemes of [[], ["fp8"], ["int8"]]) {
    for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG]) {
      for (const group of catalog) {
        for (const artifact of group.artifacts) {
          assert.deepEqual(
            curatedRowLabelFor(artifact.repoId, catalog, "dense-quant", schemes)
              ?.tags ?? [],
            curatedRowLabelFor(artifact.repoId, catalog, "accelerated")?.tags ??
              [],
            `${artifact.repoId} ${schemes.join(",") || "none"}`,
          );
        }
      }
    }
  }
});
