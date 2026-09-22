// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A dense-quant host loads an official image pipeline from the hosted pre-quantised transformer, so
// the picker must name that precision AND judge fit against the quantised resident size. Both read
// `dense_quant_schemes`; older backends omit it, and absent is [] with nothing changed.

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_CATALOG,
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  artifactForRepoId,
  catalogToModelOptions,
  curatedArtifactFit,
  curatedArtifactFitsDevice,
  curatedDisplayNameFor,
  curatedRowLabelFor,
  groupForRepoId,
  pickDefaultArtifact,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

const Z_TURBO = "Tongyi-MAI/Z-Image-Turbo";
const QWEN_IMAGE = "Qwen/Qwen-Image";
const QWEN_2512 = "Qwen/Qwen-Image-2512";
const QWEN_21 = "Qwen/Qwen-Image-2.1";
const H3 = "MiniMaxAI/MiniMax-H3";
const notDownloaded = () => false;

/** A card with plenty of host RAM: only the GPU figure varies between cases. */
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
  assert.equal(
    curatedArtifactFitsDevice(Z_TURBO, IMAGE_CATALOG, onCard(24, ["fp8"])),
    false,
  );
});

test("the two schemes are sized apart, since they are different artifacts", () => {
  assert.equal(
    curatedArtifactFitsDevice(QWEN_IMAGE, IMAGE_CATALOG, onCard(64, ["fp8"])),
    true,
  );
  assert.equal(
    curatedArtifactFitsDevice(QWEN_IMAGE, IMAGE_CATALOG, onCard(64, ["int8"])),
    false,
  );
});

test("the 2512 pick is judged by the checkpoint the backend seeds for it, not by bf16", () => {
  assert.equal(
    curatedArtifactFitsDevice(QWEN_2512, IMAGE_CATALOG, onCard(64)),
    false,
  );
  assert.equal(
    curatedArtifactFitsDevice(QWEN_2512, IMAGE_CATALOG, onCard(64, ["int8"])),
    true,
  );
  const group = groupForRepoId(QWEN_2512, IMAGE_CATALOG);
  assert.ok(group);
  assert.equal(
    pickDefaultArtifact(group, {
      ...onCard(64, ["int8"]),
      isDownloaded: notDownloaded,
    }).repoId,
    QWEN_2512,
  );
  assert.equal(
    pickDefaultArtifact(group, { ...onCard(64), isDownloaded: notDownloaded })
      .format,
    "bnb-4bit",
  );
});

test("a host with no scheme, and a row with no hosted checkpoint, are unchanged", () => {
  assert.equal(
    curatedArtifactFitsDevice(Z_TURBO, IMAGE_CATALOG, onCard(40, [])),
    false,
  );
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
  assert.equal(
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", ["int8"])?.name,
    "Z-Image-Turbo (Fast FP8)",
  );
  assert.equal(
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", [])?.name,
    "Z-Image-Turbo (Fast)",
  );
  assert.equal(
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant")?.name,
    "Z-Image-Turbo (Fast)",
  );
  assert.equal(
    curatedDisplayNameFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", ["fp8"]),
    curatedRowLabelFor(Z_TURBO, IMAGE_CATALOG, "dense-quant", ["fp8"])?.name,
  );
  assert.equal(
    catalogToModelOptions(IMAGE_CATALOG, "dense-quant", ["int8"]).find(
      (option) => option.id === Z_TURBO,
    )?.name,
    "Z-Image-Turbo (Fast FP8)",
  );
});

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
      name: "MiniMax H3 (Fast FP8)",
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

test("every diffusion GGUF row is tagged Slow, not only H3's", () => {
  for (const [repoId, catalog, name] of [
    ["unsloth/Z-Image-Turbo-GGUF", IMAGE_CATALOG, "Z-Image-Turbo-GGUF (Slow)"],
    ["unsloth/FLUX.1-schnell-GGUF", IMAGE_CATALOG, "FLUX.1-schnell-GGUF (Slow)"],
    ["unsloth/LTX-2.3-GGUF", VIDEO_CATALOG, "LTX-2.3-GGUF (Slow)"],
  ] as const) {
    assert.equal(
      curatedRowLabelFor(repoId, catalog, "dense-quant", ["fp8"])?.name,
      name,
      repoId,
    );
    assert.equal(
      curatedRowLabelFor(repoId, catalog, "accelerated")?.name,
      name,
      repoId,
    );
    assert.equal(
      curatedDisplayNameFor(repoId, catalog, "dense-quant", ["fp8"]),
      name,
      repoId,
    );
  }
});

test("a host with no accelerator is not told which row is slow", () => {
  // Off an accelerator the GGUF is the only row that runs, so the qualifier would read as a
  // warning about the user's one option rather than a comparison.
  for (const host of ["gguf-only", "unknown"] as const) {
    for (const [repoId, catalog] of [
      ["unsloth/Z-Image-Turbo-GGUF", IMAGE_CATALOG],
      ["unsloth/LTX-2.3-GGUF", VIDEO_CATALOG],
      ["unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG],
    ] as const) {
      const name = curatedRowLabelFor(repoId, catalog, host, ["fp8"])?.name;
      assert.equal(name?.includes("Slow"), false, `${repoId} ${host}`);
    }
  }
});

test("an audio GGUF keeps its plain name, since it has no dense sibling", () => {
  for (const repoId of [
    "unsloth/orpheus-3b-0.1-ft-GGUF",
    "unslothai/Qwen3-ASR-0.6B-GGUF",
  ]) {
    for (const group of AUDIO_CATALOG) {
      if (!group.artifacts.some((a) => a.repoId === repoId)) continue;
      const name = curatedRowLabelFor(repoId, AUDIO_CATALOG, "dense-quant", [
        "fp8",
      ])?.name;
      assert.equal(name?.includes("Slow"), false, repoId);
    }
  }
});

test("the scheme reaches the name and never the chip", () => {
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

// Show the verdict's estimate, not the dense catalog size.
test("Qwen-Image-2.1 is badged with the size its verdict used", () => {
  const card = onCard(22.49, ["int8", "fp8"]);
  const fit = curatedArtifactFit(QWEN_21, IMAGE_CATALOG, card);
  assert.ok(fit?.sizeGb !== undefined);
  assert.ok(Math.abs(fit.sizeGb - 27.01) < 0.01, String(fit.sizeGb));
  assert.equal(fit.allowanceGb, 22.49 * 0.7);
  assert.equal(fit.fits, false);
  assert.equal(curatedArtifactFitsDevice(QWEN_21, IMAGE_CATALOG, card), fit.fits);
  // No dense-quant scheme reported: the dense figure.
  assert.equal(curatedArtifactFit(QWEN_21, IMAGE_CATALOG, onCard(22.49, []))?.sizeGb, 33);
});

test("a transcription row judged on RAM names RAM as the budget's device", () => {
  const WHISPER = "unsloth/whisper-large-v3";
  const onRam = curatedArtifactFit(WHISPER, AUDIO_CATALOG, { gpuGb: 1, systemRamGb: 5 });
  assert.equal(onRam?.fits, false);
  assert.equal(onRam?.device, "RAM");
  assert.equal(onRam?.deviceGb, 5);
  assert.equal(onRam?.allowanceGb, 5 * 0.7);
  const onGpu = curatedArtifactFit(WHISPER, AUDIO_CATALOG, { gpuGb: 5, systemRamGb: 1 });
  assert.equal(onGpu?.device, "GPU");
  assert.equal(onGpu?.deviceGb, 5);
});

test("Qwen-Image-2.1 routes every card as on main", () => {
  const group = groupForRepoId(QWEN_21, IMAGE_CATALOG);
  assert.ok(group);
  const pick = (gpuGb: number) =>
    pickDefaultArtifact(group, {
      ...onCard(gpuGb, ["int8"]),
      isDownloaded: notDownloaded,
    }).repoId;
  // Include 36 GiB to catch changes just below the existing routing threshold.
  assert.equal(pick(24), "unsloth/Qwen-Image-2.1-GGUF");
  assert.equal(pick(31.84), "unsloth/Qwen-Image-2.1-GGUF");
  assert.equal(pick(36), "unsloth/Qwen-Image-2.1-GGUF");
  assert.equal(pick(39.5), QWEN_21);
});
