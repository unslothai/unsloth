// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// MiniMax-H3's Diffusers INT8 row versus its GGUF (stable-diffusion.cpp) row. The catalog's static
// tiers send a 48 GB card with ~77 GiB available RAM to the GGUF row, which measured 3.2x slower than
// the INT8 path. The backend now reports the tiers it can actually run (`/api/system.diffusers_offload_tiers`);
// the picker unions them with the catalog's, so they widen and never narrow.

import assert from "node:assert/strict";
import test from "node:test";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  catalogGroupFitsDevice,
  curatedArtifactFitsDevice,
  groupForRepoId,
  pickDefaultArtifact,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

const H3 = "MiniMaxAI/MiniMax-H3";
const h3Group = groupForRepoId(H3, VIDEO_CATALOG);
if (!h3Group)
  throw new Error("MiniMax-H3 group missing from the video catalog");
const h3 = h3Group;
const notDownloaded = () => false;
// The shape /api/system reports, after normalisation. Values mirror the backend table.
const EXTRA = {
  "minimaxai/minimax-h3": [
    { gpuGb: 14, systemRamGb: 66, requiresQuantisedStreaming: true },
  ],
};

// GiB as the picker sees them: nvidia-smi MiB / 1024, psutil available / 1024**3.
const HOSTS = {
  ada48: { gpuGb: 47.99, systemRamGb: 77 },
  rtx3090: { gpuGb: 24, systemRamGb: 80 },
  a100_40: { gpuGb: 39.39, systemRamGb: 72 },
  g4: { gpuGb: 95.59, systemRamGb: 160 },
  // 16 / 12 GB cards on a 96 GB RAM desktop (~88 GiB available).
  card16: { gpuGb: 15.99, systemRamGb: 88 },
  card12: { gpuGb: 12, systemRamGb: 88 },
};

test("without backend tiers every listed consumer host still lands on GGUF (base routing)", () => {
  for (const [name, host] of Object.entries(HOSTS)) {
    if (name === "g4") continue;
    const pick = pickDefaultArtifact(h3, {
      ...host,
      quantisedStreaming: true,
      isDownloaded: notDownloaded,
    });
    assert.equal(pick.format, "gguf", name);
  }
});

test("backend tiers make the Diffusers INT8 row the default where it runs", () => {
  for (const name of [
    "ada48",
    "rtx3090",
    "a100_40",
    "g4",
    "card16",
  ] as const) {
    const budget = {
      ...HOSTS[name],
      quantisedStreaming: true,
      extraOffloadFitTiers: EXTRA,
    };
    assert.equal(
      pickDefaultArtifact(h3, { ...budget, isDownloaded: notDownloaded })
        .format,
      "bf16",
      name,
    );
    assert.equal(
      curatedArtifactFitsDevice(H3, VIDEO_CATALOG, budget),
      true,
      name,
    );
    assert.equal(catalogGroupFitsDevice(h3, budget, notDownloaded), true, name);
  }
});

test("GGUF stays the fallback when RAM, VRAM or streaming cannot fit", () => {
  const cases = [
    { gpuGb: 8, systemRamGb: 120, quantisedStreaming: true },
    { gpuGb: 48, systemRamGb: 40, quantisedStreaming: true },
    // The page's default request (1344x768, default length) needs ~13.5 GiB, so a 12 GB card keeps GGUF as its
    // default; the Diffusers row still renders 960x544 there when chosen.
    { gpuGb: 12, systemRamGb: 88, quantisedStreaming: true },
    // A 64 GB RAM desktop (~58 GiB available) is under the measured 70 GB streamed-set floor.
    { gpuGb: 12, systemRamGb: 58, quantisedStreaming: true },
    { gpuGb: 24, systemRamGb: 58, quantisedStreaming: true },
    { gpuGb: 48, systemRamGb: 120, quantisedStreaming: false },
    { gpuGb: 48, systemRamGb: 120, quantisedStreaming: undefined },
  ];
  for (const c of cases) {
    const pick = pickDefaultArtifact(h3, {
      ...c,
      extraOffloadFitTiers: EXTRA,
      isDownloaded: notDownloaded,
    });
    assert.equal(pick.format, "gguf", JSON.stringify(c));
  }
});

test("backend tiers only widen: a host the catalog admits stays admitted with none reported", () => {
  const g4 = { ...HOSTS.g4, quantisedStreaming: true };
  assert.equal(curatedArtifactFitsDevice(H3, VIDEO_CATALOG, g4), true);
  assert.equal(
    curatedArtifactFitsDevice(H3, VIDEO_CATALOG, {
      ...g4,
      extraOffloadFitTiers: {},
    }),
    true,
  );
});

test("backend tiers never replace the size rule of an artifact without catalog tiers", () => {
  const id = "Qwen/Qwen-Image-2512";
  const budget = { gpuGb: 24, systemRamGb: 64, quantisedStreaming: true };
  const before = curatedArtifactFitsDevice(id, IMAGE_CATALOG, budget);
  assert.equal(before, false);
  assert.equal(
    curatedArtifactFitsDevice(id, IMAGE_CATALOG, {
      ...budget,
      extraOffloadFitTiers: {
        [id.toLowerCase()]: [{ gpuGb: 1, systemRamGb: 1 }],
      },
    }),
    before,
  );
});
