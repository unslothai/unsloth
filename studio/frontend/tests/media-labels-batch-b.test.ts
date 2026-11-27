// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Labels from the NVIDIA feedback: multi-GPU means one computer (p.13, p.29), and training's NF4 is not NVFP4 (p.21).

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

for (const path of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  test(`${path}: the GPU hint says multi-GPU is the cards in one computer`, () => {
    const src = readSrc(path);
    const select = src.slice(src.indexOf('label="GPU"'));
    const hint = select.slice(0, select.indexOf("value={selectedGpu}"));
    assert.match(hint, /Multi-GPU means the cards inside this one computer/);
    assert.match(hint, /two machines \(for example two DGX Sparks\)[^"]*is not part of this release/);
    // Automatic placement does use a second card, so the hint must not say the model never spans cards.
    assert.doesNotMatch(hint, /never split across cards/);
  });
}

test("training names NF4 as bitsandbytes, not NVFP4", () => {
  const panel = readSrc("features/images/train/diffusion-train-panel.tsx");
  assert.match(panel, /if \(m === "nf4"\) return "NF4 \(bitsandbytes, not NVFP4\)";/);
  assert.match(panel, /NF4 is the bitsandbytes 4-bit format QLoRA uses, not NVIDIA's NVFP4/);
});
