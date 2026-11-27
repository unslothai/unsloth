// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A refused FP8 load keeps the resident BF16 build, so its record key does not move: without a failure nonce the
// Precision select kept showing FP8 while Generate ran BF16 (NVIDIA feedback p.6, p.23).

import assert from "node:assert/strict";
import test from "node:test";

import { reseedAction } from "../src/lib/failed-load-reseed.ts";
import { readSrc } from "./helpers/kit.ts";

test("a failed load with the old build resident reseeds from that build", () => {
  assert.equal(reseedAction({ loaded: true }, true, true), "record");
});

test("a failed load that left nothing loaded shows Auto, not the refused pick", () => {
  assert.equal(reseedAction({ loaded: false }, false, true), "auto");
});

test("a plain unload keeps the user's picks, and an unknown status changes nothing", () => {
  assert.equal(reseedAction({ loaded: false }, false, false), "none");
  assert.equal(reseedAction(null, false, true), "none");
  assert.equal(reseedAction({ loaded: true }, false, true), "none");
});

for (const [page, path] of [
  ["images", "features/images/images-page.tsx"],
  ["video", "features/video/video-page.tsx"],
] as const) {
  const src = readSrc(path);

  test(`${page}: both failure paths bump the reseed after re-reading status`, () => {
    const bumps = src.match(/void refreshStatus\(\)\.finally\(\(\) => setFailedLoadReseed\(\(n\) => n \+ 1\)\);/g) ?? [];
    assert.equal(bumps.length, 2, "the polled phase: error branch and the start-request catch");
    const errorBranch = src.slice(src.indexOf('if (p.phase === "error") {'));
    assert.match(errorBranch.slice(0, errorBranch.indexOf("return;")), /setFailedLoadReseed/);
    const startCatch = src.slice(src.indexOf('reportLoadFailure(err, "Failed to start load")'));
    assert.match(startCatch.slice(0, 300), /setFailedLoadReseed/);
  });

  test(`${page}: the reseed effect runs on the failure nonce and snaps every Advanced select`, () => {
    assert.match(src, /\}, \[resolvedKey, failedLoadReseed\]\);/);
    const effect = src.slice(src.indexOf("const action = reseedAction(status, record != null, afterFailedLoad);"));
    const autoBranch = effect.slice(0, effect.indexOf("return;"));
    for (const setter of ["setTransformerQuant", "setMemoryMode", "setAttentionBackend", "setInt8Attention"]) {
      assert.match(autoBranch, new RegExp(`${setter}\\("auto"\\)`));
    }
    if (page === "images") assert.match(autoBranch, /setTextEncoderQuant\("auto"\)/);
  });
}
