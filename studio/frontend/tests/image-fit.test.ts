// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MAX_UPLOAD_IMAGE_SIDE,
  fitOutputType,
  fitWithinMaxSide,
} from "../src/lib/image-fit.ts";

test("the upload cap matches the backend decode limit", () => {
  const backend = readFileSync(
    new URL("../../backend/core/inference/diffusion.py", import.meta.url),
    "utf8",
  );
  const match = /def decode_b64_image\([\s\S]*?max_side: int = (\d+)/.exec(backend);
  assert.ok(match, "decode_b64_image max_side default not found");
  assert.equal(MAX_UPLOAD_IMAGE_SIDE, Number(match[1]));
});

test("a 24MP iPhone photo is scaled to fit at the same aspect", () => {
  assert.deepEqual(fitWithinMaxSide(5712, 4284, 4096), { width: 4096, height: 3072 });
  assert.deepEqual(fitWithinMaxSide(4284, 5712, 4096), { width: 3072, height: 4096 });
  assert.deepEqual(fitWithinMaxSide(8064, 6048, 4096), { width: 4096, height: 3072 });
});

test("images that already fit are left alone, so a second pass is a no-op", () => {
  assert.equal(fitWithinMaxSide(4032, 3024, 4096), null);
  assert.equal(fitWithinMaxSide(4096, 4096, 4096), null);
  const once = fitWithinMaxSide(9999, 7777, 4096);
  assert.ok(once);
  assert.equal(fitWithinMaxSide(once.width, once.height, 4096), null);
});

test("extreme aspects never round to zero or past the cap", () => {
  assert.deepEqual(fitWithinMaxSide(20000, 3, 4096), { width: 4096, height: 1 });
  assert.equal(fitWithinMaxSide(0, 5000, 4096), null);
});

test("only JPEG sources are re-encoded as JPEG", () => {
  assert.equal(fitOutputType("data:image/jpeg;base64,AAAA"), "image/jpeg");
  assert.equal(fitOutputType("data:image/png;base64,AAAA"), "image/png");
  assert.equal(fitOutputType("data:image/webp;base64,AAAA"), "image/png");
  assert.equal(fitOutputType("data:image/gif;base64,AAAA"), "image/png");
});
