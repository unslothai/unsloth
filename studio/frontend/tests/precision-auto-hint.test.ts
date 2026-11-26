// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Auto keeps BF16 for families where INT8 measured no faster or visibly different (Klein 4B, Lumina 2, HiDream), so
// the Precision hint and option must not promise "the fastest precision" (NVIDIA feedback p.5).

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

for (const path of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  const src = readSrc(path);
  const select = src.slice(src.indexOf('label="Precision"'));
  const block = select.slice(0, select.indexOf("/>"));

  test(`${path}: the Auto hint says Auto may keep BF16 and does not promise the fastest precision`, () => {
    assert.doesNotMatch(block, /Auto picks the fastest precision/);
    assert.doesNotMatch(block, /fastest for GPU/);
    assert.match(block, /keeps (plain )?(BF16|bf16) for models where quantising was measured as no faster/);
    assert.match(block, /Auto is not always the fastest choice/);
  });
}
