// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const RUNTIME_PROVIDER = readSrc("features/chat/runtime-provider.tsx");
const SHARED_COMPOSER = readSrc("features/chat/shared-composer.tsx");

test("the chat composer toasts an image over the size limit before refusing it", () => {
  const sizeCheck = RUNTIME_PROVIDER.match(
    /class VisionImageAdapter[\s\S]*?if \(picked\.size > maxSize\) \{([\s\S]*?)\n {4}\}/,
  );
  assert.ok(sizeCheck, "VisionImageAdapter size check not found");
  assert.match(sizeCheck[1], /toast\.error\([\s\S]*?throw new Error\(/);
});

test("the compare composer reports an image over the size limit instead of skipping it", () => {
  const sizeCheck = SHARED_COMPOSER.match(
    /if \(file\.size > MAX_IMAGE_SIZE\) \{\s*(\w+) \?\?= file\.name;\s*continue;\s*\}/,
  );
  assert.ok(sizeCheck, "oversized compare images are dropped without a record");
  assert.match(
    SHARED_COMPOSER,
    new RegExp(`if \\(${sizeCheck[1]}\\) \\{\\s*toast\\.error\\(`),
  );
});
