// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getChatImageSizeError } from "../src/features/chat/image-normalize.ts";
import { readSrc } from "./helpers/kit.ts";

const RUNTIME_PROVIDER = readSrc("features/chat/runtime-provider.tsx");
const SHARED_COMPOSER = readSrc("features/chat/shared-composer.tsx");
const MB = 1024 * 1024;

test("an image over 20 MB is named in the refusal", () => {
  assert.equal(getChatImageSizeError({ name: "a.png", size: 20 * MB }), null);
  assert.equal(
    getChatImageSizeError({ name: "big.png", size: 20 * MB + 1 }),
    "big.png is over the 20 MB image limit.",
  );
});

test("the chat composer toasts an oversized image before refusing it", () => {
  assert.match(
    RUNTIME_PROVIDER,
    /class VisionImageAdapter[\s\S]*?const sizeError = getChatImageSizeError\(picked\);\s*if \(sizeError\) \{\s*toast\.error\(sizeError\);\s*throw new Error\(sizeError\);/,
  );
});

test("the compare composer toasts each oversized image, including a pasted one", () => {
  assert.match(
    SHARED_COMPOSER,
    /const sizeError = getChatImageSizeError\(file\);\s*if \(sizeError\) \{\s*toast\.error\(sizeError\);\s*continue;/,
  );
  assert.match(
    SHARED_COMPOSER,
    /const supported = files\.some\(\s*\(file\) => isAudioAttachmentFile\(file\) \|\| isChatImageFile\(file\),\s*\);/,
  );
});
