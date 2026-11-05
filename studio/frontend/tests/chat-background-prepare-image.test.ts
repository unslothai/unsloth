// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { prepareBackgroundImage } from "../src/features/chat-background/image.ts";
import { imageHeaderSize } from "../src/features/chat-background/model.ts";

// Rejected before any decoding, so no DOM is needed.
test("uploads that fail validation reject with 'file'", async () => {
  await assert.rejects(prepareBackgroundImage(new Blob([new Uint8Array(4)], { type: "image/gif" })), { message: "file" });
  await assert.rejects(prepareBackgroundImage(new Blob([], { type: "image/png" })), { message: "file" });
});

const png = (w: number, h: number) => {
  const bytes = new Uint8Array(33);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52]);
  new DataView(bytes.buffer).setUint32(16, w);
  new DataView(bytes.buffer).setUint32(20, h);
  return bytes;
};
// SOI, an APP0 segment to skip, then SOF2 (progressive) with height 300 and width 400.
const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc2, 0, 11, 8, 0x01, 0x2c, 0x01, 0x90, 3, 0, 0, 0]);
const webpX = new Uint8Array(30);
webpX.set([...new TextEncoder().encode("RIFF"), 0, 0, 0, 0, ...new TextEncoder().encode("WEBPVP8X")]);
webpX.set([0x3f, 0x1f, 0, 0x1f, 0x0f, 0], 24); // 8000 x 3872, stored minus one

test("image sizes are read from PNG, JPEG and WebP headers", () => {
  assert.deepEqual(imageHeaderSize(png(1920, 1080)), [1920, 1080]);
  assert.deepEqual(imageHeaderSize(JPEG), [400, 300]);
  assert.deepEqual(imageHeaderSize(webpX), [8000, 3872]);
  assert.equal(imageHeaderSize(new Uint8Array([1, 2, 3])), null);
});

test("an oversized or unreadable image is refused before it is decoded", async () => {
  const huge = new Blob([png(20_000, 20_000)], { type: "image/png" });
  await assert.rejects(prepareBackgroundImage(huge), { message: "dimensions" });
  await assert.rejects(prepareBackgroundImage(new Blob([new Uint8Array(40)], { type: "image/png" })), { message: "decode" });
});
