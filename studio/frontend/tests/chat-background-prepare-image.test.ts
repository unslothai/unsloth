// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  dataUrlToBlob,
  prepareBackgroundImage,
} from "../src/features/chat-background/image.ts";

test("a base64 data URL decodes back to its bytes and mime type", async () => {
  const bytes = new Uint8Array([0, 1, 2, 127, 128, 253, 254, 255]);
  const dataUrl = `data:image/png;base64,${Buffer.from(bytes).toString("base64")}`;
  const blob = await dataUrlToBlob(dataUrl);
  assert.equal(blob.type, "image/png");
  assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), bytes);
});

test("malformed data URLs are rejected as decode failures", async () => {
  for (const dataUrl of [
    "https://example.com/a.png",
    "data:image/png,AAAA", // not base64
    "data:image/png;base64,@@@@", // characters outside the base64 alphabet
    "data:;base64,AAAA", // no mime type
  ]) {
    await assert.rejects(dataUrlToBlob(dataUrl), { message: "decode" }, dataUrl);
  }
});

test("uploads that fail validation are rejected before any decoding", async () => {
  await assert.rejects(
    prepareBackgroundImage(new Blob([new Uint8Array(4)], { type: "image/gif" })),
    { message: "file" },
  );
  await assert.rejects(
    prepareBackgroundImage(new Blob([], { type: "image/png" })),
    { message: "file" },
  );
});
