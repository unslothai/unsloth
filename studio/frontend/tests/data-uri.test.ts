// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { decodeDataUri } from "../src/lib/data-uri.ts";

const INVALID_DATA_URI_RE = /Invalid data URI/;

test("decodes base64 data URIs with their media type", () => {
  const decoded = decodeDataUri("data:audio/wav;base64,AAH6/w==");

  assert.equal(decoded.mimeType, "audio/wav");
  assert.deepEqual(Array.from(decoded.bytes), [0, 1, 250, 255]);
});

test("preserves commas in percent-encoded data URI payloads", () => {
  const decoded = decodeDataUri("data:text/plain,hello,world%20again");

  assert.equal(decoded.mimeType, "text/plain");
  assert.equal(new TextDecoder().decode(decoded.bytes), "hello,world again");
});

test("uses the RFC default media type when it is omitted", () => {
  const decoded = decodeDataUri("data:,plain%20text");

  assert.equal(decoded.mimeType, "text/plain;charset=US-ASCII");
  assert.equal(new TextDecoder().decode(decoded.bytes), "plain text");
});

test("rejects data URIs without a payload separator", () => {
  assert.throws(
    () => decodeDataUri("data:image/png;base64"),
    INVALID_DATA_URI_RE,
  );
});
