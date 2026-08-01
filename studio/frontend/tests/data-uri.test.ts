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

// The expectations below were taken from Chromium, Firefox and WebKit, which
// all agree: percent-decoding a data URI is byte-oriented, not UTF-8 text.

test("decodes percent escapes that are not valid UTF-8", () => {
  // decodeURIComponent() throws URIError on these; a browser returns the octets.
  assert.deepEqual(
    Array.from(decodeDataUri("data:audio/wav,%FF%00%80").bytes),
    [255, 0, 128],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:application/octet-stream,%FF").bytes),
    [255],
  );
});

test("leaves malformed percent escapes as literal characters", () => {
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain,%G0").bytes),
    [37, 71, 48],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain,abc%").bytes),
    [97, 98, 99, 37],
  );
});

test("does not treat a base64x parameter as base64", () => {
  // The old `/;base64/i` matched inside `;base64x`; the anchored form must not.
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain;base64x,QUJD").bytes),
    [81, 85, 74, 68],
  );
});

test("percent-decodes a base64 payload before decoding it", () => {
  // atob() would throw InvalidCharacterError on the escapes.
  assert.deepEqual(
    Array.from(decodeDataUri("data:audio/wav;base64,SGVsbG8%3D").bytes),
    [72, 101, 108, 108, 111],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:audio/wav;base64,AAH6%2Fw%3D%3D").bytes),
    [0, 1, 250, 255],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain;base64,QUJ%44").bytes),
    [65, 66, 67],
  );
});
