// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { decodeDataUri, isDataUri } from "../src/lib/data-uri.ts";

const INVALID_DATA_URI_RE = /Invalid data URI/;
const DEFAULT_MIME = "text/plain;charset=US-ASCII";

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

test("treats base64 as the marker only when it ends the metadata", () => {
  // A mid-metadata `base64` segment is an ordinary parameter.
  assert.deepEqual(
    Array.from(
      decodeDataUri("data:text/plain;base64;charset=utf-8,SGVsbG8=").bytes,
    ),
    [83, 71, 86, 115, 98, 71, 56, 61],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:base64,SGVsbG8=").bytes),
    [83, 71, 86, 115, 98, 71, 56, 61],
  );
  assert.deepEqual(
    Array.from(
      decodeDataUri("data:text/plain;charset=utf-8;base64,SGVsbG8=").bytes,
    ),
    [72, 101, 108, 108, 111],
  );
});

test("ignores a URL fragment", () => {
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain,abc#frag").bytes),
    [97, 98, 99],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain;base64,SGVsbG8=#frag").bytes),
    [72, 101, 108, 108, 111],
  );
  // An escaped hash is payload, not a fragment.
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain,abc%23hash").bytes),
    [97, 98, 99, 35, 104, 97, 115, 104],
  );
});

test("falls back to the default media type when there is no slash", () => {
  assert.equal(decodeDataUri("data:base64,SGVsbG8=").mimeType, DEFAULT_MIME);
  assert.equal(decodeDataUri("data:;base64,AAA=").mimeType, DEFAULT_MIME);
  assert.equal(
    decodeDataUri("data:image/png;base64,QUJD").mimeType,
    "image/png",
  );
});

test("decodes a large base64 payload without stalling", () => {
  // The 20 MiB attachment cap must not take seconds of blocked UI.
  const payload = btoa("x".repeat(3 * 1024 * 1024));
  const started = Date.now();
  const decoded = decodeDataUri(`data:image/png;base64,${payload}`);
  assert.equal(decoded.bytes.length, 3 * 1024 * 1024);
  assert.ok(
    Date.now() - started < 2000,
    `decoding took ${Date.now() - started}ms`,
  );
});

test("treats the data scheme case-insensitively", () => {
  // URL schemes are case-insensitive and all three engines render DATA:.
  assert.ok(isDataUri("DATA:image/png;base64,QUJD"));
  assert.ok(isDataUri("Data:image/png;base64,QUJD"));
  assert.ok(isDataUri("data:image/png;base64,QUJD"));
  assert.ok(!isDataUri("https://example.com/a.png"));
  assert.deepEqual(
    Array.from(decodeDataUri("DATA:text/plain;base64,QUJD").bytes),
    [65, 66, 67],
  );
});

test("decodes an escape-heavy payload without stalling", () => {
  // Encoded SVG text alternates literals and escapes, which used to allocate
  // a separate array per run.
  const source = "a%20".repeat(400000);
  const started = Date.now();
  const decoded = decodeDataUri(`data:image/svg+xml,${source}`);
  assert.equal(decoded.bytes.length, 800000);
  assert.equal(decoded.bytes[0], 97);
  assert.equal(decoded.bytes[1], 32);
  assert.ok(
    Date.now() - started < 2000,
    `decoding took ${Date.now() - started}ms`,
  );
});

test("removes URL tabs and newlines the way the URL parser does", () => {
  // Firefox and WebKit strip these before parsing, per the URL standard.
  // Chromium keeps them for a data: URL passed to fetch, so this follows the
  // standard and the majority.
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain;base64\n,SGVsbG8=").bytes),
    [72, 101, 108, 108, 111],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain;base64\t,SGVsbG8=").bytes),
    [72, 101, 108, 108, 111],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain;bas\ne64,SGVsbG8=").bytes),
    [72, 101, 108, 108, 111],
  );
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain,ab\ncd").bytes),
    [97, 98, 99, 100],
  );
  // An escaped newline is payload, not URL whitespace.
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain,ab%0Acd").bytes),
    [97, 98, 10, 99, 100],
  );
});

test("trims leading and trailing C0 controls and spaces", () => {
  // All three engines render ` data:image/png;...` and decode these.
  assert.ok(isDataUri(" data:text/plain,abc"));
  assert.ok(isDataUri("\u0000data:text/plain,abc"));
  assert.ok(isDataUri("  DATA:text/plain,abc"));
  for (const uri of [
    " data:text/plain,abc",
    "  data:text/plain,abc",
    "\u0000data:text/plain,abc",
    "\u001fdata:text/plain,abc",
    "data:text/plain,abc ",
  ]) {
    assert.deepEqual(Array.from(decodeDataUri(uri).bytes), [97, 98, 99], uri);
  }
  // A space inside the payload is content, not URL whitespace.
  assert.deepEqual(
    Array.from(decodeDataUri("data:text/plain,a bc").bytes),
    [97, 32, 98, 99],
  );
});

test("detects the scheme past any number of leading controls", () => {
  // All three engines decode these; a fixed-size prefix window could not.
  const lead = [" ".repeat(30), "\u0000".repeat(40), "  \u0000 \t"];
  for (const prefix of lead) {
    assert.ok(
      isDataUri(`${prefix}data:text/plain,abc`),
      JSON.stringify(prefix),
    );
    assert.deepEqual(
      Array.from(decodeDataUri(`${prefix}data:text/plain,abc`).bytes),
      [97, 98, 99],
    );
  }
  // Tabs and newlines are removed inside the scheme too.
  assert.ok(isDataUri("da\nta:text/plain,abc"));
  assert.ok(isDataUri("da\tta:text/plain,abc"));
  // A space is not removed, so this is not a data URL in any engine.
  assert.ok(!isDataUri("da ta:text/plain,abc"));
  assert.ok(!isDataUri("https://example.com/a.png"));
  assert.ok(!isDataUri("dat"));
  assert.ok(!isDataUri(""));
});
