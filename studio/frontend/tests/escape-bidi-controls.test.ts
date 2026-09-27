// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { escapeBidiControls } from "../src/lib/escape-bidi-controls.ts";

test("a right-to-left override is shown instead of obeyed", () => {
  const shown = escapeBidiControls("https://example.com/safe/\u202egpj.exe");
  assert.equal(shown, "https://example.com/safe/\\u202egpj.exe");
});

test("every bidi control is escaped, including the isolates and the marks", () => {
  for (const control of [
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\u200e",
    "\u200f",
    "\u061c",
  ]) {
    const escaped = escapeBidiControls(`a${control}b`);
    assert.ok(
      !escaped.includes(control),
      `U+${(control.codePointAt(0) ?? 0).toString(16)} survived escaping`,
    );
    assert.match(escaped, /^a\\u[0-9a-f]{4}b$/);
  }
});

test("ordinary text and non-bidi unicode are left alone", () => {
  for (const value of [
    "https://example.com/path/to/page.html?utm_source=lambda&q=alpha%20beta#section-2",
    "",
    "example.com",
    "https://example.com/caf\u00e9/\u4e2d\u6587?q=1",
    "a\u200db",
  ]) {
    assert.equal(escapeBidiControls(value), value);
  }
});
