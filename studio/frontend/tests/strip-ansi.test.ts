// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { stripAnsi } from "../src/lib/strip-ansi.ts";

const ESC = String.fromCharCode(27);

test("strips SGR colour sequences from ls --color / grep --color output", () => {
  const coloured = `${ESC}[32mfile.txt${ESC}[0m\n${ESC}[01;31mmatch${ESC}[0m`;
  assert.equal(stripAnsi(coloured), "file.txt\nmatch");
});

test("strips cursor / erase CSI used by npm and cargo progress", () => {
  const progress = `Downloading${ESC}[2K${ESC}[1GDone`;
  assert.equal(stripAnsi(progress), "DownloadingDone");
});

test("leaves plain text and newlines alone", () => {
  assert.equal(stripAnsi("hello\nworld"), "hello\nworld");
  assert.equal(stripAnsi(""), "");
});

test("strips pytest-style green pass markers", () => {
  const line = `${ESC}[32mPASSED${ESC}[0m tests/strip-ansi.test.ts`;
  assert.equal(stripAnsi(line), "PASSED tests/strip-ansi.test.ts");
});

test("shrinks colourised output so tailing counts visible glyphs not escapes", () => {
  const coloured = `${"ok\n".repeat(10)}${ESC}[32m${"x".repeat(500)}${ESC}[0m`;
  const cleaned = stripAnsi(coloured);
  assert.equal(cleaned.includes(ESC), false);
  assert.ok(cleaned.length < coloured.length);
});
