// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  stripAnsi,
  stripAnsiDeep,
  stringifyToolResult,
} from "../src/lib/strip-ansi.ts";

const ESC = String.fromCharCode(27);
const BEL = String.fromCharCode(7);

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

test("strips OSC terminal hyperlinks terminated by BEL", () => {
  const linked = `${ESC}]8;;file:///tmp/demo${BEL}file.txt${ESC}]8;;${BEL}`;
  assert.equal(stripAnsi(linked), "file.txt");
});

test("strips OSC sequences terminated by ST (ESC backslash)", () => {
  const titled = `${ESC}]0;npm install${ESC}\\`;
  assert.equal(stripAnsi(titled), "");
});

test("strips SCS charset resets emitted by terminfo sgr0", () => {
  const reset = `${ESC}(B${ESC}[mok`;
  assert.equal(stripAnsi(reset), "ok");
});

test("strips many unterminated OSC introducers without quadratic work", () => {
  const garbage = `${ESC}]`.repeat(500) + "ok";
  const cleaned = stripAnsi(garbage);
  assert.equal(cleaned, "ok");
});

test("stripAnsiDeep cleans nested tool object fields before JSON display", () => {
  const payload = {
    stdout: `${ESC}[32mfile.txt${ESC}[0m`,
    nested: [{ line: `${ESC}[31merror${ESC}[0m` }],
  };
  const cleaned = stripAnsiDeep(payload);
  assert.equal(cleaned.stdout, "file.txt");
  assert.equal(cleaned.nested[0]?.line, "error");
});

test("stringifyToolResult avoids \\u001b litter for structured tool output", () => {
  const rendered = stringifyToolResult({
    stdout: `${ESC}[32mfile.txt${ESC}[0m`,
  });
  assert.equal(rendered.includes("\\u001b"), false);
  assert.equal(rendered.includes("[32m"), false);
  assert.match(rendered, /"stdout": "file\.txt"/);
});
