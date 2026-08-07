// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { stripAnsi, stringifyToolResult } from "../src/lib/strip-ansi.ts";

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
  const garbage = `${ESC}]`.repeat(500);
  assert.equal(stripAnsi(garbage), "");
});

test("hides partial OSC payloads while a hyperlink is still streaming", () => {
  const partial = `${ESC}]8;;file:///tmp/demo`;
  assert.equal(stripAnsi(partial), "");
});

test("strips DCS payloads terminated by ST", () => {
  const dcs = `${ESC}Pfake-sixel${ESC}\\ok`;
  assert.equal(stripAnsi(dcs), "ok");
});

test("strips DEC save, restore, and full reset controls", () => {
  assert.equal(stripAnsi(`${ESC}7text${ESC}8`), "text");
  assert.equal(stripAnsi(`${ESC}chello`), "hello");
});

test("an aborted CSI does not swallow the sequence that follows it", () => {
  assert.equal(stripAnsi(`${ESC}[${ESC}[32mhi`), "hi");
  assert.equal(stripAnsi(`${ESC}[32\nplain`), "\nplain");
});

test("stringifyToolResult cleans nested tool object fields before JSON display", () => {
  const rendered = stringifyToolResult({
    stdout: `${ESC}[32mfile.txt${ESC}[0m`,
    nested: [{ line: `${ESC}[31merror${ESC}[0m` }],
  });
  const parsed = JSON.parse(rendered);
  assert.equal(parsed.stdout, "file.txt");
  assert.equal(parsed.nested[0]?.line, "error");
});

test("stringifyToolResult strips ANSI out of object keys too", () => {
  const rendered = stringifyToolResult({ [`${ESC}[32mstdout`]: "ok" });
  assert.equal(rendered.includes("\\u001b"), false);
  assert.match(rendered, /"stdout": "ok"/);
});

test("stringifyToolResult keeps toJSON serialization (Date stays an ISO string)", () => {
  const rendered = stringifyToolResult({ at: new Date("2020-01-02T03:04:05Z") });
  assert.match(rendered, /"at": "2020-01-02T03:04:05\.000Z"/);
});

test("stringifyToolResult avoids \\u001b litter for structured tool output", () => {
  const rendered = stringifyToolResult({
    stdout: `${ESC}[32mfile.txt${ESC}[0m`,
  });
  assert.equal(rendered.includes("\\u001b"), false);
  assert.equal(rendered.includes("[32m"), false);
  assert.match(rendered, /"stdout": "file\.txt"/);
});
