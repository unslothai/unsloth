// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  stripAnsi,
  stringifyToolResult,
  tailToolOutput,
} from "../src/lib/strip-ansi.ts";

const ESC = String.fromCharCode(27);
const BEL = String.fromCharCode(7);

const CAN = String.fromCharCode(0x18);
const SUB = String.fromCharCode(0x1a);

const C1_DCS = String.fromCharCode(0x90);
const C1_CSI = String.fromCharCode(0x9b);
const C1_ST = String.fromCharCode(0x9c);
const C1_OSC = String.fromCharCode(0x9d);

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


test("tailing uses cleaned glyph counts and enforces both boundaries", () => {
  const visible = "x".repeat(200_000);
  const coloured = `${ESC}[32m${visible}${ESC}[0m`;
  assert.equal(tailToolOutput(coloured).hiddenChars > 0, true);
  assert.deepEqual(tailToolOutput(stripAnsi(coloured)), {
    visible,
    hiddenLines: 0,
    hiddenChars: 0,
  });

  const lines = Array.from({ length: 2001 }, (_, index) => String(index)).join("\n");
  const tail = tailToolOutput(lines);
  assert.equal(tail.hiddenLines, 1);
  assert.equal(tail.visible.startsWith("1\n"), true);
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


test("streaming controls stay hidden until complete and reveal later text", () => {
  let accumulated = `${ESC}]8;;file:///tmp/demo`;
  assert.equal(stripAnsi(accumulated), "");
  accumulated += `${BEL}link${ESC}]8;;${BEL}`;
  assert.equal(stripAnsi(accumulated), "link");

  accumulated = `${ESC}[31`;
  assert.equal(stripAnsi(accumulated), "");
  accumulated += `mred${ESC}[0m`;
  assert.equal(stripAnsi(accumulated), "red");
});

test("CAN and SUB cancel control strings without hiding following text", () => {
  assert.equal(stripAnsi(`${ESC}]0;title${CAN}visible`), "visible");
  assert.equal(stripAnsi(`${C1_OSC}0;title${SUB}visible`), "visible");
  assert.equal(stripAnsi(`${ESC}[31${CAN}visible`), "visible");
});


test("malformed SCS leaves its violating byte and following text visible", () => {
  assert.equal(stripAnsi(`${ESC}(\nplain`), "\nplain");
});

test("strips DCS payloads terminated by ST", () => {
  const dcs = `${ESC}Pfake-sixel${ESC}\\ok`;
  assert.equal(stripAnsi(dcs), "ok");
});


test("strips SOS payloads and 8-bit C1 control forms", () => {
  assert.equal(stripAnsi(`${ESC}Xprivate${ESC}\\ok`), "ok");
  assert.equal(stripAnsi(`${C1_CSI}32mgreen${C1_CSI}0m`), "green");
  assert.equal(stripAnsi(`${C1_OSC}0;title${C1_ST}ok`), "ok");
  assert.equal(stripAnsi(`${C1_DCS}payload${C1_ST}ok`), "ok");
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


test("stringifyToolResult preserves fields whose cleaned keys collide", () => {
  const rendered = stringifyToolResult({
    [`${ESC}[31mkey`]: "ansi",
    key: "plain",
    [`${ESC}[32mkey`]: "second ansi",
  });
  const parsed = JSON.parse(rendered);
  assert.deepEqual(parsed, {
    "key [ansi]": "ansi",
    key: "plain",
    "key [ansi 2]": "second ansi",
  });
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
