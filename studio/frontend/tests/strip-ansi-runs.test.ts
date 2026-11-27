// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Oracle: main's per-character stripAnsi and split-based tailToolOutput.

import assert from "node:assert/strict";
import test from "node:test";

import { stripAnsi, tailToolOutput } from "../src/lib/strip-ansi.ts";

const ESC = 0x1b;
const BEL = 0x07;

const CAN = 0x18;
const SUB = 0x1a;
const C1_DCS = 0x90;
const C1_SOS = 0x98;
const C1_CSI = 0x9b;
const C1_ST = 0x9c;
const C1_OSC = 0x9d;
const C1_PM = 0x9e;
const C1_APC = 0x9f;

function isCsiFinal(code: number): boolean {
  return code >= 0x40 && code <= 0x7e;
}

function isCsiParameterOrIntermediate(code: number): boolean {
  return code >= 0x20 && code <= 0x3f;
}

function isScsFinal(code: number): boolean {
  return (code >= 0x30 && code <= 0x3f) || isCsiFinal(code);
}

function isEscStringControlIntroducer(code: number): boolean {
  return code === 0x50 || code === 0x58 || code === 0x5e || code === 0x5f;
}

function isC1StringControlIntroducer(code: number): boolean {
  return code === C1_DCS || code === C1_SOS || code === C1_PM || code === C1_APC;
}

function consumeStringControl(
  text: string,
  afterIntro: number,
  allowBelTerminator: boolean,
): number {
  let cursor = afterIntro;
  while (cursor < text.length) {
    const code = text.charCodeAt(cursor);

    if (code === CAN || code === SUB) {
      return cursor + 1;
    }
    if (allowBelTerminator && code === BEL) {
      return cursor + 1;
    }
    if (code === C1_ST) {
      return cursor + 1;
    }
    if (code === ESC) {
      if (cursor + 1 < text.length && text.charCodeAt(cursor + 1) === 0x5c) {
        return cursor + 2;
      }
      break;
    }
    cursor += 1;
  }
  return cursor;
}

function consumeCsi(text: string, afterIntro: number): number {
  let cursor = afterIntro;
  while (
    cursor < text.length &&
    isCsiParameterOrIntermediate(text.charCodeAt(cursor))
  ) {
    cursor += 1;
  }
  if (cursor < text.length && isCsiFinal(text.charCodeAt(cursor))) {
    cursor += 1;
  }
  return cursor;
}

function oracleStripAnsi(text: string): string {
  let out = "";
  let index = 0;
  while (index < text.length) {
    const code = text.charCodeAt(index);

    if (code === CAN || code === SUB) {
      index += 1;
      continue;
    }

    if (code === C1_CSI) {
      index = consumeCsi(text, index + 1);
      continue;
    }
    if (code === C1_OSC) {
      index = consumeStringControl(text, index + 1, true);
      continue;
    }
    if (isC1StringControlIntroducer(code)) {
      index = consumeStringControl(text, index + 1, false);
      continue;
    }
    if (code >= 0x80 && code <= 0x9f) {
      index += 1;
      continue;
    }
    if (code !== ESC) {
      out += text[index] ?? "";
      index += 1;
      continue;
    }
    if (index + 1 >= text.length) {
      break;
    }

    const next = text.charCodeAt(index + 1);
    if (next === 0x5b) {
      index = consumeCsi(text, index + 2);
      continue;
    }
    if (next === 0x5d) {
      index = consumeStringControl(text, index + 2, true);
      continue;
    }
    if (isEscStringControlIntroducer(next)) {
      index = consumeStringControl(text, index + 2, false);
      continue;
    }
    if (next >= 0x20 && next <= 0x2f) {
      index += 2;
      while (
        index < text.length &&
        text.charCodeAt(index) >= 0x20 &&
        text.charCodeAt(index) <= 0x2f
      ) {
        index += 1;
      }
      if (index < text.length && isScsFinal(text.charCodeAt(index))) {
        index += 1;
      }
      continue;
    }
    if (next >= 0x30 && next <= 0x7e && next !== 0x5b && next !== 0x5d) {
      index += 2;
      continue;
    }
    index += 1;
  }
  return out;
}


const TAIL_LINES = 2000;
const TAIL_CHARS = 200_000;

interface OracleTail {
  visible: string;
  hiddenLines: number;
  hiddenChars: number;
}

function oracleTail(text: string): OracleTail {
  let visible = text;
  let hiddenLines = 0;
  let hiddenChars = 0;
  const lines = visible.split("\n");
  if (lines.length > TAIL_LINES) {
    hiddenLines = lines.length - TAIL_LINES;
    visible = lines.slice(hiddenLines).join("\n");
  }
  if (visible.length > TAIL_CHARS) {
    hiddenChars = visible.length - TAIL_CHARS;
    visible = visible.slice(hiddenChars);
  }
  return { visible, hiddenLines, hiddenChars };
}


const ALPHABET = [
  "a", "b", " ", "\n", "\n", "\x1b", "[", "]", "0", ";", "m", "\x07", "\x18", "\x1a",
  "\x90", "\x98", "\x9b", "\x9c", "\x9d", "\x9e", "\x9f", "\x85", "\\", "P", "X", "^",
  "_", "(", "B", "#", "8", "\u00e9", "\u{1f600}", "\x7f", "/", "?", "@", "~",
];

function same(text: string) {
  assert.equal(stripAnsi(text), oracleStripAnsi(text));
  assert.deepEqual(tailToolOutput(text), oracleTail(text));
}

test("stripAnsi and tailToolOutput match the reference on random control mixes", () => {
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < 20000; i++) {
    let text = "";
    const length = Math.floor(rnd() * 40);
    for (let j = 0; j < length; j++) {
      text += ALPHABET[Math.floor(rnd() * ALPHABET.length)];
    }
    same(text);
  }
});

test("tailToolOutput keeps the same lines at the 2000-line boundary", () => {
  for (const lines of [1, 1999, 2000, 2001, 4000]) {
    for (const lead of ["", "\n", "\n\n"]) {
      for (const trail of ["", "\n", "x"]) {
        same(lead + Array.from({ length: lines }, (_, k) => `line ${k}`).join("\n") + trail);
        same(lead + "\n".repeat(lines) + trail);
      }
    }
  }
  same("y".repeat(250_000));
  same("z\n".repeat(150_000));
  same(`\x1b[31m${"q".repeat(300_000)}${"\x1b[0m\n".repeat(3000)}`);
  same("");
});
