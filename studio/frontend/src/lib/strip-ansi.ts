// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ESC = 27;
const BEL = 7;
const ESCAPE_CHAR = String.fromCharCode(ESC);

function isCsiFinal(code: number): boolean {
  return code >= 0x40 && code <= 0x7e;
}

/** CSI parameter (0x30-0x3f) and intermediate (0x20-0x2f) bytes. */
function isCsiParameter(code: number): boolean {
  return code >= 0x20 && code <= 0x3f;
}

function isScsFinal(code: number): boolean {
  return (code >= 0x30 && code <= 0x3f) || isCsiFinal(code);
}

function isStringControlIntroducer(code: number): boolean {
  // DCS (P), PM (^), APC (_)
  return code === 0x50 || code === 0x5e || code === 0x5f;
}

/** Advance past a DCS / PM / APC payload, stopping at ST (ESC \\) or the next ESC. */
function consumeStringControl(text: string, afterIntro: number): number {
  let j = afterIntro;
  while (j < text.length) {
    if (text.charCodeAt(j) === ESC) {
      if (j + 1 < text.length && text.charCodeAt(j + 1) === 0x5c) {
        return j + 2;
      }
      break;
    }
    j += 1;
  }
  return j;
}

/** Advance past an OSC payload, stopping at BEL, ST (ESC \\), or the next ESC. */
function consumeOsc(text: string, afterIntro: number): number {
  let j = afterIntro;
  while (j < text.length) {
    const osc = text.charCodeAt(j);
    if (osc === BEL) {
      return j + 1;
    }
    if (osc === ESC) {
      if (j + 1 < text.length && text.charCodeAt(j + 1) === 0x5c) {
        return j + 2;
      }
      break;
    }
    j += 1;
  }
  return j;
}

/** Strip SGR / CSI / SCS / OSC / string-control sequences in one linear pass. */
export function stripAnsi(text: string): string {
  let out = "";
  let index = 0;
  while (index < text.length) {
    const code = text.charCodeAt(index);
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
      index += 2;
      while (index < text.length && isCsiParameter(text.charCodeAt(index))) {
        index += 1;
      }
      // An aborted CSI (ESC or a newline before the final byte) leaves index on
      // the offending byte so the next sequence is not swallowed with it.
      if (index < text.length && isCsiFinal(text.charCodeAt(index))) {
        index += 1;
      }
      continue;
    }
    if (next === 0x5d) {
      index = consumeOsc(text, index + 2);
      continue;
    }
    if (isStringControlIntroducer(next)) {
      index = consumeStringControl(text, index + 2);
      continue;
    }
    if (next >= 0x20 && next <= 0x2f) {
      index += 2;
      while (index < text.length && !isScsFinal(text.charCodeAt(index))) {
        index += 1;
      }
      if (index < text.length) {
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

/**
 * JSON.stringify replacer stripping ANSI from string values and object keys.
 * A replacer sees values after toJSON, so Date and friends keep serializing
 * normally; pre-walking the object would have flattened them to {}.
 */
function stripAnsiReplacer(_key: string, value: unknown): unknown {
  if (typeof value === "string") {
    return stripAnsi(value);
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }
  const entries = Object.entries(value);
  // Rebuild only for escaped keys; anything else keeps its native serialization.
  if (!entries.some(([key]) => key.includes(ESCAPE_CHAR))) {
    return value;
  }
  return Object.fromEntries(
    entries.map(([key, entry]) => [stripAnsi(key), entry]),
  );
}

/** Plain-text tool result for a <pre>: strings directly, objects as stripped JSON. */
export function stringifyToolResult(result: unknown): string {
  return typeof result === "string"
    ? stripAnsi(result)
    : JSON.stringify(result, stripAnsiReplacer, 2);
}
