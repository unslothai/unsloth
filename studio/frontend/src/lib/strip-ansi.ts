// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

/** CSI parameter (0x30-0x3f) and intermediate (0x20-0x2f) bytes. */
function isCsiParameterOrIntermediate(code: number): boolean {
  return code >= 0x20 && code <= 0x3f;
}

function isScsFinal(code: number): boolean {
  return (code >= 0x30 && code <= 0x3f) || isCsiFinal(code);
}

function isEscStringControlIntroducer(code: number): boolean {
  // DCS (P), SOS (X), PM (^), APC (_)
  return code === 0x50 || code === 0x58 || code === 0x5e || code === 0x5f;
}

function isC1StringControlIntroducer(code: number): boolean {
  return code === C1_DCS || code === C1_SOS || code === C1_PM || code === C1_APC;
}

/** Advance past an OSC / DCS / SOS / PM / APC payload. */
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
      // Preserve a new escape introducer so the outer scanner can consume it.
      break;
    }
    cursor += 1;
  }
  // Unterminated string controls are intentionally hidden while streaming.
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
  // An aborted CSI (ESC/C1/newline before the final byte) leaves the cursor on
  // the offending byte so a following sequence is not swallowed with it.
  if (cursor < text.length && isCsiFinal(text.charCodeAt(cursor))) {
    cursor += 1;
  }
  return cursor;
}

/** Strip ECMA-48 CSI / SCS / OSC / string-control sequences in one linear pass. */
export function stripAnsi(text: string): string {
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
    // Other C1 bytes are single control functions and have no printable form.
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


/** Tail-line cap so a huge output never mounts a megabyte <pre> block. */
const TAIL_LINES = 2000;
/** Char backstop for pathological single-line outputs. */
const TAIL_CHARS = 200_000;

export interface ToolOutputTail {
  visible: string;
  hiddenLines: number;
  hiddenChars: number;
}

export function tailToolOutput(text: string): ToolOutputTail {
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

  const entries = Object.entries(value).map(([key, entry]) => ({
    key,
    cleanedKey: stripAnsi(key),
    entry,
  }));
  if (entries.every(({ key, cleanedKey }) => key === cleanedKey)) {
    return value;
  }

  // A stripped key can collide with a real key (or another stripped key). Keep
  // every field visible rather than silently dropping one through fromEntries.
  const plainKeys = new Set(
    entries
      .filter(({ key, cleanedKey }) => key === cleanedKey)
      .map(({ key }) => key),
  );
  const usedKeys = new Set<string>();
  return Object.fromEntries(
    entries.map(({ key, cleanedKey, entry }) => {
      if (key === cleanedKey) {
        usedKeys.add(key);
        return [key, entry];
      }

      let displayKey = cleanedKey;
      let suffix = 1;
      while (plainKeys.has(displayKey) || usedKeys.has(displayKey)) {
        const label = suffix === 1 ? "ansi" : `ansi ${suffix}`;
        displayKey = `${cleanedKey} [${label}]`;
        suffix += 1;
      }
      usedKeys.add(displayKey);
      return [displayKey, entry];
    }),
  );
}

/** Plain-text tool result for a <pre>: strings directly, objects as stripped JSON. */
export function stringifyToolResult(result: unknown): string {
  return typeof result === "string"
    ? stripAnsi(result)
    : JSON.stringify(result, stripAnsiReplacer, 2);
}
