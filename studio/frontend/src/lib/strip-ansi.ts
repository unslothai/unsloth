// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ESC = 27;
const BEL = 7;

function isCsiFinal(code: number): boolean {
  return code >= 0x40 && code <= 0x7e;
}

function isScsFinal(code: number): boolean {
  return (code >= 0x30 && code <= 0x3f) || isCsiFinal(code);
}

/** Strip SGR / CSI / SCS / OSC sequences in one linear pass. */
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
      while (index < text.length && !isCsiFinal(text.charCodeAt(index))) {
        index += 1;
      }
      if (index < text.length) {
        index += 1;
      }
      continue;
    }
    if (next === 0x5d) {
      let j = index + 2;
      let terminated = false;
      while (j < text.length) {
        const osc = text.charCodeAt(j);
        if (osc === BEL) {
          j += 1;
          terminated = true;
          break;
        }
        if (osc === ESC) {
          if (j + 1 < text.length && text.charCodeAt(j + 1) === 0x5c) {
            j += 2;
            terminated = true;
          }
          break;
        }
        j += 1;
      }
      index = terminated ? j : index + 2;
      continue;
    }
    if (next >= 0x40 && next <= 0x5f) {
      index += 2;
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
    index += 1;
  }
  return out;
}

/** Recursively strip ANSI from string leaves (objects / arrays from tool JSON). */
export function stripAnsiDeep<T>(value: T): T {
  if (typeof value === "string") {
    return stripAnsi(value) as T;
  }
  if (Array.isArray(value)) {
    return value.map((item) => stripAnsiDeep(item)) as T;
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, stripAnsiDeep(entry)]),
    ) as T;
  }
  return value;
}

/** Plain-text tool result for a <pre>: strings directly, objects after deep strip. */
export function stringifyToolResult(result: unknown): string {
  return typeof result === "string"
    ? stripAnsi(result)
    : JSON.stringify(stripAnsiDeep(result), null, 2);
}
