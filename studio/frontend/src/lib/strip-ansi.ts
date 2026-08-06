// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// CSI / Fe / OSC escape sequences as emitted by colourised CLI tools (ls,
// grep, npm, cargo, pytest, …) and terminal hyperlinks. Built via fromCharCode
// so the source stays free of a literal ESC that some editors / log scrapers
// trip over.
const ESC = String.fromCharCode(27);
const BEL = String.fromCharCode(7);
const ANSI_ESCAPE_PATTERN = new RegExp(
  [
    `${ESC}(?:[@-Z\\-_]|\\[[0-?]*[ -/]*[@-~])`,
    `${ESC}\\](?:[\\s\\S]*?)(?:${BEL}|${ESC}\\\\)`,
  ].join("|"),
  "g",
);

/** Strip SGR / CSI / OSC sequences so tool output is readable in a plain <pre>. */
export function stripAnsi(text: string): string {
  return text.replace(ANSI_ESCAPE_PATTERN, "");
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
