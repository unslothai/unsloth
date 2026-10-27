// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function isSingleJsonObject(text: string): boolean {
  try {
    const value = JSON.parse(text);
    return typeof value === "object" && value !== null && !Array.isArray(value);
  } catch {
    return false;
  }
}

/** Splits slot text into top-level JSON objects; a second `{` means a reused parallel slot. */
export function splitTopLevelJsonObjects(text: string): {
  complete: string[];
  tail: string;
} {
  const unsplit = { complete: [] as string[], tail: text };
  const complete: string[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (depth === 0) {
      if (ch === "{") {
        depth = 1;
        start = i;
        continue;
      }
      if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") continue;
      return unsplit;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        const segment = text.slice(start, i + 1);
        try {
          JSON.parse(segment);
        } catch {
          // Balanced but invalid: cutting here would invent a call.
          return unsplit;
        }
        complete.push(segment);
        start = -1;
      }
    }
  }

  return {
    complete,
    tail: start === -1 ? "" : text.slice(start),
  };
}

/** Incremental split for an append-only string; rescanning per fragment is O(N^2). */
export function createBoundaryScan(): {
  feed: (text: string) => { complete: string[]; tail: string };
} {
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  let scanned = 0;
  let unsplittable = false;
  const complete: string[] = [];

  return {
    feed(text: string) {
      if (unsplittable) return { complete: [], tail: text };
      for (let i = scanned; i < text.length; i += 1) {
        const ch = text[i];
        if (inString) {
          if (escaped) escaped = false;
          else if (ch === "\\") escaped = true;
          else if (ch === '"') inString = false;
          continue;
        }
        if (depth === 0) {
          if (ch === "{") {
            depth = 1;
            start = i;
            continue;
          }
          if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") continue;
          unsplittable = true;
          return { complete: [], tail: text };
        }
        if (ch === '"') {
          inString = true;
          continue;
        }
        if (ch === "{") depth += 1;
        else if (ch === "}") {
          depth -= 1;
          if (depth === 0) {
            const segment = text.slice(start, i + 1);
            try {
              JSON.parse(segment);
            } catch {
              unsplittable = true;
              return { complete: [], tail: text };
            }
            complete.push(segment);
            start = -1;
          }
        }
      }
      scanned = text.length;
      return {
        complete: [...complete],
        tail: start === -1 ? "" : text.slice(start),
      };
    },
  };
}

/** Prefers streamed argsText for byte-exact replay; unparsable text falls back since strict
 *  templates reject the whole request. `{ _raw }` marks unparsable text and replays as `{}`. */
export function toolCallReplayArguments(
  argsText: string | undefined,
  args: unknown,
): string {
  if (
    typeof argsText === "string" &&
    argsText.length > 0 &&
    isSingleJsonObject(argsText)
  ) {
    return argsText;
  }
  const serialized = JSON.stringify(args ?? {});
  if (serialized === undefined || !isSingleJsonObject(serialized)) {
    return "{}";
  }
  const parsed = JSON.parse(serialized) as Record<string, unknown>;
  const keys = Object.keys(parsed);
  // Treat `{ _raw }` as the adapter marker only when its value equals the surviving argsText.
  if (
    keys.length === 1 &&
    keys[0] === "_raw" &&
    typeof parsed._raw === "string" &&
    // Non-empty: the adapter never writes the marker for empty text, so `{ _raw: "" }` is real.
    parsed._raw.length > 0 &&
    parsed._raw === argsText
  ) {
    return "{}";
  }
  return serialized;
}

/**
 * Prefers the backend's own encoding: re-encoding after `JSON.parse` rounds integers past
 * 2**53 and would show a value the tool is not being run with.
 */
export function toolCallArgumentsText(
  exactText: unknown,
  args: unknown,
): string {
  if (typeof exactText === "string" && exactText.length > 0) {
    try {
      JSON.parse(exactText);
      return exactText;
    } catch {
      // This card is an approval boundary, so unparsable text falls back to structured args.
    }
  }
  return JSON.stringify(args ?? {});
}

type JsonTextNode =
  | { kind: "object"; entries: Array<{ key: string; value: JsonTextNode }> }
  | { kind: "array"; items: JsonTextNode[] }
  | { kind: "scalar"; raw: string; value: unknown };

const JSON_WHITESPACE = /\s/;
const JSON_VALUE_DELIMITER = /[\s,}\]]/;

class JsonTextParser {
  private offset = 0;
  private readonly text: string;

  constructor(text: string) {
    this.text = text;
  }

  parse(): JsonTextNode {
    const value = this.parseValue();
    this.skipWhitespace();
    if (this.offset !== this.text.length) {
      throw new Error("Trailing JSON text");
    }
    return value;
  }

  private skipWhitespace(): void {
    while (JSON_WHITESPACE.test(this.text[this.offset] ?? "")) {
      this.offset += 1;
    }
  }

  private parseString(): { raw: string; value: string } {
    const start = this.offset;
    this.offset += 1;
    let escaped = false;
    while (this.offset < this.text.length) {
      const char = this.text[this.offset++];
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        const raw = this.text.slice(start, this.offset);
        return { raw, value: JSON.parse(raw) as string };
      }
    }
    throw new Error("Unterminated JSON string");
  }

  private parseValue(): JsonTextNode {
    this.skipWhitespace();
    const char = this.text[this.offset];
    if (char === "{") {
      return this.parseObject();
    }
    if (char === "[") {
      return this.parseArray();
    }
    if (char === '"') {
      const parsed = this.parseString();
      return { kind: "scalar", raw: parsed.raw, value: parsed.value };
    }

    const start = this.offset;
    while (
      this.offset < this.text.length &&
      !JSON_VALUE_DELIMITER.test(this.text[this.offset])
    ) {
      this.offset += 1;
    }
    const raw = this.text.slice(start, this.offset);
    if (!raw) {
      throw new Error("Missing JSON value");
    }
    return { kind: "scalar", raw, value: JSON.parse(raw) as unknown };
  }

  private parseObject(): JsonTextNode {
    this.offset += 1;
    const entries: Array<{ key: string; value: JsonTextNode }> = [];
    this.skipWhitespace();
    if (this.text[this.offset] === "}") {
      this.offset += 1;
      return { kind: "object", entries };
    }
    while (this.offset < this.text.length) {
      this.skipWhitespace();
      if (this.text[this.offset] !== '"') {
        throw new Error("Invalid JSON key");
      }
      const key = this.parseString().value;
      this.skipWhitespace();
      if (this.text[this.offset++] !== ":") {
        throw new Error("Missing JSON colon");
      }
      entries.push({ key, value: this.parseValue() });
      this.skipWhitespace();
      const separator = this.text[this.offset++];
      if (separator === "}") {
        return { kind: "object", entries };
      }
      if (separator !== ",") {
        throw new Error("Invalid JSON object separator");
      }
    }
    throw new Error("Unterminated JSON object");
  }

  private parseArray(): JsonTextNode {
    this.offset += 1;
    const items: JsonTextNode[] = [];
    this.skipWhitespace();
    if (this.text[this.offset] === "]") {
      this.offset += 1;
      return { kind: "array", items };
    }
    while (this.offset < this.text.length) {
      items.push(this.parseValue());
      this.skipWhitespace();
      const separator = this.text[this.offset++];
      if (separator === "]") {
        return { kind: "array", items };
      }
      if (separator !== ",") {
        throw new Error("Invalid JSON array separator");
      }
    }
    throw new Error("Unterminated JSON array");
  }
}

function stringifyJson(value: unknown): string {
  return JSON.stringify(value) ?? "null";
}

function mergeJsonNode(node: JsonTextNode, current: unknown): string {
  if (
    node.kind === "object" &&
    current !== null &&
    typeof current === "object" &&
    !Array.isArray(current)
  ) {
    const record = current as Record<string, unknown>;
    const previousKeys = new Set(node.entries.map((entry) => entry.key));
    const entries = node.entries
      .filter((entry) => Object.hasOwn(record, entry.key))
      .map(
        (entry) =>
          `${JSON.stringify(entry.key)}:${mergeJsonNode(entry.value, record[entry.key])}`,
      );
    for (const [key, value] of Object.entries(record)) {
      if (!previousKeys.has(key))
        entries.push(`${JSON.stringify(key)}:${stringifyJson(value)}`);
    }
    return `{${entries.join(",")}}`;
  }
  if (node.kind === "array" && Array.isArray(current)) {
    return `[${current
      .map((value, index) =>
        index < node.items.length
          ? mergeJsonNode(node.items[index], value)
          : stringifyJson(value),
      )
      .join(",")}]`;
  }
  if (node.kind === "scalar" && Object.is(node.value, current)) {
    return node.raw;
  }
  return stringifyJson(current);
}

/** Unchanged values keep their JSON lexemes (big ints stay exact); overwritten keys re-serialize. */
export function mergedToolCallArgumentsText(
  previousText: unknown,
  mergedArgs: unknown,
  overwrittenKeys: readonly string[] = [],
): string {
  if (typeof previousText !== "string" || previousText.length === 0) {
    return stringifyJson(mergedArgs ?? {});
  }
  try {
    const parsed = new JsonTextParser(previousText).parse();
    if (
      parsed.kind !== "object" ||
      mergedArgs === null ||
      typeof mergedArgs !== "object" ||
      Array.isArray(mergedArgs)
    ) {
      return mergeJsonNode(parsed, mergedArgs);
    }
    const record = mergedArgs as Record<string, unknown>;
    const forced = new Set(overwrittenKeys);
    const previousKeys = new Set(parsed.entries.map((entry) => entry.key));
    const entries = parsed.entries
      .filter((entry) => Object.hasOwn(record, entry.key))
      .map(
        (entry) =>
          `${JSON.stringify(entry.key)}:${
            forced.has(entry.key)
              ? stringifyJson(record[entry.key])
              : mergeJsonNode(entry.value, record[entry.key])
          }`,
      );
    for (const [key, value] of Object.entries(record)) {
      if (!previousKeys.has(key))
        entries.push(`${JSON.stringify(key)}:${stringifyJson(value)}`);
    }
    return `{${entries.join(",")}}`;
  } catch {
    return stringifyJson(mergedArgs ?? {});
  }
}

/** llama-server may send decoded objects (ggml-org/llama.cpp#20198); other non-strings give "". */
export function streamedToolCallArguments(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || typeof value !== "object") {
    return "";
  }
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return "";
  }
}
