// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `true` when `text` is exactly one JSON object and nothing else. */
function isSingleJsonObject(text: string): boolean {
  try {
    const value = JSON.parse(text);
    return typeof value === "object" && value !== null && !Array.isArray(value);
  } catch {
    return false;
  }
}

/** One slot's accumulated argument text, cut into the top-level JSON objects it holds:
 *  `complete` closed, `tail` still being written. A second top-level `{` means the stream
 *  reused the slot for another parallel call, which is what turns two calls into
 *  `{"url":"a"}{"url":"b"}`. Text that is not a run of whole objects comes back whole in
 *  `tail`, leaving streams this was never meant for alone. */
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
      // A backslash escapes one character, so a run of them toggles.
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (depth === 0) {
      // Between objects only whitespace, "\r\n" as readily as "\n".
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

/** `splitTopLevelJsonObjects` over a string that only ever grows. Rescanning per fragment is
 *  O(N^2): a 20 KB argument sent a character at a time took about a second and a half on the
 *  thread that paints the stream. `feed` takes the same string extended, never a rewritten
 *  one, so a caller that splits a slot drops its scan. */
export function createBoundaryScan(): {
  feed: (text: string) => { complete: string[]; tail: string };
} {
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  let scanned = 0;
  // Once unsplittable, appending can never make it splittable again.
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

/** The `function.arguments` string to replay for a stored tool call. `argsText` is what the
 *  provider streamed and is preferred so replay is byte-exact. Text that does not parse would
 *  be replayed on every later request in the thread, and strict chat templates reject the
 *  whole request rather than one call, so it falls back to the structured args. `{ _raw }` is
 *  the adapter's marker for unparsable text, so a thread carrying one replays as `{}`. */
export function toolCallReplayArguments(
  argsText: string | undefined,
  args: unknown,
): string {
  if (
    typeof argsText === "string" &&
    argsText.length > 0 &&
    // One object, because that is what `function.arguments` is. A run of them, an array, a scalar or
    // a half-written object gets the whole request rejected, not just the one call.
    isSingleJsonObject(argsText)
  ) {
    return argsText;
  }
  const serialized = JSON.stringify(args ?? {});
  // Not a set of named parameters, whatever else it might be.
  if (serialized === undefined || !isSingleJsonObject(serialized)) {
    return "{}";
  }
  const parsed = JSON.parse(serialized) as Record<string, unknown>;
  const keys = Object.keys(parsed);
  // The adapter writes `{ _raw }` holding the exact text it could not parse, so a lone `_raw`
  // whose value IS that text is the marker, and replaying it would send a parameter no tool
  // declares. `_raw` is not reserved and an MCP server's schema is its own, so recognise it only
  // when the surviving text proves it: a thread stored before `argsText` was kept has nothing to
  // compare against, and guessing from the value's shape would discard a real argument.
  if (
    keys.length === 1 &&
    keys[0] === "_raw" &&
    typeof parsed._raw === "string" &&
    // Non-empty, or the equality proves nothing: the adapter only writes the marker for parsed text,
    // so `{ _raw: "" }` is a real argument.
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
      // Unparseable text cannot be these arguments, and this card is an approval
      // boundary, so fall back to the structured ones.
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

/**
 * Unchanged values keep their original JSON lexemes (an executed integer past 2**53 stays
 * exact); keys in `overwrittenKeys` are re-serialized from the tool_end object.
 */
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
