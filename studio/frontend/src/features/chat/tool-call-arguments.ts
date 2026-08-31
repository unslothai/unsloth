// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The `function.arguments` string to replay for a stored tool call.
 *
 * `argsText` is the text the provider streamed and is preferred so replay is
 * byte-exact. Text that does not parse would be replayed on every later request
 * in the thread, and strict chat templates reject the whole request rather than
 * one call, so it falls back to the structured args the part already carries.
 */
export function toolCallReplayArguments(
  argsText: string | undefined,
  args: unknown,
): string {
  if (typeof argsText === "string" && argsText.length > 0) {
    try {
      JSON.parse(argsText);
      return argsText;
    } catch {
      // unparsable, so the structured args below stand in for it
    }
  }
  return JSON.stringify(args ?? {});
}

/**
 * The text to render for a tool call's arguments.
 *
 * `JSON.parse` rounds an integer past 2**53 while reading the event, so re-encoding the
 * parsed arguments in the browser would show a value the tool is not being run with. The
 * backend sends its own encoding of the same arguments, which is preferred when present.
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
      // Text that does not parse cannot be these arguments, so it is not what the tool
      // runs with either. This card is an approval boundary, so fall back to the
      // structured arguments rather than print something unreadable next to Allow.
      // `toolCallReplayArguments` above screens the same field the same way.
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
 * The text to render once a later event has had the chance to merge more arguments in.
 *
 * Unchanged values retain their original JSON lexemes while new metadata is added from
 * the structured object. Keys explicitly overwritten by tool_end are serialized from
 * that event instead. This keeps an executed integer past 2**53 exact without dropping
 * Gemini result parts or intentional tool_end replacements.
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
