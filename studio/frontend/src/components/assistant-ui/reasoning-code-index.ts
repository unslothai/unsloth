// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ReasoningCodeLine,
  ReasoningFragment,
} from "./reasoning-transcript-index.ts";

const CHARACTERS = 8192;
const LINES = 16;
type Options = {
  bodyStart: number;
  indent: number;
  language: string | null;
  key: string;
  document: number;
  start: number;
};

/** Retain completed rows; only the bounded, unfinished row is revisited on append. */
export class ReasoningCodeIndex {
  private rows: ReasoningFragment[] = [];
  private cursor: number;
  private line = 0;
  private column = 0;
  private committedSource = "";
  private context = { source: "", incomplete: true };
  private options: Options;
  private previousText = "";
  private previousEnd = -1;
  private result: ReasoningFragment[] = [];

  constructor(options: Options) {
    this.options = options;
    this.cursor = options.bodyStart;
  }

  update(text: string, bodyEnd: number, incomplete: boolean): ReasoningFragment[] {
    this.context.incomplete = incomplete;
    if (text === this.previousText && bodyEnd === this.previousEnd)
      return this.result;
    this.previousText = text;
    this.previousEnd = bodyEnd;
    // A fence's terminal newline is not an extra blank code line. It becomes
    // interior on the next append and is then consumed by the pending row.
    let end = bodyEnd;
    let trailingIndent = 0;
    while (
      trailingIndent < this.options.indent &&
      text[end - trailingIndent - 1] === " "
    )
      trailingIndent += 1;
    if (text[end - trailingIndent - 1] === "\n") end -= trailingIndent;
    if (text[end - 1] === "\n") end -= text[end - 2] === "\r" ? 2 : 1;
    else if (text[end - 1] === "\r") end -= 1;
    const result = [...this.rows];
    let cursor = this.cursor;
    let line = this.line;
    let column = this.column;
    let group: ReasoningCodeLine[] = [];
    let characters = 0;
    let pendingSource = "";
    const flush = (commit: boolean) => {
      if (!group.length) return;
      const first = group[0];
      const context = this.context;
      const row: ReasoningFragment = {
        key: `${this.options.key}:code:${first.line}:${first.column}`,
        document: this.options.document,
        start: this.options.start,
        end: this.options.start + cursor,
        text: group.map((part) => part.text).join("\n"),
        first: result.length === 0,
        last: !commit,
        code: {
          get source() {
            return context.source;
          },
          get incomplete() {
            return context.incomplete;
          },
          language: this.options.language,
          lines: group,
        },
      };
      result.push(row);
      if (commit) {
        this.rows.push(row);
        this.cursor = cursor;
        this.line = line;
        this.column = column;
        this.committedSource += pendingSource;
        pendingSource = "";
      }
      group = [];
      characters = 0;
    };
    while (cursor < end) {
      const newline = text.indexOf("\n", cursor);
      const rawEnd = newline < 0 || newline >= end ? end : newline;
      const lineEnd =
        text[rawEnd - 1] === "\r" && newline === rawEnd ? rawEnd - 1 : rawEnd;
      let from = cursor;
      let carry = "";
      if (column === 0) {
        let indentation = 0;
        while (indentation < this.options.indent && from < lineEnd) {
          if (text[from] === " ") indentation += 1;
          else if (text[from] === "\t") indentation += 4 - (indentation % 4);
          else break;
          from += 1;
        }
        carry = " ".repeat(Math.max(0, indentation - this.options.indent));
      }
      const budget = CHARACTERS - characters - carry.length;
      let stop = Math.min(lineEnd, from + budget);
      if (stop < lineEnd && /[\uD800-\uDBFF]/.test(text[stop - 1])) stop -= 1;
      if (stop <= from && from < lineEnd) {
        flush(true);
        continue;
      }
      const part = carry + text.slice(from, stop);
      group.push({ line, column, text: part });
      characters += part.length;
      pendingSource += part;
      if (stop === lineEnd && rawEnd < end) {
        cursor = rawEnd + 1;
        line += 1;
        column = 0;
        pendingSource += "\n";
        characters += 1;
      } else {
        cursor = stop === lineEnd ? rawEnd : stop;
        column += part.length;
      }
      if (
        cursor < end &&
        (characters >= CHARACTERS - 1 || group.length >= LINES)
      )
        flush(true);
    }
    if (!group.length && result.length === 0)
      group.push({ line: 0, column: 0, text: "" });
    flush(false);
    this.context.source = this.committedSource + pendingSource;
    if (cursor > end)
      this.context.source = this.context.source.replace(/\n$/, "");
    if (result.length && !result.at(-1)!.last)
      result[result.length - 1] = { ...result.at(-1)!, last: true };
    this.result = result;
    return result;
  }
}
