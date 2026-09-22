// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import remend from "remend";
import { markdownBlockFallback } from "./markdown-block-fallback.ts";

export const REASONING_TRANSCRIPT_THRESHOLD = 16_384;
export const REASONING_FRAGMENT_CHARACTERS = 8_192;
const CODE_LINES_PER_FRAGMENT = 16;

export type ReasoningCodeLine = { line: number; column: number; text: string };
export type ReasoningReadingAnchor = { text: string; top: number };
export type ReasoningFragment = {
  key: string;
  document: number;
  start: number;
  end: number;
  text: string;
  renderText?: string;
  /** Continuations have no paragraph gap or repeated code header. */
  first: boolean;
  last: boolean;
  code?: {
    source: string;
    language: string | null;
    lines: readonly ReasoningCodeLine[];
  };
};

type Block = {
  start: number;
  end: number;
  text: string;
  continued: boolean;
  prefix?: string;
};
type Fence = { marker: string; scan: number };

type MarkdownNode = {
  type: string;
  value?: string;
  position?: { start: { offset?: number }; end: { offset?: number } };
  children?: MarkdownNode[];
};

/** Used once at the short-to-windowed transition, not on streaming updates. */
export function findReasoningAnchor(
  fragments: readonly ReasoningFragment[],
  text: string,
): number {
  const plain = (node: MarkdownNode): string =>
    node.value ?? node.children?.map(plain).join("") ?? "";
  return fragments.findIndex((fragment) =>
    (fragment.code
      ? fragment.text
      : plain(fromMarkdown(fragment.renderText ?? fragment.text))
    ).includes(text),
  );
}

// Context is render-only: copy/export always use the original document. A long
// paragraph can cross a bold/code span; a quoted or nested fence can cross many
// fragments. Reopen only the construct at the split, never an unrelated block.
function continuationPrefix(
  text: string,
  nodes: readonly MarkdownNode[],
): string {
  let last = nodes.at(-1);
  while (last?.children?.length) last = last.children.at(-1);
  if (last?.type === "code") {
    const start = last.position?.start.offset ?? 0;
    const lineStart = text.lastIndexOf("\n", start - 1) + 1;
    const lineEnd = text.indexOf("\n", start);
    const opener = text.slice(lineStart, lineEnd < 0 ? text.length : lineEnd);
    const match = /^([\t >]*(?:[-+*] |\d+[.)] )?)(`{3,}|~{3,})([^\n]*)$/.exec(
      opener,
    );
    const close = /^[\t >]*(`{3,}|~{3,})[\t ]*$/.exec(
      text.trimEnd().split("\n").at(-1) ?? "",
    );
    if (
      match &&
      !(
        close &&
        close[1][0] === match[2][0] &&
        close[1].length >= match[2].length
      )
    )
      return `${opener}\n`;
  }
  const trimmed = text.trimEnd();
  const repaired = remend(trimmed);
  const suffix = repaired.startsWith(trimmed)
    ? repaired.slice(trimmed.length)
    : "";
  return suffix && /^[*_`]+$/.test(suffix)
    ? [...suffix].reverse().join("")
    : "";
}

function fenceAt(source: string, start: number): Fence | null {
  const newline = source.indexOf("\n", start);
  if (newline < 0) return null;
  const match = /^( {0,3})(`{3,}|~{3,})([^\n]*)$/.exec(
    source.slice(start, newline),
  );
  if (!match || (match[2][0] === "`" && match[3].includes("`"))) return null;
  return { marker: match[2], scan: newline + 1 };
}

/** Returns the end of a complete closing line. Only newly appended lines are scanned. */
function fenceEnd(source: string, fence: Fence): number | null {
  for (let from = fence.scan; from < source.length; ) {
    const newline = source.indexOf("\n", from);
    const end = newline < 0 ? source.length : newline;
    const line = source.slice(from, end);
    const match = /^ {0,3}(`{3,}|~{3,})[\t ]*\r?$/.exec(line);
    if (
      match &&
      match[1][0] === fence.marker[0] &&
      match[1].length >= fence.marker.length
    ) {
      // A partial closing line can acquire non-whitespace on the next append.
      return newline < 0 ? null : newline + 1;
    }
    if (newline < 0) break;
    from = newline + 1;
    fence.scan = from;
  }
  return null;
}

/** A UTF-16 boundary which cannot bisect CRLF or an astral character. */
export function reasoningFragmentEnd(
  text: string,
  start: number,
  budget: number,
): number {
  let end = Math.min(text.length, start + budget);
  if (
    end < text.length &&
    (/[\uD800-\uDBFF]/.test(text[end - 1]) ||
      (text[end - 1] === "\r" && text[end] === "\n"))
  )
    end -= 1;
  return Math.max(start + 1, end);
}

// Parse a bounded unfinished tail, retaining complete blocks. In particular, an open
// fence bypasses Markdown parsing after its opener: appending a token to a 250K fence
// must not ask a Markdown parser to read those 250K characters again.
class DocumentIndex {
  source = "";
  blocks: Block[] = [];
  private committed: Block[] = [];
  private offset = 0;
  private fence: Fence | null = null;
  private continued = false;
  private prefix = "";
  generation = 0;

  update(source: string): void {
    if (source === this.source) return;
    if (!source.startsWith(this.source)) {
      this.committed = [];
      this.offset = 0;
      this.fence = null;
      this.continued = false;
      this.prefix = "";
      this.generation += 1;
    }
    this.source = source;
    const pending: Block[] = [];
    while (this.offset < source.length) {
      if (!this.prefix) this.fence ??= fenceAt(source, this.offset);
      if (this.fence) {
        const end = fenceEnd(source, this.fence);
        const block = {
          start: this.offset,
          end: end ?? source.length,
          text: source.slice(this.offset, end ?? source.length),
          continued: false,
        };
        if (end === null) {
          pending.push(block);
          break;
        }
        this.committed.push(block);
        this.offset = end;
        this.fence = null;
        this.continued = false;
        continue;
      }

      const end = reasoningFragmentEnd(
        source,
        this.offset,
        REASONING_FRAGMENT_CHARACTERS,
      );
      const tail = source.slice(this.offset, end);
      const parsedText = this.prefix + tail;
      const nodes = fromMarkdown(parsedText, {
        extensions: [gfm(), math({ singleDollarTextMath: true })],
        mdastExtensions: [gfmFromMarkdown(), mathFromMarkdown()],
      }).children;
      // Leave the last node editable (Setext headings, list continuations, open fences).
      const lastStart = Math.max(
        0,
        (nodes.at(-1)?.position?.start.offset ?? 0) - this.prefix.length,
      );
      if (lastStart > 0) {
        if (nodes.length === 1) {
          this.committed.push({
            start: this.offset,
            end: this.offset + lastStart,
            text: tail.slice(0, lastStart),
            continued: this.continued,
            prefix: this.prefix,
          });
        }
        for (let n = 0; n < nodes.length - 1; n += 1) {
          const start =
            n === 0
              ? 0
              : Math.max(
                  0,
                  nodes[n].position!.start.offset! - this.prefix.length,
                );
          const stop = Math.max(
            0,
            nodes[n + 1].position!.start.offset! - this.prefix.length,
          );
          this.committed.push({
            start: this.offset + start,
            end: this.offset + stop,
            text: tail.slice(start, stop),
            continued: n === 0 && this.continued,
            prefix: n === 0 ? this.prefix : "",
          });
        }
        this.offset += lastStart;
        this.continued = false;
        this.prefix = "";
        continue;
      }
      if (end < source.length) {
        // A single enormous paragraph/list/line still has to remain bounded. Prefer a
        // complete line, then whitespace, and keep the continuation visually adjacent.
        let cut = tail.lastIndexOf("\n");
        if (cut < tail.length / 2) cut = tail.lastIndexOf(" ");
        if (cut < tail.length / 2) cut = tail.length;
        else cut += 1;
        this.committed.push({
          start: this.offset,
          end: this.offset + cut,
          text: tail.slice(0, cut),
          continued: this.continued,
          prefix: this.prefix,
        });
        this.prefix = continuationPrefix(
          this.prefix + tail.slice(0, cut),
          nodes,
        );
        this.offset += cut;
        this.continued = true;
        continue;
      }
      pending.push({
        start: this.offset,
        end,
        text: tail,
        continued: this.continued,
        prefix: this.prefix,
      });
      break;
    }
    this.blocks = [...this.committed, ...pending];
  }
}

function fragmentsOf(
  block: Block,
  document: number,
  generation: number,
): ReasoningFragment[] {
  const fallback = markdownBlockFallback(
    block.prefix ? block.prefix + block.text : block.text,
  );
  const key = `${document}:${generation}:${block.start}`;
  if (!fallback.fenced) {
    return [
      {
        key,
        document,
        start: block.start,
        end: block.end,
        text: block.text,
        renderText: block.prefix ? block.prefix + block.text : undefined,
        first: !block.continued,
        last: true,
      },
    ];
  }
  // Match the highlighter's line convention while retaining original bytes in DocumentIndex.
  const source = fallback.text.replace(/\r\n/g, "\n").replace(/\n$/, "");
  const lines = source.split("\n");
  const result: ReasoningFragment[] = [];
  let group: ReasoningCodeLine[] = [];
  let characters = 0;
  const flush = () => {
    if (group.length === 0) return;
    const firstLine = group[0];
    result.push({
      key: `${key}:code:${firstLine.line}:${firstLine.column}`,
      document,
      start: block.start,
      end: block.end,
      text: group.map((line) => line.text).join("\n"),
      first: result.length === 0,
      last: false,
      code: { source, language: fallback.language, lines: group },
    });
    group = [];
    characters = 0;
  };
  for (let line = 0; line < lines.length; line += 1) {
    const text = lines[line];
    for (let column = 0; column < Math.max(1, text.length); ) {
      const end = reasoningFragmentEnd(
        text,
        column,
        REASONING_FRAGMENT_CHARACTERS,
      );
      const part = text.slice(column, end);
      if (
        characters + part.length + 1 > REASONING_FRAGMENT_CHARACTERS ||
        group.length === CODE_LINES_PER_FRAGMENT
      )
        flush();
      group.push({ line, column, text: part });
      characters += part.length + 1;
      column = end;
    }
  }
  flush();
  result[result.length - 1].last = true;
  return result;
}

/** Stored Markdown documents never share fence or paragraph state. */
export class ReasoningTranscriptIndex {
  private documents: DocumentIndex[] = [];
  private fragments = new WeakMap<Block, ReasoningFragment[]>();

  update(sources: readonly string[]): ReasoningFragment[] {
    this.documents.length = sources.length;
    return sources.flatMap((source, document) => {
      const index = (this.documents[document] ??= new DocumentIndex());
      index.update(source);
      return index.blocks.flatMap((block) => {
        let fragments = this.fragments.get(block);
        if (!fragments) {
          fragments = fragmentsOf(block, document, index.generation);
          this.fragments.set(block, fragments);
        }
        return fragments;
      });
    });
  }
}
