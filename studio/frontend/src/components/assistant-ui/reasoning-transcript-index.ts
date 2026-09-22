// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import remend from "remend";
import type { Definition, RootContent } from "mdast";
import { markdownBlockFallback } from "./markdown-block-fallback.ts";
import { ReasoningCodeIndex } from "./reasoning-code-index.ts";

export const REASONING_TRANSCRIPT_THRESHOLD = 16_384;
export const REASONING_FRAGMENT_CHARACTERS = 8_192;

export type ReasoningCodeLine = { line: number; column: number; text: string };
export type ReasoningReadingAnchor = {
  text: string;
  top: number;
  occurrence?: number;
};
export type ReasoningFragment = {
  key: string;
  document: number;
  start: number;
  end: number;
  text: string;
  renderText?: string;
  /** Synthetic list containers continue an item without repeating its marker. */
  listContinuationDepth?: number;
  tableContinuation?: boolean;
  hidden?: boolean;
  /** Continuations have no paragraph gap or repeated code header. */
  first: boolean;
  last: boolean;
  code?: {
    source: string;
    incomplete: boolean;
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
  listContinuationDepth?: number;
  tableContinuation?: boolean;
  hidden?: boolean;
  fence?: {
    bodyStart: number;
    bodyEnd: number;
    incomplete: boolean;
    indent: number;
    language: string | null;
    index?: ReasoningCodeIndex;
  };
};
type Fence = {
  marker: string;
  scan: number;
  bodyStart: number;
  closeStart?: number;
  block?: Block;
};

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
  occurrence = 0,
): number {
  return resolveReasoningAnchor(fragments, { text, top: 0, occurrence }).index;
}

export function resolveReasoningAnchor(
  fragments: readonly ReasoningFragment[],
  anchor: ReasoningReadingAnchor,
): ReasoningReadingAnchor & { index: number } {
  const { text, occurrence = 0 } = anchor;
  const plain = (node: MarkdownNode): string =>
    node.value ?? node.children?.map(plain).join("") ?? "";
  const texts = fragments.map((fragment) =>
    fragment.code
      ? fragment.text
      : plain(fromMarkdown(fragment.renderText ?? fragment.text)),
  );
  const source = texts.join("");
  let start = -1;
  for (let i = 0; i <= occurrence; i += 1) {
    start = source.indexOf(text, start + 1);
    if (start < 0) return { ...anchor, index: -1 };
  }
  let offset = 0;
  for (const [index, part] of texts.entries()) {
    if (offset + part.length > start) {
      const local = start - offset;
      const needle = part.slice(
        local,
        Math.min(part.length, local + text.length),
      );
      let repeated = 0;
      for (
        let at = part.indexOf(needle);
        at >= 0 && at < local;
        at = part.indexOf(needle, at + 1)
      )
        repeated += 1;
      return { ...anchor, text: needle, occurrence: repeated, index };
    }
    offset += part.length;
  }
  return { ...anchor, index: -1 };
}

// Context is render-only: copy/export always use the original document. A long
// paragraph can cross a bold/code span; a quoted or nested fence can cross many
// fragments. Reopen only the construct at the split, never an unrelated block.
function continuationPrefix(
  text: string,
  nodes: readonly MarkdownNode[],
  cut: number,
): { prefix: string; listDepth: number; table?: boolean } {
  const atCut = (siblings: readonly MarkdownNode[]) => {
    if (siblings.some((node) => node.position?.start.offset === cut))
      return undefined;
    return siblings.find(
      (node, i) =>
        (node.position?.start.offset ?? 0) < cut &&
        ((node.position?.end.offset ?? 0) >= cut ||
          (i === siblings.length - 1 &&
            (node.type === "code" || node.type === "blockquote") &&
            !text.slice(node.position?.end.offset, cut).trim())),
    );
  };
  let last = atCut(nodes);
  const table = last?.type === "table" ? last : undefined;
  if (table) {
    const start = table.position?.start.offset ?? 0;
    const headerEnd = text.indexOf("\n", start);
    const delimiterEnd = text.indexOf("\n", headerEnd + 1);
    if (headerEnd >= 0 && delimiterEnd >= 0)
      return {
        prefix: text.slice(start, delimiterEnd + 1),
        listDepth: 0,
        table: true,
      };
  }
  const containers: { marker: string; list: boolean }[] = [];
  while (last) {
    if (last.type === "listItem") {
      const start = last.position?.start.offset ?? 0;
      const marker = /^ {0,3}(?:[-+*]|\d+[.)])[\t ]+/.exec(text.slice(start));
      if (marker) containers.push({ marker: marker[0], list: true });
    }
    if (last.type === "blockquote")
      containers.push({ marker: "> ", list: false });
    const child = last.children && atCut(last.children);
    if (!child) break;
    last = child;
  }
  let indent = "";
  let listPrefix = "";
  for (const [i, { marker, list }] of containers.entries()) {
    listPrefix += marker;
    indent += list ? " ".repeat(marker.length) : marker;
    if (
      list &&
      (containers.slice(i + 1).some((item) => item.list) ||
        last?.type === "list")
    )
      listPrefix += "\n" + indent;
  }
  if (last?.type === "code") {
    const start = last.position?.start.offset ?? 0;
    const lineStart = text.lastIndexOf("\n", start - 1) + 1;
    const lineEnd = text.indexOf("\n", start);
    const opener = text.slice(lineStart, lineEnd < 0 ? text.length : lineEnd);
    const match = /^([\t >]*(?:[-+*] |\d+[.)] )?)(`{3,}|~{3,})([^\n]*)$/.exec(
      opener,
    );
    const close = /^[\t >]*(`{3,}|~{3,})[\t ]*$/.exec(
      text.slice(0, cut).trimEnd().split("\n").at(-1) ?? "",
    );
    if (
      match &&
      !(
        close &&
        close[1][0] === match[2][0] &&
        close[1].length >= match[2].length
      )
    )
      return { prefix: `${opener}\n`, listDepth: 0 };
  }
  const trimmed = text.slice(0, cut).trimEnd();
  const repaired = remend(trimmed);
  const suffix = repaired.startsWith(trimmed)
    ? repaired.slice(trimmed.length)
    : "";
  const inlinePrefix =
    suffix && /^[*_`]+$/.test(suffix) ? [...suffix].reverse().join("") : "";
  return {
    prefix: listPrefix + inlinePrefix,
    listDepth: containers.filter((item) => item.list).length,
  };
}

function* referenceDefinitions(
  nodes: readonly RootContent[],
): Generator<Definition> {
  for (const node of nodes) {
    if (node.type === "definition") yield node;
    else if ("children" in node) yield* referenceDefinitions(node.children);
  }
}

function definitionSource(node: Definition): string {
  // Serialize without the quote/list container: definitions are document-scoped.
  const escape = (value: string) =>
    value.replace(/&/g, "&amp;").replace(/[\\"]/g, "\\$&");
  const url = escape(
    node.url.replace(/[<>\s]/g, (character) => encodeURIComponent(character)),
  );
  const title = node.title == null ? "" : ` "${escape(node.title)}"`;
  return `[${node.label ?? node.identifier}]: <${url}>${title}`;
}

const normalizeReference = (label: string): string =>
  label
    .replace(/[\t\n\r ]+/g, " ")
    .trim()
    .toLowerCase()
    .toUpperCase();

function fenceAt(source: string, start: number): Fence | null {
  const newline = source.indexOf("\n", start);
  if (newline < 0) return null;
  const match = /^( {0,3})(`{3,}|~{3,})([^\n]*)$/.exec(
    source.slice(start, newline),
  );
  if (!match || (match[2][0] === "`" && match[3].includes("`"))) return null;
  return { marker: match[2], scan: newline + 1, bodyStart: newline + 1 };
}

/** Returns the end of a complete closing line. Only newly appended lines are scanned. */
function fenceEnd(source: string, fence: Fence): number | null {
  fence.closeStart = undefined;
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
      fence.closeStart = from;
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
  private listContinuationDepth = 0;
  private tableContinuation = false;
  definitions = new Map<string, { start: number; source: string }>();
  definitionsRevision = 0;
  generation = 0;

  update(source: string): void {
    if (source === this.source) return;
    if (!source.startsWith(this.source)) {
      this.committed = [];
      this.offset = 0;
      this.fence = null;
      this.continued = false;
      this.prefix = "";
      this.listContinuationDepth = 0;
      this.tableContinuation = false;
      this.definitions.clear();
      this.definitionsRevision += 1;
      this.generation += 1;
    }
    // Only the unfinished tail can change a previously seen definition.
    const previousDefinitions = new Map(this.definitions);
    for (const [key, definition] of this.definitions) {
      if (definition.start >= this.offset) this.definitions.delete(key);
    }
    this.source = source;
    const pending: Block[] = [];
    while (this.offset < source.length) {
      if (!this.prefix) this.fence ??= fenceAt(source, this.offset);
      if (this.fence) {
        const end = fenceEnd(source, this.fence);
        const block = (this.fence.block ??= {
          start: this.offset,
          end: end ?? source.length,
          text: source.slice(this.offset, end ?? source.length),
          continued: false,
          fence: {
            bodyStart: this.fence.bodyStart - this.offset,
            bodyEnd: 0,
            incomplete: true,
            indent: /^ */.exec(source.slice(this.offset))![0].length,
            language: markdownBlockFallback(
              source.slice(this.offset, this.fence.bodyStart),
            ).language,
          },
        });
        block.end = end ?? source.length;
        block.text = source.slice(this.offset, block.end);
        block.fence!.bodyEnd =
          (this.fence.closeStart ?? block.end) - this.offset;
        block.fence!.incomplete = this.fence.closeStart === undefined;
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
      for (const node of referenceDefinitions(nodes)) {
        const start =
          this.offset + node.position!.start.offset! - this.prefix.length;
        const key = normalizeReference(node.identifier);
        const previous = this.definitions.get(key);
        // CommonMark uses the first definition; only its still-streaming source can change.
        if (previous && previous.start !== start) continue;
        const definition = definitionSource(node);
        if (previous?.source === definition) continue;
        this.definitions.set(key, { start, source: definition });
      }
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
            listContinuationDepth: this.listContinuationDepth,
            tableContinuation: this.tableContinuation,
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
            listContinuationDepth: n === 0 ? this.listContinuationDepth : 0,
            tableContinuation: n === 0 && this.tableContinuation,
            hidden: nodes[n].type === "definition",
          });
        }
        this.offset += lastStart;
        this.continued = false;
        this.prefix = "";
        this.listContinuationDepth = 0;
        this.tableContinuation = false;
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
          listContinuationDepth: this.listContinuationDepth,
          tableContinuation: this.tableContinuation,
        });
        const context = continuationPrefix(
          parsedText,
          nodes,
          this.prefix.length + cut,
        );
        this.prefix = context.prefix;
        this.listContinuationDepth = context.listDepth;
        this.tableContinuation = Boolean(context.table);
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
        listContinuationDepth: this.listContinuationDepth,
        tableContinuation: this.tableContinuation,
        hidden: nodes.every((node) => node.type === "definition"),
      });
      break;
    }
    this.blocks = [...this.committed, ...pending];
    if (
      previousDefinitions.size !== this.definitions.size ||
      [...this.definitions].some(
        ([key, definition]) =>
          previousDefinitions.get(key)?.source !== definition.source,
      )
    )
      this.definitionsRevision += 1;
  }
}

function fragmentsOf(
  block: Block,
  document: number,
  generation: number,
): ReasoningFragment[] {
  const key = `${document}:${generation}:${block.start}`;
  return [
    {
      key,
      document,
      start: block.start,
      end: block.end,
      text: block.text,
      renderText: block.prefix ? block.prefix + block.text : undefined,
      listContinuationDepth: block.listContinuationDepth,
      tableContinuation: block.tableContinuation,
      hidden: block.hidden,
      first: !block.continued,
      last: true,
    },
  ];
}

/** Stored Markdown documents never share fence or paragraph state. */
export class ReasoningTranscriptIndex {
  private documents: DocumentIndex[] = [];
  private fragments = new WeakMap<
    Block,
    {
      revision: number;
      text: string;
      base: ReasoningFragment[];
      rows: ReasoningFragment[];
    }
  >();

  update(sources: readonly string[]): ReasoningFragment[] {
    this.documents.length = sources.length;
    return sources.flatMap((source, document) => {
      const index = (this.documents[document] ??= new DocumentIndex());
      index.update(source);
      return index.blocks.flatMap((block) => {
        // Preview-capable fences retain the established renderer and its settings.
        // Ordinary code keeps the same incremental renderer and scroll geometry.
        if (
          block.fence &&
          (block.text.length > REASONING_FRAGMENT_CHARACTERS ||
            !["mermaid", "svg", "xml", "html"].includes(
              block.fence.language?.toLowerCase() ?? "",
            ))
        ) {
          const fence = block.fence;
          fence.index ??= new ReasoningCodeIndex({
            bodyStart: fence.bodyStart,
            indent: fence.indent,
            language: fence.language,
            key: `${document}:${index.generation}:${block.start}`,
            document,
            start: block.start,
          });
          return fence.index.update(block.text, fence.bodyEnd, fence.incomplete);
        }
        let cached = this.fragments.get(block);
        if (
          !cached ||
          cached.revision !== index.definitionsRevision ||
          cached.text !== block.text
        ) {
          const base =
            cached?.text === block.text
              ? cached.base
              : fragmentsOf(block, document, index.generation);
          const rows = base.map((row) => ({ ...row }));
          if (index.definitions.size) {
            for (const row of rows) {
              if (row.code) continue;
              const rendered = row.renderText ?? row.text;
              const used = new Set<string>();
              for (const match of rendered.matchAll(
                /\[((?:\\.|[^\]\\]){1,999})\]/g,
              )) {
                const definition = index.definitions.get(
                  normalizeReference(match[1]),
                );
                if (definition) used.add(definition.source);
              }
              if (used.size)
                row.renderText = [...used].join("\n\n") + "\n\n" + rendered;
            }
          }
          // New definitions only invalidate consumers; ordinary prose and code keep
          // their memoized row identity even when a distant reference is completed.
          if (cached?.text === block.text) {
            for (let i = 0; i < rows.length; i += 1) {
              if (rows[i].renderText === cached.rows[i]?.renderText)
                rows[i] = cached.rows[i];
            }
          }
          cached = {
            revision: index.definitionsRevision,
            text: block.text,
            base,
            rows,
          };
          this.fragments.set(block, cached);
        }
        return cached.rows;
      });
    });
  }
}
