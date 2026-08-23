// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";

export type StreamingCodeFence = {
  language: string | null;
  source: string;
};

export type TerminalStreamingCodeFence = StreamingCodeFence & {
  fenceMarkdown: string;
  isClosed: boolean;
  openingLine: string;
  openingOffset: number;
  // Preserve the separator before the closing marker; `source` omits it.
  rawSource: string;
};

const FENCE_OPEN_RE = /^( {0,3})(`{3,}|~{3,})([^\r\n]*)$/;
const INFO_SEPARATOR_RE = /[ \t]+/;


export function normalizeCodeFenceLanguage(
  language: string | null,
): string | null {
  return language?.trim().split(INFO_SEPARATOR_RE, 1)[0] || null;
}

export function getCodeFenceFilename(language: string | null): string {
  const extByLanguage: Record<string, string> = {
    bash: "sh",
    "c++": "cpp",
    csharp: "cs",
    javascript: "js",
    js: "js",
    json: "json",
    jsx: "jsx",
    markdown: "md",
    md: "md",
    python: "py",
    py: "py",
    ruby: "rb",
    rust: "rs",
    shell: "sh",
    sh: "sh",
    sql: "sql",
    ts: "ts",
    tsx: "tsx",
    typescript: "ts",
    svg: "svg",
    yaml: "yml",
    yml: "yml",
  };
  const normalized = normalizeCodeFenceLanguage(language)?.toLowerCase();
  const fallbackExt = normalized?.replace(/[^a-z0-9]+/g, "-");
  const ext = normalized
    ? extByLanguage[normalized] || fallbackExt || "txt"
    : "txt";
  return `snippet.${ext}`;
}

export const isClosingFenceLine = (
  line: string,
  marker: "`" | "~",
  minimumLength: number,
): boolean => {
  let index = 0;
  while (index < line.length && index < 4 && line[index] === " ") {
    index += 1;
  }
  if (index > 3) {
    return false;
  }

  const markerStart = index;
  while (line[index] === marker) {
    index += 1;
  }
  if (index - markerStart < minimumLength) {
    return false;
  }

  for (; index < line.length; index += 1) {
    if (line[index] !== " " && line[index] !== "\t") {
      return false;
    }
  }
  return true;
};

// Does any line at or after `from`, which must be a line start, close a fence
// opened with `markerLength` copies of `marker`? A line scan, not a parse, so a
// caller that has already settled the text before `from` pays only for what
// arrived since.
export function hasClosingFenceLine(
  content: string,
  from: number,
  marker: "`" | "~",
  markerLength: number,
): boolean {
  for (let lineStart = from; lineStart <= content.length; ) {
    const lf = content.indexOf("\n", lineStart);
    const rawEnd = lf < 0 ? content.length : lf;
    const end =
      rawEnd > lineStart && content[rawEnd - 1] === "\r" ? rawEnd - 1 : rawEnd;
    if (
      isClosingFenceLine(content.slice(lineStart, end), marker, markerLength)
    ) {
      return true;
    }
    if (lf < 0) return false;
    lineStart = lf + 1;
  }
  return false;
}

// The marker a fence opening at `offset` uses, or null when that is not a fence
// opening line.
export function readFenceMarker(
  content: string,
  offset: number,
): { marker: "`" | "~"; markerLength: number } | null {
  const lf = content.indexOf("\n", offset);
  const rawEnd = lf < 0 ? content.length : lf;
  const end =
    rawEnd > offset && content[rawEnd - 1] === "\r" ? rawEnd - 1 : rawEnd;
  const opening = content.slice(offset, end).match(FENCE_OPEN_RE);
  return opening
    ? { marker: opening[2][0] as "`" | "~", markerLength: opening[2].length }
    : null;
}

type FenceOpening = {
  language: string | null;
  marker: "`" | "~";
  markerLength: number;
  sourceStart: number;
};

const parseFenceOpening = (blockContent: string): FenceOpening | null => {
  const openingLf = blockContent.indexOf("\n");
  if (openingLf < 0) {
    return null;
  }

  const openingEnd =
    openingLf > 0 && blockContent[openingLf - 1] === "\r"
      ? openingLf - 1
      : openingLf;
  const opening = blockContent.slice(0, openingEnd).match(FENCE_OPEN_RE);
  if (!opening) {
    return null;
  }

  const marker = opening[2][0] as "`" | "~";
  const info = opening[3];
  // Backtick fences cannot contain backticks in the info string.
  if (marker === "`" && info.includes("`")) {
    return null;
  }

  return {
    language: normalizeCodeFenceLanguage(info),
    marker,
    markerLength: opening[2].length,
    sourceStart: openingLf + 1,
  };
};

type ClosingFence =
  | { rawSourceEnd: number; sourceEnd: number }
  | "content-after-close"
  | null;

const findClosingFence = (
  blockContent: string,
  opening: FenceOpening,
): ClosingFence => {
  let lineStart = opening.sourceStart;
  while (lineStart <= blockContent.length) {
    const lf = blockContent.indexOf("\n", lineStart);
    const rawLineEnd = lf < 0 ? blockContent.length : lf;
    const lineEnd =
      rawLineEnd > lineStart && blockContent[rawLineEnd - 1] === "\r"
        ? rawLineEnd - 1
        : rawLineEnd;
    const line = blockContent.slice(lineStart, lineEnd);

    if (isClosingFenceLine(line, opening.marker, opening.markerLength)) {
      const suffixStart = lf < 0 ? blockContent.length : lf + 1;
      // A provider block containing content after the close is not code-only. Let
      // Streamdown render it rather than hiding that suffix inside this replacement.
      if (blockContent.slice(suffixStart).trim().length > 0) {
        return "content-after-close";
      }

      let sourceEnd = lineStart;
      if (
        sourceEnd > opening.sourceStart &&
        blockContent[sourceEnd - 1] === "\n"
      ) {
        sourceEnd -= 1;
        if (
          sourceEnd > opening.sourceStart &&
          blockContent[sourceEnd - 1] === "\r"
        ) {
          sourceEnd -= 1;
        }
      }
      return { rawSourceEnd: lineStart, sourceEnd };
    }

    if (lf < 0) {
      return null;
    }
    lineStart = lf + 1;
  }
  return null;
};

// Streamdown passes one parser block here. The shared artifact helper intentionally
// recognizes only a closed, unindented ``` fence, so it cannot select a still-open
// fence. This parser is deliberately narrower than a Markdown splitter: it reads one
// opening line and, when present, its matching closing line. Openness is derived
// from those bytes even after transport completion; every source byte between
// the syntactic lines is returned unchanged.
type ParsedStreamingCodeFence = StreamingCodeFence & {
  isClosed: boolean;
  openingLine: string;
  rawSource: string;
};

const parseStreamingCodeFence = (
  blockContent: string,
): ParsedStreamingCodeFence | null => {
  const opening = parseFenceOpening(blockContent);
  if (!opening) {
    return null;
  }

  const closing = findClosingFence(blockContent, opening);
  if (closing === "content-after-close") {
    return null;
  }
  if (closing) {
    return {
      isClosed: true,
      language: opening.language,
      openingLine: blockContent.slice(0, opening.sourceStart),
      rawSource: blockContent.slice(
        opening.sourceStart,
        closing.rawSourceEnd,
      ),
      source: blockContent.slice(opening.sourceStart, closing.sourceEnd),
    };
  }
  return {
    isClosed: false,
    language: opening.language,
    openingLine: blockContent.slice(0, opening.sourceStart),
    rawSource: blockContent.slice(opening.sourceStart),
    source: blockContent.slice(opening.sourceStart),
  };
};

export function getStreamingCodeFence(
  blockContent: string,
): StreamingCodeFence | null {
  const parsed = parseStreamingCodeFence(blockContent);
  return parsed && { language: parsed.language, source: parsed.source };
}

type PositionedMarkdownNode = {
  children?: readonly PositionedMarkdownNode[];
  position?: {
    end: { offset?: number };
    start: { offset?: number };
  };
  type: string;
};

// Validate the offset with CommonMark itself rather than treating every line
// that resembles a fence as Markdown. In particular, a marker inside a raw HTML
// block is an HTML byte, not a fenced-code opener. The parser's final top-level
// code node also proves that a nonterminal fence is not selected; a real terminal
// closer remains explicit in the returned syntactic state.
const terminalCodeNodeOffset = (markdown: string): number | null => {
  const root = fromMarkdown(markdown) as PositionedMarkdownNode;
  const node = root.children?.at(-1);
  const start = node?.position?.start.offset;
  const end = node?.position?.end.offset;
  if (
    node?.type !== "code" ||
    start === undefined ||
    end === undefined ||
    markdown.slice(end).trim().length > 0
  ) {
    return null;
  }
  const lineStart = markdown.lastIndexOf("\n", start - 1) + 1;
  const indentation = markdown.slice(lineStart, start);
  return indentation.length <= 3 && /^ *$/.test(indentation)
    ? lineStart
    : start;
};

// Find only the final fenced code node inside one exact raw mutable provider
// block. The returned offset is relative to that block; callers retain their
// parser/block boundary and never search or slice the canonical whole message.
export function getTerminalStreamingCodeFence(
  blockContent: string,
  // The offset this fence had on the previous chunk. An open fence absorbs every
  // appended character -- CommonMark ends one only at a matching closing line or
  // the end of the document -- so while it stays open its offset cannot move and
  // the whole-tail parse below is recomputing a constant. Trusted only while the
  // line scan still reports it open, which is the case the argument covers.
  knownOpeningOffset?: number,
): TerminalStreamingCodeFence | null {
  if (knownOpeningOffset !== undefined) {
    const fenceMarkdown = blockContent.slice(knownOpeningOffset);
    const parsed = parseStreamingCodeFence(fenceMarkdown);
    if (parsed && !parsed.isClosed) {
      return { ...parsed, fenceMarkdown, openingOffset: knownOpeningOffset };
    }
  }

  const openingOffset = terminalCodeNodeOffset(blockContent);
  if (openingOffset === null) {
    return null;
  }

  const fenceMarkdown = blockContent.slice(openingOffset);
  const parsed = parseStreamingCodeFence(fenceMarkdown);
  if (!parsed) {
    return null;
  }
  return {
    fenceMarkdown,
    isClosed: parsed.isClosed,
    language: parsed.language,
    openingLine: parsed.openingLine,
    openingOffset,
    rawSource: parsed.rawSource,
    source: parsed.source,
  };
}

// The real Qwen 27B profile spends its dominant non-idle samples in TextMate
// while one still-open TypeScript fence keeps growing. Four KiB preserves live
// colour for ordinary snippets, but stops a generated file before its repeated
// whole-tail parse/tokenization becomes a multi-frame task.
export const OVERSIZED_OPEN_CODE_CHARS = 4 * 1024;

// A 7,000-character final tree stayed above the accepted 50-FPS floor, while a
// 19,265-character tree produced a measured 321ms mount. Bound the automatic
// rich transition between those observations. This is deliberately UTF-16 code
// units (`String.length`), not bytes: it counts the same canonical source string
// that the code element and copy/download actions receive. The limit is inclusive.
export const MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS = 16 * 1024;

export const shouldAutoHighlightStreamingCode = (source: string): boolean =>
  source.length <= MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS;

// Keep the wrapper mounted when the fence closes so the exact plain source can
// paint before a permitted final highlighted subtree mounts. Extreme sources use
// the same wrapper but remain plain after completion.
export const isOversizedStreamingCode = (sourceCharacters: number): boolean =>
  sourceCharacters >= OVERSIZED_OPEN_CODE_CHARS;
