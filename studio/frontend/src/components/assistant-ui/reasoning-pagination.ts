// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getCompletedCodeFences } from "./streaming-render-schedule.ts";

// One page is several screens of rich Markdown, but remains far below the native
// WebKit DOM cliff measured on long local-model reasoning traces.
export const REASONING_PAGE_CHARACTERS = 8_192;

const BLOCK_BOUNDARY_SEARCH_CHARACTERS = 1_024;

const MAX_SYNTHETIC_FENCE_MARKER = 256;
const MAX_SYNTHETIC_FENCE_INFO = 128;
const FENCE_LINE_RE = /^( {0,3})(`{3,}|~{3,})(.*)$/;

const DISPLAY_MATH_LINE_RE = /^( {0,3})(\${2,})(.*)$/;

const RAW_HTML_OPEN_RE = /^(?: {0,3})<(pre|script|style|textarea)(?:\s|>|$)/i;

type FenceState = {
  character: string;
  indent: string;
  info: string;
  length: number;
  openingOffset: number;
};
type DisplayMathState = {
  indent: string;
  length: number;
};

type RawHtmlState = {
  tag: string;
};

type BoundaryState = {
  displayMath: DisplayMathState | null;
  fence: FenceState | null;
  rawHtml: RawHtmlState | null;
};

export type ReasoningMarkdownPage = {
  end: number;
  hasEarlier: boolean;
  hasNewer: boolean;
  markdown: string;
  canonicalCodeSources: readonly (string | null)[];
  start: number;
};

type ReasoningMarkdownPageOptions = {
  enabled: boolean;
  end?: number | null;
  maxCharacters?: number;

  streaming?: boolean;
};

function nextFenceState(
  line: string,
  state: FenceState | null,
  openingOffset: number,
): FenceState | null {
  const match = line.match(FENCE_LINE_RE);
  if (!match) return state;

  const marker = match[2];
  if (state) {
    const isCloser =
      marker[0] === state.character &&
      marker.length >= state.length &&
      match[3].trim().length === 0;
    return isCloser ? null : state;
  }

  return {
    character: marker[0],
    indent: match[1],
    info: match[3],
    length: marker.length,
    openingOffset,
  };
}

function nextDisplayMathState(
  line: string,
  state: DisplayMathState | null,
): DisplayMathState | null {
  const match = line.match(DISPLAY_MATH_LINE_RE);
  if (!match) return state;

  if (state) {
    return match[2].length >= state.length && match[3].trim().length === 0
      ? null
      : state;
  }
  return match[3].includes("$")
    ? null
    : { indent: match[1], length: match[2].length };
}
function rawHtmlCloses(line: string, tag: string): boolean {
  return line.toLowerCase().includes(`</${tag}>`);
}

function nextBoundaryState(
  line: string,
  openingOffset: number,
  state: BoundaryState,
): BoundaryState {
  if (state.fence) {
    return {
      ...state,
      fence: nextFenceState(line, state.fence, openingOffset),
    };
  }
  if (state.rawHtml) {
    return rawHtmlCloses(line, state.rawHtml.tag)
      ? { ...state, rawHtml: null }
      : state;
  }
  if (state.displayMath) {
    return {
      ...state,
      displayMath: nextDisplayMathState(line, state.displayMath),
    };
  }

  const rawHtmlOpening = line.match(RAW_HTML_OPEN_RE);
  if (rawHtmlOpening) {
    const tag = rawHtmlOpening[1].toLowerCase();
    return rawHtmlCloses(line, tag) ? state : { ...state, rawHtml: { tag } };
  }

  const displayMath = nextDisplayMathState(line, null);
  if (displayMath) return { ...state, displayMath };

  return {
    ...state,
    fence: nextFenceState(line, null, openingOffset),
  };
}

function boundaryStateAt(markdown: string, offset: number): BoundaryState {
  let lineStart = 0;
  let state: BoundaryState = {
    displayMath: null,
    fence: null,
    rawHtml: null,
  };

  while (lineStart < offset) {
    const newline = markdown.indexOf("\n", lineStart);
    const lineEnd = newline < 0 ? markdown.length : newline;
    if (lineEnd > offset) break;

    state = nextBoundaryState(
      markdown.slice(lineStart, lineEnd),
      lineStart,
      state,
    );
    if (newline < 0) break;
    lineStart = newline + 1;
  }

  return state;
}

function canonicalFenceSource(
  markdown: string,
  state: FenceState,
): string | null {
  const openingNewline = markdown.indexOf("\n", state.openingOffset);
  if (openingNewline < 0) return "";
  const sourceStart = openingNewline + 1;
  let lineStart = sourceStart;

  while (lineStart < markdown.length) {
    const newline = markdown.indexOf("\n", lineStart);
    const lineEnd = newline < 0 ? markdown.length : newline;
    const match = markdown.slice(lineStart, lineEnd).match(FENCE_LINE_RE);
    const closes =
      match &&
      match[2][0] === state.character &&
      match[2].length >= state.length &&
      match[3].trim().length === 0;
    if (closes) {
      let sourceEnd = lineStart;
      if (markdown.slice(sourceStart, sourceEnd).endsWith("\r\n"))
        sourceEnd -= 2;
      else if (markdown.slice(sourceStart, sourceEnd).endsWith("\n"))
        sourceEnd -= 1;
      return markdown.slice(sourceStart, sourceEnd).replace(/\r\n?/g, "\n");
    }
    if (newline < 0) break;
    lineStart = newline + 1;
  }

  return markdown.slice(sourceStart).replace(/\r\n?/g, "\n");
}

function syntheticFenceOpening(state: FenceState): string {
  const marker = state.character.repeat(
    Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER),
  );
  const info = state.info.slice(0, MAX_SYNTHETIC_FENCE_INFO);
  return `${state.indent}${marker}${info}`;
}

function syntheticFenceClosing(state: FenceState): string {
  return state.character.repeat(
    Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER),
  );
}
function syntheticDisplayMathDelimiter(state: DisplayMathState): string {
  return `${state.indent}${"$".repeat(
    Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER),
  )}`;
}

function pageStart(
  markdown: string,
  end: number,
  maxCharacters: number,
  streaming: boolean,
): number {
  const stride = Math.max(1, Math.floor(maxCharacters / 2));
  const target = streaming
    ? Math.max(0, Math.floor((end - stride) / stride) * stride)
    : Math.max(0, end - maxCharacters);
  if (target === 0) return 0;

  // Prefer a complete Markdown block when one is nearby. Long unbroken blocks
  // still fall back to a complete line, then to the exact character budget.
  const boundaryWindow = markdown.slice(
    target,
    Math.min(end, target + BLOCK_BOUNDARY_SEARCH_CHARACTERS),
  );
  const blockBoundary = /\r?\n[\t ]*\r?\n/.exec(boundaryWindow);
  if (blockBoundary) {
    const nextBlock = target + blockBoundary.index + blockBoundary[0].length;
    if (nextBlock < end) return nextBlock;
  }

  const newline = markdown.indexOf("\n", target);
  const nextLine = newline < 0 ? target : newline + 1;
  return nextLine < end ? nextLine : target;
}

// Select exactly one newest-first page. `end` is an absolute source boundary
// returned as `start` by the next-newer page, so an older page remains stable
// while new tokens append. Partial giant fences receive temporary local markers;
// the persisted source and the reasoning Copy action remain untouched.
export function selectReasoningMarkdownPage(
  markdown: string,
  options: ReasoningMarkdownPageOptions,
): ReasoningMarkdownPage {
  if (!options.enabled) {
    return {
      end: markdown.length,
      hasEarlier: false,
      hasNewer: false,
      markdown,
      canonicalCodeSources: [],
      start: 0,
    };
  }

  const maxCharacters = Math.max(
    1,
    Math.floor(options.maxCharacters ?? REASONING_PAGE_CHARACTERS),
  );
  const end = Math.max(
    0,
    Math.min(markdown.length, Math.floor(options.end ?? markdown.length)),
  );
  const start = pageStart(
    markdown,
    end,
    maxCharacters,
    Boolean(options.streaming && end === markdown.length),
  );
  let pageMarkdown = markdown.slice(start, end);
  const openingState = boundaryStateAt(markdown, start);
  const closingState = boundaryStateAt(markdown, end);

  if (openingState.rawHtml) {
    pageMarkdown = `<${openingState.rawHtml.tag}>\n${pageMarkdown}`;
  } else if (openingState.fence) {
    pageMarkdown = `${syntheticFenceOpening(openingState.fence)}\n${pageMarkdown}`;
  } else if (openingState.displayMath) {
    pageMarkdown = `${syntheticDisplayMathDelimiter(openingState.displayMath)}\n${pageMarkdown}`;
  }
  if (closingState.rawHtml) {
    pageMarkdown += `${pageMarkdown.endsWith("\n") ? "" : "\n"}</${closingState.rawHtml.tag}>\n`;
  } else if (closingState.fence) {
    pageMarkdown += `${pageMarkdown.endsWith("\n") ? "" : "\n"}${syntheticFenceClosing(closingState.fence)}\n`;
  } else if (closingState.displayMath) {
    pageMarkdown += `${pageMarkdown.endsWith("\n") ? "" : "\n"}${syntheticDisplayMathDelimiter(closingState.displayMath)}\n`;
  }

  const canonicalCodeSources = getCompletedCodeFences(pageMarkdown).map(
    () => null,
  ) as (string | null)[];
  if (openingState.fence && canonicalCodeSources.length > 0) {
    canonicalCodeSources[0] = canonicalFenceSource(
      markdown,
      openingState.fence,
    );
  }
  if (closingState.fence && canonicalCodeSources.length > 0) {
    const index =
      openingState.fence?.openingOffset === closingState.fence.openingOffset
        ? 0
        : canonicalCodeSources.length - 1;
    canonicalCodeSources[index] = canonicalFenceSource(
      markdown,
      closingState.fence,
    );
  }

  return {
    end,
    hasEarlier: start > 0,
    hasNewer: end < markdown.length,
    markdown: pageMarkdown,
    canonicalCodeSources,
    start,
  };
}
