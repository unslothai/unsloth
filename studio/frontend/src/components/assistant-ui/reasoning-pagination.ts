// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One page is several screens of rich Markdown, but remains far below the native
// WebKit DOM cliff measured on long local-model reasoning traces.
export const REASONING_PAGE_CHARACTERS = 8_192;

const BLOCK_BOUNDARY_SEARCH_CHARACTERS = 1_024;

const MAX_SYNTHETIC_FENCE_MARKER = 256;
const MAX_SYNTHETIC_FENCE_INFO = 128;
const FENCE_LINE_RE = /^( {0,3})(`{3,}|~{3,})(.*)$/;

type FenceState = {
  character: string;
  indent: string;
  info: string;
  length: number;
};

export type ReasoningMarkdownPage = {
  end: number;
  hasEarlier: boolean;
  hasNewer: boolean;
  markdown: string;
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
  };
}

function fenceStateAt(markdown: string, offset: number): FenceState | null {
  let lineStart = 0;
  let state: FenceState | null = null;

  while (lineStart < offset) {
    const newline = markdown.indexOf("\n", lineStart);
    const lineEnd = newline < 0 ? markdown.length : newline;
    if (lineEnd > offset) break;

    state = nextFenceState(markdown.slice(lineStart, lineEnd), state);
    if (newline < 0) break;
    lineStart = newline + 1;
  }

  return state;
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
  const openingFence = fenceStateAt(markdown, start);
  const closingFence = fenceStateAt(markdown, end);

  if (openingFence) {
    pageMarkdown = `${syntheticFenceOpening(openingFence)}\n${pageMarkdown}`;
  }
  if (closingFence) {
    pageMarkdown += `${pageMarkdown.endsWith("\n") ? "" : "\n"}${syntheticFenceClosing(closingFence)}\n`;
  }

  return {
    end,
    hasEarlier: start > 0,
    hasNewer: end < markdown.length,
    markdown: pageMarkdown,
    start,
  };
}
