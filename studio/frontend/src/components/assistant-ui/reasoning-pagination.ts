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
const BLOCK_QUOTE_PREFIX_RE = /^( {0,3}>[ \t]?)/;
const LIST_PREFIX_RE = /^( {0,3})(?:[*+-]|\d{1,9}[.)])([ \t]{1,4})/;
const RAW_HTML_TAG_RE = /^(?: {0,3})<(pre|script|style|textarea)(?:\s|>|$)/i;
const HTML_COMMENT_RE = /^(?: {0,3})<!--/;
const HTML_CDATA_RE = /^(?: {0,3})<!\[CDATA\[/;
const HTML_PROCESSING_RE = /^(?: {0,3})<\?/;
const HTML_DECLARATION_RE = /^(?: {0,3})<![A-Za-z]/;
const HTML_TAG_RE = /^(?: {0,3})<\/?([A-Za-z][A-Za-z0-9-]*)(?:\s|\/?>|$)/;
const HTML_COMPLETE_TAG_RE =
  /^(?: {0,3})(?:<\/[A-Za-z][A-Za-z0-9-]*\s*>|<[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z_:][A-Za-z0-9_.:-]*(?:\s*=\s*(?:[^ "'=<>`]+|'[^']*'|"[^"]*"))?)*\s*\/?>)\s*$/;
const HTML_BLOCK_NAMES = new Set(
  "address article aside base basefont blockquote body caption center col colgroup dd details dialog dir div dl dt fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu menuitem nav noframes ol optgroup option p param search section summary table tbody td tfoot th thead title tr track ul".split(
    " ",
  ),
);
const VOID_HTML_NAMES = new Set(
  "base basefont col frame hr link menuitem param track".split(" "),
);

const OPAQUE_HTML_BLOCKS = [
  [HTML_COMMENT_RE, "-->"],
  [HTML_CDATA_RE, "]]>"],
  [HTML_PROCESSING_RE, "?>"],
  [HTML_DECLARATION_RE, ">"],
] as const;

type ContainerState = {
  continuationPrefix: string;
  openingPrefix: string;
};

type FenceState = {
  character: string;
  container: ContainerState;
  indent: string;
  info: string;
  length: number;
  openingOffset: number;
};

type DisplayMathState = {
  container: ContainerState;
  indent: string;
  length: number;
};

type RawHtmlState = {
  close: "blank" | { caseInsensitive: boolean; value: string };
  container: ContainerState;
  syntheticTag: string | null;
};

type BoundaryState = {
  atBlockBoundary: boolean;
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

type ContainerLine = ContainerState & { content: string };

function parseContainerLine(line: string): ContainerLine {
  let content = line;
  let continuationPrefix = "";
  let openingPrefix = "";

  for (let depth = 0; depth < 16; depth += 1) {
    const quote = content.match(BLOCK_QUOTE_PREFIX_RE);
    if (quote) {
      content = content.slice(quote[0].length);
      continuationPrefix += quote[0];
      openingPrefix += quote[0];
      continue;
    }

    const list = content.match(LIST_PREFIX_RE);
    if (!list) break;
    content = content.slice(list[0].length);
    continuationPrefix += " ".repeat(list[0].length);
    openingPrefix += list[0];
  }

  return { content, continuationPrefix, openingPrefix };
}

function containerContent(
  line: string,
  container: ContainerState,
): string | null {
  const prefix = container.continuationPrefix;
  if (!prefix) return line;
  if (line.startsWith(prefix)) return line.slice(prefix.length);
  return line === prefix.trimEnd() || line.trim().length === 0 ? "" : null;
}

function nextFenceState(
  line: string,
  state: FenceState | null,
  openingOffset: number,
): FenceState | null {
  const parsed = state ? null : parseContainerLine(line);
  const content = state
    ? containerContent(line, state.container)
    : (parsed?.content ?? null);
  if (content === null) return state;

  const match = content.match(FENCE_LINE_RE);
  if (!match) return state;
  const marker = match[2];
  if (state) {
    return marker[0] === state.character &&
      marker.length >= state.length &&
      match[3].trim().length === 0
      ? null
      : state;
  }

  return {
    character: marker[0],
    container: parsed as ContainerState,
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
  const parsed = state ? null : parseContainerLine(line);
  const content = state
    ? containerContent(line, state.container)
    : (parsed?.content ?? null);
  if (content === null) return state;

  const match = content.match(DISPLAY_MATH_LINE_RE);
  if (!match) return state;
  if (state) {
    return match[2].length >= state.length && match[3].trim().length === 0
      ? null
      : state;
  }
  return match[3].includes("$")
    ? null
    : {
        container: parsed as ContainerState,
        indent: match[1],
        length: match[2].length,
      };
}

type RawHtmlStart = { recognized: boolean; state: RawHtmlState | null };

function rawHtmlStart(
  parsed: ContainerLine,
  atBlockBoundary: boolean,
): RawHtmlStart {
  const content = parsed.content;
  const container = parsed as ContainerState;
  const rawTag = content.match(RAW_HTML_TAG_RE);
  if (rawTag) {
    const tag = rawTag[1].toLowerCase();
    const value = `</${tag}>`;
    return {
      recognized: true,
      state: content.toLowerCase().includes(value)
        ? null
        : {
            close: { caseInsensitive: true, value },
            container,
            syntheticTag: tag,
          },
    };
  }

  const opaque = OPAQUE_HTML_BLOCKS.find(([opening]) => opening.test(content));
  if (opaque) {
    return {
      recognized: true,
      state: content.includes(opaque[1])
        ? null
        : {
            close: { caseInsensitive: false, value: opaque[1] },
            container,
            syntheticTag: null,
          },
    };
  }

  const tag = content.match(HTML_TAG_RE)?.[1]?.toLowerCase();
  const isBlockTag = tag !== undefined && HTML_BLOCK_NAMES.has(tag);
  if (
    !isBlockTag &&
    !(atBlockBoundary && tag && HTML_COMPLETE_TAG_RE.test(content))
  ) {
    return { recognized: false, state: null };
  }
  return {
    recognized: true,
    state: {
      close: "blank",
      container,
      syntheticTag: tag && !VOID_HTML_NAMES.has(tag) ? tag : "div",
    },
  };
}

function rawHtmlContinues(line: string, state: RawHtmlState): boolean {
  const content = containerContent(line, state.container);
  if (content === null) return true;
  if (state.close === "blank") return content.trim().length > 0;
  const haystack = state.close.caseInsensitive
    ? content.toLowerCase()
    : content;
  const needle = state.close.caseInsensitive
    ? state.close.value.toLowerCase()
    : state.close.value;
  return !haystack.includes(needle);
}

function nextBoundaryState(
  line: string,
  openingOffset: number,
  state: BoundaryState,
): BoundaryState {
  if (state.fence) {
    const fence = nextFenceState(line, state.fence, openingOffset);
    return { ...state, atBlockBoundary: fence === null, fence };
  }
  if (state.rawHtml) {
    const rawHtml = rawHtmlContinues(line, state.rawHtml)
      ? state.rawHtml
      : null;
    return { ...state, atBlockBoundary: rawHtml === null, rawHtml };
  }
  if (state.displayMath) {
    const displayMath = nextDisplayMathState(line, state.displayMath);
    return {
      ...state,
      atBlockBoundary: displayMath === null,
      displayMath,
    };
  }

  const parsed = parseContainerLine(line);
  const rawHtml = rawHtmlStart(parsed, state.atBlockBoundary);
  if (rawHtml.recognized) {
    return {
      ...state,
      atBlockBoundary: rawHtml.state === null,
      rawHtml: rawHtml.state,
    };
  }

  const displayMath = nextDisplayMathState(line, null);
  if (displayMath) {
    return { ...state, atBlockBoundary: false, displayMath };
  }
  const fence = nextFenceState(line, null, openingOffset);
  if (fence) return { ...state, atBlockBoundary: false, fence };
  return { ...state, atBlockBoundary: parsed.content.trim().length === 0 };
}

function boundaryStateAt(markdown: string, offset: number): BoundaryState {
  let lineStart = 0;
  let state: BoundaryState = {
    atBlockBoundary: true,
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

  const lines: string[] = [];
  let lineStart = openingNewline + 1;
  while (lineStart <= markdown.length) {
    const newline = markdown.indexOf("\n", lineStart);
    const rawLineEnd = newline < 0 ? markdown.length : newline;
    const lineEnd =
      rawLineEnd > lineStart && markdown[rawLineEnd - 1] === "\r"
        ? rawLineEnd - 1
        : rawLineEnd;
    const rawLine = markdown.slice(lineStart, lineEnd);
    const content = containerContent(rawLine, state.container);
    if (content === null) return lines.join("\n");

    const match = content.match(FENCE_LINE_RE);
    if (
      match &&
      match[2][0] === state.character &&
      match[2].length >= state.length &&
      match[3].trim().length === 0
    ) {
      return lines.join("\n");
    }

    let indentation = 0;
    while (indentation < state.indent.length && content[indentation] === " ") {
      indentation += 1;
    }
    lines.push(content.slice(indentation));
    if (newline < 0) break;
    lineStart = newline + 1;
  }
  return lines.join("\n");
}

function syntheticFenceOpening(state: FenceState): string {
  const marker = state.character.repeat(
    Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER),
  );
  const info = state.info.slice(0, MAX_SYNTHETIC_FENCE_INFO);
  return `${state.container.openingPrefix}${state.indent}${marker}${info}`;
}

function syntheticFenceClosing(state: FenceState): string {
  const marker = state.character.repeat(
    Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER),
  );
  return `${state.container.continuationPrefix}${marker}`;
}

function syntheticDisplayMathDelimiter(state: DisplayMathState): string {
  const marker = "$".repeat(Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER));
  return `${state.container.openingPrefix}${state.indent}${marker}`;
}

function syntheticDisplayMathClosing(state: DisplayMathState): string {
  const marker = "$".repeat(Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER));
  return `${state.container.continuationPrefix}${marker}`;
}

function syntheticRawHtmlOpening(state: RawHtmlState): string {
  const marker = state.syntheticTag ? `<${state.syntheticTag}>` : "<!--";
  return `${state.container.openingPrefix}${marker}`;
}

function syntheticRawHtmlClosing(state: RawHtmlState): string {
  const marker = state.syntheticTag ? `</${state.syntheticTag}>` : "-->";
  return `${state.container.continuationPrefix}${marker}`;
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
    pageMarkdown = `${syntheticRawHtmlOpening(openingState.rawHtml)}\n${pageMarkdown}`;
  } else if (openingState.fence) {
    pageMarkdown = `${syntheticFenceOpening(openingState.fence)}\n${pageMarkdown}`;
  } else if (openingState.displayMath) {
    pageMarkdown = `${syntheticDisplayMathDelimiter(openingState.displayMath)}\n${pageMarkdown}`;
  }
  if (closingState.rawHtml) {
    pageMarkdown += `${pageMarkdown.endsWith("\n") ? "" : "\n"}${syntheticRawHtmlClosing(closingState.rawHtml)}\n`;
  } else if (closingState.fence) {
    pageMarkdown += `${pageMarkdown.endsWith("\n") ? "" : "\n"}${syntheticFenceClosing(closingState.fence)}\n`;
  } else if (closingState.displayMath) {
    pageMarkdown += `${pageMarkdown.endsWith("\n") ? "" : "\n"}${syntheticDisplayMathClosing(closingState.displayMath)}\n`;
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
