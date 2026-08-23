// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getCompletedCodeFences } from "./streaming-render-schedule.ts";

// One page is several screens of rich Markdown, but remains far below the native
// WebKit DOM cliff measured on long local-model reasoning traces.
export const REASONING_PAGE_CHARACTERS = 8_192;

const BLOCK_BOUNDARY_SEARCH_CHARACTERS = 1_024;

const MAX_SYNTHETIC_FENCE_MARKER = 256;
const MAX_SYNTHETIC_FENCE_INFO = 128;

const MAX_SYNTHETIC_HTML_OPENING = 256;
const PAGE_BOUNDARY_CONTEXT_CHARACTERS = 128;
// `.` never matches CR, so both accept the CR of a CRLF ending explicitly.
const FENCE_LINE_RE = /^( {0,3})(`{3,}|~{3,})(.*)\r?$/;
const TABLE_DELIMITER_CELL_RE = /^:?-+:?$/;
const DISPLAY_MATH_LINE_RE = /^( {0,3})(\${2,})(.*)\r?$/;
const BLOCK_QUOTE_PREFIX_RE = /^( {0,3}>[ \t]?)/;
const LIST_PREFIX_RE = /^( {0,3})(?:[*+-]|\d{1,9}[.)])([ \t]{1,4})/;
const RAW_HTML_TAG_RE = /^(?: {0,3})<(pre|script|style|textarea)(?:\s|>|$)/i;
const HTML_COMMENT_RE = /^(?: {0,3})<!--/;
const HTML_CDATA_RE = /^(?: {0,3})<!\[CDATA\[/;
const HTML_PROCESSING_RE = /^(?: {0,3})<\?/;
const HTML_DECLARATION_RE = /^(?: {0,3})<![A-Za-z]/;
const HTML_TAG_RE = /^(?: {0,3})<\/?([A-Za-z][A-Za-z0-9-]*)(?:\s|\/?>|$)/;
const HTML_OPENING_TAG_RE =
  /^(?: {0,3})(<[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z_:][A-Za-z0-9_.:-]*(?:\s*=\s*(?:[^ "'=<>`]+|'[^']*'|"[^"]*"))?)*\s*\/?>)/;
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
  [HTML_COMMENT_RE, "<!--", "-->"],
  [HTML_CDATA_RE, "<![CDATA[", "]]>"],
  [HTML_PROCESSING_RE, "<?", "?>"],
  [HTML_DECLARATION_RE, "<!A", ">"],
] as const;

type ContainerFrame =
  | { kind: "quote"; opening: string }
  | { continuation: string; kind: "list"; opening: string };

type ContainerState = {
  continuationPrefix: string;
  frames: readonly ContainerFrame[];
  hasList: boolean;
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
  syntheticClosing: string;
  syntheticOpening: string;
  syntheticTag: string | null;
};

type TableHeaderState = {
  container: ContainerState;
  markdown: string;
};

type TableState = TableHeaderState & {
  delimiter: string;
};

type BoundaryState = {
  atBlockBoundary: boolean;
  displayMath: DisplayMathState | null;
  fence: FenceState | null;

  list: ContainerState | null;
  rawHtml: RawHtmlState | null;
  table: TableState | null;
  tableHeader: TableHeaderState | null;
};

export type ReasoningMarkdownPage = {
  end: number;
  hasEarlier: boolean;
  hasNewer: boolean;
  markdown: string;
  canonicalCodeSources: readonly (string | null)[];
  start: number;
};

export type ReasoningPageBoundary = {
  after: string;
  before: string;
  end: number;
};

export function createReasoningPageBoundary(
  markdown: string,
  end: number,
): ReasoningPageBoundary {
  const boundedEnd = Math.max(0, Math.min(markdown.length, Math.floor(end)));
  return {
    after: markdown.slice(
      boundedEnd,
      boundedEnd + PAGE_BOUNDARY_CONTEXT_CHARACTERS,
    ),
    before: markdown.slice(
      Math.max(0, boundedEnd - PAGE_BOUNDARY_CONTEXT_CHARACTERS),
      boundedEnd,
    ),
    end: boundedEnd,
  };
}

export function isReasoningPageBoundaryValid(
  markdown: string,
  boundary: ReasoningPageBoundary,
): boolean {
  return (
    boundary.end <= markdown.length &&
    markdown.slice(boundary.end, boundary.end + boundary.after.length) ===
      boundary.after &&
    markdown.slice(boundary.end - boundary.before.length, boundary.end) ===
      boundary.before
  );
}

type ReasoningMarkdownPageOptions = {
  enabled: boolean;
  end?: number | null;
  maxCharacters?: number;

  streaming?: boolean;
};

type ContainerLine = ContainerState & { content: string };

// Splitting on "\n" leaves the CR of a CRLF ending on the line.
function lineAt(markdown: string, start: number, end: number): string {
  return markdown.slice(
    start,
    end > start && markdown[end - 1] === "\r" ? end - 1 : end,
  );
}

function firstLineOf(markdown: string): string {
  const newline = markdown.indexOf("\n");
  return lineAt(markdown, 0, newline < 0 ? markdown.length : newline);
}

// Synthetic scaffolding joins in the document's own line ending.
function documentLineEnding(markdown: string): string {
  const newline = markdown.indexOf("\n");
  return newline > 0 && markdown[newline - 1] === "\r" ? "\r\n" : "\n";
}

function parseContainerLine(line: string): ContainerLine {
  let content = line;
  let continuationPrefix = "";
  const frames: ContainerFrame[] = [];
  let openingPrefix = "";

  for (let depth = 0; depth < 16; depth += 1) {
    const quote = content.match(BLOCK_QUOTE_PREFIX_RE);
    if (quote) {
      content = content.slice(quote[0].length);
      continuationPrefix += quote[0];
      frames.push({ kind: "quote", opening: quote[0] });
      openingPrefix += quote[0];
      continue;
    }

    const list = content.match(LIST_PREFIX_RE);
    if (!list) break;
    content = content.slice(list[0].length);
    const continuation = " ".repeat(list[0].length);
    continuationPrefix += continuation;
    frames.push({ continuation, kind: "list", opening: list[0] });
    openingPrefix += list[0];
  }

  return {
    content,
    continuationPrefix,
    frames,
    hasList: frames.some((frame) => frame.kind === "list"),
    openingPrefix,
  };
}

function containerContent(
  line: string,
  container: ContainerState,
): string | null {
  let content = line;
  for (const frame of container.frames) {
    if (frame.kind === "quote") {
      const quote = content.match(BLOCK_QUOTE_PREFIX_RE);
      if (!quote) return content.trim().length === 0 ? "" : null;
      content = content.slice(quote[0].length);
      continue;
    }
    if (!content.startsWith(frame.continuation)) {
      return content.trim().length === 0 ? "" : null;
    }
    content = content.slice(frame.continuation.length);
  }
  return content;
}

function parseWithinContainer(
  content: string,
  outer: ContainerState,
): ContainerLine {
  const nested = parseContainerLine(content);
  return {
    content: nested.content,
    continuationPrefix: outer.continuationPrefix + nested.continuationPrefix,
    frames: [...outer.frames, ...nested.frames],
    hasList: outer.hasList || nested.hasList,
    openingPrefix: outer.openingPrefix + nested.openingPrefix,
  };
}

function hasUnescapedTablePipe(content: string): boolean {
  let backslashes = 0;
  for (const character of content) {
    if (character === "\\") {
      backslashes += 1;
      continue;
    }
    if (character === "|" && backslashes % 2 === 0) return true;
    backslashes = 0;
  }
  return false;
}

function isTableDelimiter(content: string): boolean {
  const trimmed = content.trim();
  if (!hasUnescapedTablePipe(trimmed)) return false;
  const cells = trimmed.replace(/^\|/, "").replace(/\|$/, "").split("|");
  return (
    cells.length > 0 &&
    cells.every((cell) => TABLE_DELIMITER_CELL_RE.test(cell.trim()))
  );
}

function isTableBodyContent(content: string): boolean {
  if (content.trim().length === 0) return false;
  const parsed = parseContainerLine(content);
  if (parsed.frames.length > 0) return false;
  if (rawHtmlStart(parsed, true).recognized) return false;
  if (nextFenceState(content, null, 0, parsed)) return false;
  if (nextDisplayMathState(content, null, parsed)) return false;
  return !/^(?: {0,3})(?:#{1,6}(?:[ \t]|$)|(?:[*_-][ \t]*){3,})/.test(content);
}

function nextFenceState(
  line: string,
  state: FenceState | null,
  openingOffset: number,
  parsedOverride?: ContainerLine,
): FenceState | null {
  const parsed = state ? null : (parsedOverride ?? parseContainerLine(line));
  const content = state
    ? containerContent(line, state.container)
    : (parsed?.content ?? null);
  if (content === null) return null;

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
  parsedOverride?: ContainerLine,
): DisplayMathState | null {
  const parsed = state ? null : (parsedOverride ?? parseContainerLine(line));
  const content = state
    ? containerContent(line, state.container)
    : (parsed?.content ?? null);
  if (content === null) return null;

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

            syntheticClosing: `</${tag}>`,
            syntheticOpening: boundedRawHtmlOpening(
              content.match(HTML_OPENING_TAG_RE)?.[1] ?? `<${tag}>`,
              tag,
            ),
            syntheticTag: tag,
          },
    };
  }

  const opaque = OPAQUE_HTML_BLOCKS.find(([opening]) => opening.test(content));
  if (opaque) {
    return {
      recognized: true,
      state: content.includes(opaque[2])
        ? null
        : {
            close: { caseInsensitive: false, value: opaque[2] },
            container,
            syntheticClosing: opaque[2],
            syntheticOpening: opaque[1],
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
  const syntheticTag = tag && !VOID_HTML_NAMES.has(tag) ? tag : "div";
  const openingTag = content.match(HTML_OPENING_TAG_RE)?.[1];
  return {
    recognized: true,
    state: {
      close: "blank",
      container,

      syntheticClosing: `</${syntheticTag}>`,
      syntheticOpening:
        openingTag && !openingTag.trimEnd().endsWith("/>")
          ? boundedRawHtmlOpening(openingTag, syntheticTag)
          : `<${syntheticTag}>`,
      syntheticTag,
    },
  };
}

function rawHtmlContinues(line: string, state: RawHtmlState): boolean {
  const content = containerContent(line, state.container);
  if (content === null) return false;
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
  let current = state;
  if (current.table) {
    const content = containerContent(line, current.table.container);
    if (content !== null && isTableBodyContent(content)) {
      return { ...current, atBlockBoundary: false, tableHeader: null };
    }
    current = {
      ...current,
      atBlockBoundary: true,
      table: null,
      tableHeader: null,
    };
  }
  if (current.fence) {
    if (containerContent(line, current.fence.container) === null) {
      current = { ...current, atBlockBoundary: true, fence: null };
    } else {
      const fence = nextFenceState(line, current.fence, openingOffset);
      return { ...current, atBlockBoundary: fence === null, fence };
    }
  }
  if (current.rawHtml) {
    if (containerContent(line, current.rawHtml.container) === null) {
      current = { ...current, atBlockBoundary: true, rawHtml: null };
    } else {
      const rawHtml = rawHtmlContinues(line, current.rawHtml)
        ? current.rawHtml
        : null;
      return { ...current, atBlockBoundary: rawHtml === null, rawHtml };
    }
  }
  if (current.displayMath) {
    if (containerContent(line, current.displayMath.container) === null) {
      current = { ...current, atBlockBoundary: true, displayMath: null };
    } else {
      const displayMath = nextDisplayMathState(line, current.displayMath);
      return {
        ...current,
        atBlockBoundary: displayMath === null,
        displayMath,
      };
    }
  }

  let parsed: ContainerLine;
  if (current.list) {
    const content = containerContent(line, current.list);
    if (content === null) {
      current = { ...current, atBlockBoundary: true, list: null };
      parsed = parseContainerLine(line);
    } else {
      parsed = parseWithinContainer(content, current.list);
    }
  } else {
    parsed = parseContainerLine(line);
  }

  if (current.tableHeader) {
    const content = containerContent(line, current.tableHeader.container);
    if (content !== null && isTableDelimiter(content)) {
      return {
        ...current,
        atBlockBoundary: false,
        list: parsed.hasList ? parsed : null,
        table: {
          ...current.tableHeader,
          delimiter: line,
        },
        tableHeader: null,
      };
    }
  }

  const list = parsed.hasList ? parsed : null;
  const rawHtml = rawHtmlStart(parsed, current.atBlockBoundary);
  if (rawHtml.recognized) {
    return {
      ...current,
      atBlockBoundary: rawHtml.state === null,
      list,
      rawHtml: rawHtml.state,
      tableHeader: null,
    };
  }

  const displayMath = nextDisplayMathState(line, null, parsed);
  if (displayMath) {
    return {
      ...current,
      atBlockBoundary: false,
      displayMath,
      list,
      tableHeader: null,
    };
  }
  const fence = nextFenceState(line, null, openingOffset, parsed);
  if (fence) {
    return {
      ...current,
      atBlockBoundary: false,
      fence,
      list,
      tableHeader: null,
    };
  }
  return {
    ...current,
    atBlockBoundary: parsed.content.trim().length === 0,
    list,
    tableHeader: hasUnescapedTablePipe(parsed.content)
      ? { container: parsed, markdown: line }
      : null,
  };
}

function boundaryStateAt(markdown: string, offset: number): BoundaryState {
  let lineStart = 0;
  let state: BoundaryState = {
    atBlockBoundary: true,
    displayMath: null,
    fence: null,

    list: null,
    rawHtml: null,
    table: null,
    tableHeader: null,
  };

  while (lineStart < offset) {
    const newline = markdown.indexOf("\n", lineStart);
    const lineEnd = newline < 0 ? markdown.length : newline;
    if (lineEnd > offset) break;

    state = nextBoundaryState(
      lineAt(markdown, lineStart, lineEnd),
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

  // Each line keeps its own ending so the source stays byte exact.
  let source = "";
  let ending = "";
  let lineStart = openingNewline + 1;
  while (lineStart <= markdown.length) {
    const newline = markdown.indexOf("\n", lineStart);
    const rawLineEnd = newline < 0 ? markdown.length : newline;
    const rawLine = lineAt(markdown, lineStart, rawLineEnd);
    const content = containerContent(rawLine, state.container);
    if (content === null) return source;

    const match = content.match(FENCE_LINE_RE);
    if (
      match &&
      match[2][0] === state.character &&
      match[2].length >= state.length &&
      match[3].trim().length === 0
    ) {
      return source;
    }

    let indentation = 0;
    while (indentation < state.indent.length && content[indentation] === " ") {
      indentation += 1;
    }
    source += ending + content.slice(indentation);
    ending = markdown.slice(lineStart + rawLine.length, rawLineEnd + 1);
    if (newline < 0) break;
    lineStart = newline + 1;
  }
  return source;
}

function syntheticFenceOpening(state: FenceState): string {
  const marker = state.character.repeat(
    Math.min(state.length, MAX_SYNTHETIC_FENCE_MARKER),
  );
  const info = state.info.slice(0, MAX_SYNTHETIC_FENCE_INFO);
  return `${state.container.openingPrefix}${state.indent}${marker}${info}`;
}

function boundedRawHtmlOpening(opening: string, tag: string): string {
  if (opening.length <= MAX_SYNTHETIC_HTML_OPENING) return opening;
  const keepsDetailsOpen =
    tag === "details" && /\sopen(?:\s|=|\/?\s*>)/i.test(opening);
  return `<${tag}${keepsDetailsOpen ? " open" : ""}>`;
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
  return `${state.container.openingPrefix}${state.syntheticOpening}`;
}

function syntheticRawHtmlClosing(state: RawHtmlState): string {
  return `${state.container.continuationPrefix}${state.syntheticClosing}`;
}

function safeCharacterBoundary(markdown: string, index: number): number {
  const previous = markdown.charCodeAt(index - 1);
  const current = markdown.charCodeAt(index);
  return previous >= 0xd800 &&
    previous <= 0xdbff &&
    current >= 0xdc00 &&
    current <= 0xdfff
    ? index + 1
    : index;
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
  if (newline < 0) return safeCharacterBoundary(markdown, target);
  const nextLine = newline + 1;
  return nextLine < end ? nextLine : safeCharacterBoundary(markdown, target);
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
  const newline = documentLineEnding(markdown);

  if (openingState.rawHtml) {
    pageMarkdown = `${syntheticRawHtmlOpening(openingState.rawHtml)}${newline}${pageMarkdown}`;
  } else if (openingState.fence) {
    pageMarkdown = `${syntheticFenceOpening(openingState.fence)}${newline}${pageMarkdown}`;
  } else if (openingState.displayMath) {
    pageMarkdown = `${syntheticDisplayMathDelimiter(openingState.displayMath)}${newline}${pageMarkdown}`;
  } else if (openingState.table) {
    pageMarkdown = `${openingState.table.markdown}${newline}${openingState.table.delimiter}${newline}${pageMarkdown}`;
  } else if (openingState.tableHeader) {
    const content = containerContent(
      firstLineOf(pageMarkdown),
      openingState.tableHeader.container,
    );
    if (content !== null && isTableDelimiter(content)) {
      pageMarkdown = `${openingState.tableHeader.markdown}${newline}${pageMarkdown}`;
    }
  } else if (openingState.list) {
    if (
      containerContent(firstLineOf(pageMarkdown), openingState.list) !== null
    ) {
      pageMarkdown = `${openingState.list.openingPrefix.trimEnd()}${newline}${pageMarkdown}`;
    }
  }
  const closingBreak = pageMarkdown.endsWith("\n") ? "" : newline;
  if (closingState.rawHtml) {
    pageMarkdown += `${closingBreak}${syntheticRawHtmlClosing(closingState.rawHtml)}${newline}`;
  } else if (closingState.fence) {
    pageMarkdown += `${closingBreak}${syntheticFenceClosing(closingState.fence)}${newline}`;
  } else if (closingState.displayMath) {
    pageMarkdown += `${closingBreak}${syntheticDisplayMathClosing(closingState.displayMath)}${newline}`;
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
