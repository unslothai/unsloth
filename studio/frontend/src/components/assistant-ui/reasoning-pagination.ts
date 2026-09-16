// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Short traces keep the existing renderer and DOM. Once pagination starts, only
// one page is mounted and each page stays small enough for responsive selection
// and browser find.
export const REASONING_PAGINATION_THRESHOLD = 16_384;
export const REASONING_PAGE_CHARACTERS = 8_192;

const BOUNDARY_SEARCH_CHARACTERS = 1_024;
const BOUNDARY_CONTEXT_CHARACTERS = 128;
const FENCE_LINE_RE = /^( {0,3})(`{3,}|~{3,})(.*)\r?$/;

const HIGH_SURROGATE_RE = /[\uD800-\uDBFF]/;
const LOW_SURROGATE_RE = /[\uDC00-\uDFFF]/;

type FenceRange = {
  end: number;
  start: number;
};

export type ReasoningPage = {
  end: number;
  hasEarlier: boolean;
  hasNewer: boolean;
  markdown: string;
  oversizedCode: boolean;
  start: number;
};

type ReasoningPageOptions = {
  end?: number | null;
  maxCharacters?: number;
  streaming?: boolean;
};

export type ReasoningPageBoundary = {
  after: string;
  before: string;
  end: number;
};

export function shouldPaginateReasoning(markdown: string): boolean {
  return markdown.length > REASONING_PAGINATION_THRESHOLD;
}

export function createReasoningPageBoundary(
  markdown: string,
  end: number,
): ReasoningPageBoundary {
  const boundedEnd = Math.max(0, Math.min(markdown.length, Math.floor(end)));
  return {
    after: markdown.slice(boundedEnd, boundedEnd + BOUNDARY_CONTEXT_CHARACTERS),
    before: markdown.slice(
      Math.max(0, boundedEnd - BOUNDARY_CONTEXT_CHARACTERS),
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

type OpenFence = {
  character: string;
  length: number;
  start: number;
};

function consumeFenceLine(
  line: string,
  lineStart: number,
  lineEnd: number,
  open: OpenFence | null,
): { closed?: FenceRange; open: OpenFence | null } {
  const match = line.match(FENCE_LINE_RE);
  if (!match) {
    return { open };
  }
  const marker = match[2];
  if (
    open &&
    marker[0] === open.character &&
    marker.length >= open.length &&
    match[3].trim().length === 0
  ) {
    return {
      closed: { start: open.start, end: lineEnd },
      open: null,
    };
  }
  if (open) {
    return { open };
  }
  return {
    open: {
      character: marker[0],
      length: marker.length,
      start: lineStart,
    },
  };
}

// Streaming appends should not rescan a million-character trace just to choose
// its last 8 KiB. Only complete newly appended lines advance this scanner; a
// partial trailing line is reconsidered when its newline arrives.
export class ReasoningPageSelector {
  private committedOffset = 0;
  private documentIndex = -1;

  // Each reasoning part is an independent Markdown document, even when adjacent.
  selectDocument(
    documents: readonly string[],
    options: ReasoningPageOptions = {},
  ): ReasoningPage & { documentIndex: number } {
    const total = documents.reduce((length, text) => length + text.length, 0);
    const end = Math.max(0, Math.min(total, options.end ?? total));
    let offset = 0;
    let index = 0;
    while (
      index < documents.length - 1 &&
      offset + documents[index].length < end
    ) {
      offset += documents[index].length;
      index += 1;
    }
    if (index !== this.documentIndex) {
      this.reset();
      this.source = "";
      this.documentIndex = index;
    }
    const page = this.select(documents[index] ?? "", {
      ...options,
      end: end - offset,
    });
    return {
      ...page,
      documentIndex: index,
      start: offset + page.start,
      end: offset + page.end,
      hasEarlier: offset + page.start > 0,
      hasNewer: offset + page.end < total,
    };
  }

  private liveStart: number | null = null;
  private open: OpenFence | null = null;
  private ranges: FenceRange[] = [];
  private source = "";

  private reset(): void {
    this.committedOffset = 0;

    this.liveStart = null;
    this.open = null;
    this.ranges = [];
  }

  private update(markdown: string): readonly FenceRange[] {
    const previousLength = this.source.length;
    const comparisonStart = Math.max(
      0,
      previousLength - BOUNDARY_CONTEXT_CHARACTERS,
    );
    const extendsSource =
      markdown === this.source ||
      (markdown.length > previousLength &&
        markdown.slice(
          0,
          Math.min(previousLength, BOUNDARY_CONTEXT_CHARACTERS),
        ) === this.source.slice(0, BOUNDARY_CONTEXT_CHARACTERS) &&
        markdown.slice(comparisonStart, previousLength) ===
          this.source.slice(comparisonStart));
    if (!extendsSource) {
      this.reset();
    }
    this.source = markdown;

    // A giant unbroken streaming line has no fence state to advance. Search
    // only the appended suffix until a newline finally completes that line.
    if (
      extendsSource &&
      this.committedOffset < previousLength &&
      markdown.indexOf("\n", previousLength) < 0
    ) {
      return this.open
        ? [...this.ranges, { start: this.open.start, end: markdown.length }]
        : this.ranges;
    }

    while (this.committedOffset < markdown.length) {
      const newline = markdown.indexOf("\n", this.committedOffset);
      if (newline < 0) {
        break;
      }
      const result = consumeFenceLine(
        markdown.slice(this.committedOffset, newline),
        this.committedOffset,
        newline + 1,
        this.open,
      );
      this.open = result.open;
      if (result.closed) {
        this.ranges.push(result.closed);
      }
      this.committedOffset = newline + 1;
    }

    return this.open
      ? [...this.ranges, { start: this.open.start, end: markdown.length }]
      : this.ranges;
  }

  select(markdown: string, options: ReasoningPageOptions = {}): ReasoningPage {
    const fences = this.update(markdown);
    const page = selectReasoningPageWithFences(markdown, options, fences);
    if (!options.streaming) {
      return page;
    }

    const maxCharacters = Math.max(
      1,
      Math.floor(options.maxCharacters ?? REASONING_PAGE_CHARACTERS),
    );
    if (this.liveStart === null || this.liveStart > page.end) {
      this.liveStart = page.start;
    }
    const stride = Math.max(1, Math.floor(maxCharacters / 2));
    while (page.end - this.liveStart > maxCharacters) {
      // Advance by half a page. The overlap keeps context on screen and means
      // ordinary token appends do not replace/reparse the full 8 KiB page.
      this.liveStart = pageStart(
        markdown,
        this.liveStart + stride + maxCharacters,
        maxCharacters,
        fences,
      );
    }
    return reasoningPageFromBounds(markdown, this.liveStart, page.end, fences);
  }

  selectRanges(markdown: string): readonly FenceRange[] {
    return this.update(markdown);
  }
}

function fencedRanges(markdown: string): readonly FenceRange[] {
  return new ReasoningPageSelector().selectRanges(markdown);
}

function isInsideFence(offset: number, fences: readonly FenceRange[]): boolean {
  return fences.some((fence) => offset > fence.start && offset < fence.end);
}

function avoidBrokenSurrogate(markdown: string, offset: number): number {
  if (
    offset > 0 &&
    offset < markdown.length &&
    HIGH_SURROGATE_RE.test(markdown[offset - 1]) &&
    LOW_SURROGATE_RE.test(markdown[offset])
  ) {
    return offset + 1;
  }
  return offset;
}

function pageStart(
  markdown: string,
  end: number,
  maxCharacters: number,
  fences: readonly FenceRange[],
): number {
  const target = Math.max(0, end - maxCharacters);
  if (target === 0) {
    return 0;
  }
  const searchEnd = Math.min(end, target + BOUNDARY_SEARCH_CHARACTERS);

  // Prefer a paragraph boundary, then any complete line. Starting after the
  // newline preserves CRLF and avoids changing the previous page's bytes.
  for (const separator of ["\n\n", "\n"] as const) {
    let cursor = target;
    while (cursor < searchEnd) {
      const found = markdown.indexOf(separator, cursor);
      if (found < 0 || found >= searchEnd) {
        break;
      }
      const candidate = found + separator.length;
      if (!isInsideFence(candidate, fences)) {
        return candidate;
      }
      cursor = candidate;
    }
  }

  return avoidBrokenSurrogate(markdown, target);
}

function selectReasoningPageWithFences(
  markdown: string,
  options: ReasoningPageOptions,
  fences: readonly FenceRange[],
): ReasoningPage {
  const maxCharacters = Math.max(
    1,
    Math.floor(options.maxCharacters ?? REASONING_PAGE_CHARACTERS),
  );
  const end = Math.max(
    0,
    Math.min(markdown.length, Math.floor(options.end ?? markdown.length)),
  );
  const start = pageStart(markdown, end, maxCharacters, fences);
  return reasoningPageFromBounds(markdown, start, end, fences);
}

function reasoningPageFromBounds(
  markdown: string,
  start: number,
  end: number,
  fences: readonly FenceRange[],
): ReasoningPage {
  const oversizedCode = fences.some(
    (fence) =>
      (start > fence.start && start < fence.end) ||
      (end > fence.start && end < fence.end),
  );

  return {
    end,
    hasEarlier: start > 0,
    hasNewer: end < markdown.length,
    markdown: markdown.slice(start, end),
    oversizedCode,
    start,
  };
}

export function selectReasoningPage(
  markdown: string,
  options: ReasoningPageOptions = {},
): ReasoningPage {
  return selectReasoningPageWithFences(
    markdown,
    options,
    fencedRanges(markdown),
  );
}
