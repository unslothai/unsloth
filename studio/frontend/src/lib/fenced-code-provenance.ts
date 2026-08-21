// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const FENCED_CODE_PROVENANCE_KEY = "__unslothFenceProvenance";

export type FencedCodeProvenance = Readonly<{
  v: 1;
  trailingLf: readonly number[];
}>;

type TextPartLike = Readonly<{
  type?: unknown;
  text?: unknown;
  [FENCED_CODE_PROVENANCE_KEY]?: unknown;
}>;

type OpenFence = Readonly<{
  contentStart: number;
  marker: "`" | "~";
  markerLength: number;
  openingOffset: number;
  ordinal: number;
}>;

type LineCandidate = {
  indent: number;
  invalid: boolean;
  marker: "`" | "~" | null;
  markerLength: number;
  phase: "indent" | "marker" | "rest";
  restHasBacktick: boolean;
  restOnlyWhitespace: boolean;
};

type CommittedLine = Readonly<{
  end: number;
  nextOrdinal: number;
  open: OpenFence | null;
  ownedLength: number;
}>;

type OwnedOccurrence = Readonly<{
  closeEnd: number;
  ordinal: number;
}>;

export type FencedCodeProvenanceTrackerStats = Readonly<{
  rewinds: number;
  scannedCharacters: number;
}>;

export type CompletedFencedCodeOccurrence = Readonly<{
  bodyWithSeparator: string;
  closingOffset: number;
  openingOffset: number;
  ordinal: number;
}>;

const createLineCandidate = (): LineCandidate => ({
  indent: 0,
  invalid: false,
  marker: null,
  markerLength: 0,
  phase: "indent",
  restHasBacktick: false,
  restOnlyWhitespace: true,
});

const appendCandidateCharacter = (
  candidate: LineCandidate,
  character: string,
): void => {
  if (candidate.invalid) return;

  if (candidate.phase === "indent") {
    if (character === " ") {
      candidate.indent += 1;
      if (candidate.indent > 3) candidate.invalid = true;
      return;
    }
    if (character === "`" || character === "~") {
      candidate.marker = character;
      candidate.markerLength = 1;
      candidate.phase = "marker";
      return;
    }
    candidate.invalid = true;
    return;
  }

  if (candidate.phase === "marker" && character === candidate.marker) {
    candidate.markerLength += 1;
    return;
  }

  candidate.phase = "rest";
  candidate.restOnlyWhitespace &&=
    character === " " || character === "\t";
  candidate.restHasBacktick ||= character === "`";
};

const isOpeningCandidate = (
  candidate: LineCandidate,
): candidate is LineCandidate & { marker: "`" | "~" } =>
  !candidate.invalid &&
  candidate.marker !== null &&
  candidate.markerLength >= 3 &&
  !(candidate.marker === "`" && candidate.restHasBacktick);

const isClosingCandidate = (
  candidate: LineCandidate,
  open: OpenFence,
): boolean =>
  !candidate.invalid &&
  candidate.marker === open.marker &&
  candidate.markerLength >= open.markerLength &&
  candidate.restOnlyWhitespace;

const normalizeLineEndings = (text: string): string =>
  text.includes("\r") ? text.replace(/\r\n?/g, "\n") : text;

const sharedPrefixLength = (left: string, right: string): number => {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left.charCodeAt(index) === right.charCodeAt(index)) {
    index += 1;
  }
  if (
    index > 0 &&
    index < left.length &&
    index < right.length &&
    left.charCodeAt(index - 1) >= 0xd800 &&
    left.charCodeAt(index - 1) <= 0xdbff
  ) {
    index -= 1;
  }
  return index;
};

class TextPartFenceTracker {
  private raw = "";
  private canonical = "";
  private lineStart = 0;
  private candidate = createLineCandidate();
  private open: OpenFence | null = null;
  private nextOrdinal = 0;
  private lines: CommittedLine[] = [];
  private owned: OwnedOccurrence[] = [];
  private publishedLfBoundaries = new Map<number, number>();
  private scannedCharacters = 0;
  private rewinds = 0;

  private commitLine(end: number): void {
    if (this.open) {
      if (isClosingCandidate(this.candidate, this.open)) {
        if (this.publishedLfBoundaries.get(this.lineStart) === this.open.ordinal) {
          this.owned.push({ closeEnd: end, ordinal: this.open.ordinal });
        }
        this.open = null;
      }
    } else if (isOpeningCandidate(this.candidate)) {
      this.open = {
        contentStart: end,
        marker: this.candidate.marker,
        markerLength: this.candidate.markerLength,
        openingOffset: this.lineStart + this.candidate.indent,
        ordinal: this.nextOrdinal,
      };
      this.nextOrdinal += 1;
    }

    this.lines.push({
      end,
      nextOrdinal: this.nextOrdinal,
      open: this.open,
      ownedLength: this.owned.length,
    });
    this.lineStart = end;
    this.candidate = createLineCandidate();
  }

  private appendCanonical(delta: string): void {
    const base = this.canonical.length;
    for (let index = 0; index < delta.length; index += 1) {
      const character = delta[index];
      this.scannedCharacters += 1;
      if (character === "\n") {
        this.commitLine(base + index + 1);
      } else {
        appendCandidateCharacter(this.candidate, character);
      }
    }
    this.canonical += delta;
  }

  private rewind(nextCanonical: string): void {
    this.rewinds += 1;
    const shared = sharedPrefixLength(this.canonical, nextCanonical);
    const previousNewline = this.canonical.lastIndexOf("\n", Math.max(0, shared - 1));
    const safeStart = previousNewline < 0 ? 0 : previousNewline + 1;
    let retainedLines = this.lines.length;
    while (retainedLines > 0 && this.lines[retainedLines - 1].end > safeStart) {
      retainedLines -= 1;
    }
    this.lines.length = retainedLines;
    const checkpoint = this.lines.at(-1);
    this.open = checkpoint?.open ?? null;
    this.nextOrdinal = checkpoint?.nextOrdinal ?? 0;
    this.owned.length = checkpoint?.ownedLength ?? 0;
    this.lineStart = checkpoint?.end ?? 0;
    this.candidate = createLineCandidate();
    for (const [offset, ordinal] of this.publishedLfBoundaries) {
      if (
        offset > safeStart ||
        !this.open ||
        ordinal !== this.open.ordinal
      ) {
        this.publishedLfBoundaries.delete(offset);
      }
    }
    this.canonical = this.canonical.slice(0, safeStart);
    this.appendCanonical(nextCanonical.slice(safeStart));
  }

  private markPublishedBoundary(): void {
    if (
      this.open &&
      this.canonical.endsWith("\n") &&
      this.lineStart === this.canonical.length &&
      this.lineStart > this.open.contentStart
    ) {
      this.publishedLfBoundaries.set(this.lineStart, this.open.ordinal);
    }
  }

  update(raw: string): readonly number[] {
    if (raw === this.raw) {
      this.markPublishedBoundary();
      return this.currentOwnedOrdinals();
    }

    const appends =
      raw.length >= this.raw.length &&
      raw.slice(0, this.raw.length) === this.raw;
    if (appends) {
      let rawDelta = raw.slice(this.raw.length);
      if (this.raw.endsWith("\r") && rawDelta.startsWith("\n")) {
        rawDelta = rawDelta.slice(1);
      }
      this.appendCanonical(normalizeLineEndings(rawDelta));
    } else {
      this.rewind(normalizeLineEndings(raw));
    }
    this.raw = raw;
    this.markPublishedBoundary();
    return this.currentOwnedOrdinals();
  }

  private currentOwnedOrdinals(): readonly number[] {
    const ordinals = this.owned.map((entry) => entry.ordinal);
    if (
      this.open &&
      isClosingCandidate(this.candidate, this.open) &&
      this.publishedLfBoundaries.get(this.lineStart) === this.open.ordinal
    ) {
      ordinals.push(this.open.ordinal);
    }
    return ordinals;
  }

  stats(): FencedCodeProvenanceTrackerStats {
    return {
      rewinds: this.rewinds,
      scannedCharacters: this.scannedCharacters,
    };
  }
}

/**
 * Adds provenance only to assistant text snapshots that are actually published.
 * Text parts are tracked independently by their text-part ordinal; tool,
 * reasoning, source, image, and other parts are never inspected or annotated.
 */
export class AssistantFencedCodeProvenanceTracker {
  private textParts: TextPartFenceTracker[] = [];

  annotate<T extends { type: string }>(parts: readonly T[]): T[] {
    let textPartIndex = 0;
    const annotated = parts.map((part) => {
      if (part.type !== "text") return part;
      const tracker =
        this.textParts[textPartIndex] ?? new TextPartFenceTracker();
      this.textParts[textPartIndex] = tracker;
      textPartIndex += 1;

      const record = part as T & TextPartLike;
      const ordinals =
        typeof record.text === "string" ? tracker.update(record.text) : [];
      const plain = { ...record };
      delete (plain as Record<string, unknown>)[FENCED_CODE_PROVENANCE_KEY];
      if (ordinals.length === 0) return plain as T;
      return {
        ...plain,
        [FENCED_CODE_PROVENANCE_KEY]: { v: 1, trailingLf: [...ordinals] },
      } as T;
    });
    this.textParts.length = textPartIndex;
    return annotated;
  }

  stats(): FencedCodeProvenanceTrackerStats {
    return this.textParts.reduce(
      (total, tracker) => {
        const stats = tracker.stats();
        return {
          rewinds: total.rewinds + stats.rewinds,
          scannedCharacters: total.scannedCharacters + stats.scannedCharacters,
        };
      },
      { rewinds: 0, scannedCharacters: 0 },
    );
  }
}

/** Return a strictly validated, immutable occurrence list from one text part. */
export function readFencedCodeProvenance(
  part: TextPartLike | null | undefined,
): readonly number[] {
  const value = part?.[FENCED_CODE_PROVENANCE_KEY];
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const record = value as Record<string, unknown>;
  if (record.v !== 1 || !Array.isArray(record.trailingLf)) return [];
  if (Object.keys(record).some((key) => key !== "v" && key !== "trailingLf")) {
    return [];
  }

  const ordinals: number[] = [];
  let previous = -1;
  for (const value of record.trailingLf) {
    if (
      typeof value !== "number" ||
      !Number.isSafeInteger(value) ||
      value < 0 ||
      value <= previous
    ) {
      return [];
    }
    ordinals.push(value);
    previous = value;
  }
  return ordinals;
}

/**
 * Finds completed top-level CommonMark fence occurrences in document order.
 * The renderer cross-checks these offsets against mdast before applying any
 * persisted source change, so imported or stale metadata cannot guess a body.
 */
export function getCompletedFencedCodeOccurrences(
  rawMarkdown: string,
): readonly CompletedFencedCodeOccurrence[] {
  const markdown = normalizeLineEndings(rawMarkdown);
  const occurrences: CompletedFencedCodeOccurrence[] = [];
  let open: OpenFence | null = null;
  let nextOrdinal = 0;
  let lineStart = 0;

  while (lineStart <= markdown.length) {
    const newline = markdown.indexOf("\n", lineStart);
    const lineEnd = newline < 0 ? markdown.length : newline;
    const candidate = createLineCandidate();
    for (let index = lineStart; index < lineEnd; index += 1) {
      appendCandidateCharacter(candidate, markdown[index]);
    }

    if (open) {
      if (isClosingCandidate(candidate, open)) {
        occurrences.push({
          bodyWithSeparator: markdown.slice(open.contentStart, lineStart),
          closingOffset: lineStart + candidate.indent,
          openingOffset: open.openingOffset,
          ordinal: open.ordinal,
        });
        open = null;
      }
    } else if (isOpeningCandidate(candidate)) {
      open = {
        contentStart: newline < 0 ? markdown.length : newline + 1,
        marker: candidate.marker,
        markerLength: candidate.markerLength,
        openingOffset: lineStart + candidate.indent,
        ordinal: nextOrdinal,
      };
      nextOrdinal += 1;
    }

    if (newline < 0) break;
    lineStart = newline + 1;
  }
  return occurrences;
}
