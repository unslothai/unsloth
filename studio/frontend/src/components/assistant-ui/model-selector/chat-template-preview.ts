// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type GraphemeSegment = { segment: string };
type GraphemeSegmenter = { segment(input: string): Iterable<GraphemeSegment> };
type SegmenterConstructor = new (
  locales?: string | string[],
  options?: { granularity?: "grapheme" },
) => GraphemeSegmenter;

let cachedGraphemeSegmenter: GraphemeSegmenter | null | undefined;

function positiveInteger(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.max(1, Math.floor(value)) : fallback;
}

function getGraphemeSegmenter(): GraphemeSegmenter | null {
  if (cachedGraphemeSegmenter !== undefined) {
    return cachedGraphemeSegmenter;
  }
  const Segmenter =
    typeof Intl === "undefined"
      ? undefined
      : (Intl as typeof Intl & { Segmenter?: SegmenterConstructor }).Segmenter;
  cachedGraphemeSegmenter =
    typeof Segmenter === "function"
      ? new Segmenter(undefined, { granularity: "grapheme" })
      : null;
  return cachedGraphemeSegmenter;
}

function graphemes(text: string): string[] {
  const segmenter = getGraphemeSegmenter();
  if (!segmenter) {
    return Array.from(text);
  }
  return Array.from(segmenter.segment(text), (part) => part.segment);
}

export function clampLines(
  text: string,
  maxLines = 3,
  maxCharsPerLine = 88,
): string {
  if (!text) return "";
  const lineLimit = positiveInteger(maxCharsPerLine, 88);
  const maxLineCount = positiveInteger(maxLines, 3);
  const out: string[] = [];
  let truncated = false;
  for (const raw of text.split("\n")) {
    if (out.length >= maxLineCount) {
      truncated = true;
      break;
    }
    const parts = graphemes(raw);
    if (parts.length === 0) {
      out.push("");
      continue;
    }
    for (let start = 0; start < parts.length; start += lineLimit) {
      if (out.length >= maxLineCount) {
        truncated = true;
        break;
      }
      out.push(parts.slice(start, start + lineLimit).join(""));
    }
    if (truncated) break;
  }
  if (!truncated) return out.join("\n");
  const last = out.length - 1;
  if (last >= 0) {
    const parts = graphemes(out[last]);
    out[last] =
      parts.length >= lineLimit
        ? `${parts.slice(0, lineLimit - 1).join("")}…`
        : `${out[last]}…`;
  }
  return out.join("\n");
}
