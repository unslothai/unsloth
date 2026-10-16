// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const RAG_SOURCES_SENTINEL = "__RAG_SOURCES__:";

export interface Citation {
  id: string;
  filename: string;
  page?: number | null;
  score?: number | null;
  text: string;
  documentId?: string | null;
  chunkId?: string | null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// null if absent so callers fall back to generic JSON shapes.
function parseSentinelSources(result: unknown): Citation[] | null {
  if (typeof result !== "string") return null;
  const idx = result.indexOf(RAG_SOURCES_SENTINEL);
  if (idx < 0) return null;
  const payload = result.slice(idx + RAG_SOURCES_SENTINEL.length).trim();
  let rows: unknown;
  try {
    rows = JSON.parse(payload);
  } catch {
    return [];
  }
  if (!Array.isArray(rows)) return [];
  return rows.map((row, i) => {
    const r = (row ?? {}) as Record<string, unknown>;
    const documentId = typeof r.documentId === "string" ? r.documentId : null;
    const chunkId = typeof r.chunkId === "string" ? r.chunkId : null;
    const filename =
      typeof r.filename === "string" ? r.filename : `Source ${i + 1}`;
    return {
      id: chunkId ?? `${filename}-${i}`,
      filename,
      page: asNumber(r.page),
      score: asNumber(r.score),
      text: typeof r.text === "string" ? r.text : "",
      documentId,
      chunkId,
    };
  });
}

export function parseCitations(result: unknown): Citation[] {
  const sentinel = parseSentinelSources(result);
  if (sentinel !== null) return sentinel;

  let rows: unknown[] | null = null;
  if (Array.isArray(result)) {
    rows = result;
  } else if (typeof result === "string") {
    const trimmed = result.trim();
    if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) rows = parsed;
        else if (
          parsed &&
          Array.isArray((parsed as { results?: unknown }).results)
        ) {
          rows = (parsed as { results: unknown[] }).results;
        }
      } catch {
        rows = null;
      }
    }
  } else if (
    result &&
    Array.isArray((result as { results?: unknown }).results)
  ) {
    rows = (result as { results: unknown[] }).results;
  }
  if (!rows) return [];

  const citations: Citation[] = [];
  rows.forEach((row, i) => {
    if (!row || typeof row !== "object") return;
    const r = row as Record<string, unknown>;
    const text =
      typeof r.text === "string"
        ? r.text
        : typeof r.chunk === "string"
          ? r.chunk
          : typeof r.content === "string"
            ? r.content
            : "";
    const filename =
      typeof r.filename === "string"
        ? r.filename
        : typeof r.documentId === "string"
          ? r.documentId
          : `Source ${i + 1}`;
    const chunkId =
      typeof r.chunkId === "string" ? r.chunkId : `${filename}-${i}`;
    citations.push({
      id: chunkId,
      filename,
      page: asNumber(r.page),
      score: asNumber(r.score),
      text,
    });
  });
  return citations;
}

/** One citation per document, its best-scoring passage, in first-seen order. */
export function topCitationPerDocument(
  citations: readonly Citation[],
): Citation[] {
  const byDoc = new Map<string, Citation>();
  for (const c of citations) {
    const key = c.documentId ?? c.filename;
    const prev = byDoc.get(key);
    if (
      !prev ||
      (c.score ?? Number.NEGATIVE_INFINITY) >
        (prev.score ?? Number.NEGATIVE_INFINITY)
    ) {
      byDoc.set(key, c);
    }
  }
  return Array.from(byDoc.values());
}

/** The key a citation's document goes by: its id, else its filename. */
export function documentKey(citation: Citation): string {
  return citation.documentId ?? citation.filename;
}

/** Each document's cited pages, ascending and deduplicated, keyed by documentKey. */
export function citedPagesPerDocument(
  citations: readonly Citation[],
): Map<string, number[]> {
  const pages = new Map<string, Set<number>>();
  for (const c of citations) {
    const key = documentKey(c);
    const set = pages.get(key) ?? new Set<number>();
    if (c.page != null) {
      set.add(c.page);
    }
    pages.set(key, set);
  }
  return new Map(
    Array.from(pages, ([key, set]) => [key, [...set].sort((a, b) => a - b)]),
  );
}

/** An empty chunk would render as a blank card, so it is not a passage. */
export function isPassage(citation: Citation): boolean {
  return citation.text.trim().length > 0;
}

/** rank: 1-based among scored passages, across a merged thread + project search; null for a whole-document read. */
export interface Passage {
  citation: Citation;
  rank: number | null;
}

export interface DocumentPassages {
  key: string;
  filename: string;
  passages: Passage[];
}

/**
 * Documents in chip order. Ranks skip unscored passages, so ranked ones after a whole-document read start at #1.
 * `limit` keeps the first passages in retrieval order, not per document.
 */
export function passagesPerDocument(
  citations: readonly Citation[],
  limit = Number.POSITIVE_INFINITY,
): DocumentPassages[] {
  const groups = new Map<string, DocumentPassages>();
  let ranked = 0;
  let kept = 0;
  for (const citation of citations) {
    if (!isPassage(citation)) continue;
    if (kept++ >= limit) break;
    const key = documentKey(citation);
    let group = groups.get(key);
    if (!group) {
      group = { key, filename: citation.filename, passages: [] };
      groups.set(key, group);
    }
    group.passages.push({
      citation,
      rank: citation.score != null ? ++ranked : null,
    });
  }
  return Array.from(groups.values());
}

/** Only PDFs drop `#` markers (added by their Markdown conversion); code and config keep their `#` comments. */
export function passageDisplayText(text: string, filename = ""): string {
  const folded = text.trim().replace(/\n(?:[ \t]*\n)+/g, "\n");
  return /\.pdf$/i.test(filename)
    ? folded.replace(/^#{1,6}[ \t]+/gm, "")
    : folded;
}

// About four lines of text-xs in the desktop chat column, so shorter passages are not clamped for nothing.
const PASSAGE_CLAMP_CHARS = 450;
const PASSAGE_CLAMP_LINES = 4;

export function passageNeedsClamp(text: string): boolean {
  return (
    text.length > PASSAGE_CLAMP_CHARS ||
    text.split("\n").length > PASSAGE_CLAMP_LINES
  );
}

/** "p.3", "p.1–5" or "pp.1–3, 7": runs of consecutive pages collapse into ranges. */
export function pageLabel(pages: readonly number[]): string {
  const runs: string[] = [];
  for (let i = 0; i < pages.length; i++) {
    let end = i;
    while (
      end + 1 < pages.length &&
      pages[end + 1] === Number(pages[end]) + 1
    ) {
      end++;
    }
    runs.push(end === i ? `${pages[i]}` : `${pages[i]}–${pages[end]}`);
    i = end;
  }
  return `${runs.length > 1 ? "pp." : "p."}${runs.join(", ")}`;
}

/** A badge's label: the filename, then the pages it stands for, else the citation's own page. */
export function citationLabel(
  citation: Citation,
  pages?: readonly number[],
): string {
  if (pages && pages.length > 0) {
    return `${citation.filename} · ${pageLabel(pages)}`;
  }
  return citation.page != null
    ? `${citation.filename} · p.${citation.page}`
    : citation.filename;
}
