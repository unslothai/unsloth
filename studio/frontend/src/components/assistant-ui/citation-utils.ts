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

/** Whether a citation has text to show as a passage; an empty chunk would render as a blank card. */
export function isPassage(citation: Citation): boolean {
  return citation.text.trim().length > 0;
}

/** A retrieved passage and its 1-based rank among the ranked passages; null when unscored (a whole-document read). */
export interface Passage {
  citation: Citation;
  rank: number | null;
}

/** One document's passages, in retrieval order. */
export interface DocumentPassages {
  key: string;
  filename: string;
  passages: Passage[];
}

/**
 * Passages grouped by document, documents in first-seen order (as their chips). Ranks count only
 * scored passages, so ranked project passages merged after a whole-document read still start at #1.
 */
export function passagesPerDocument(
  citations: readonly Citation[],
): DocumentPassages[] {
  const groups = new Map<string, DocumentPassages>();
  let ranked = 0;
  for (const citation of citations) {
    if (!isPassage(citation)) continue;
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

/** The first `limit` passages across the groups, in order; groups left empty are dropped. */
export function firstPassages(
  groups: readonly DocumentPassages[],
  limit: number,
): DocumentPassages[] {
  const shown: DocumentPassages[] = [];
  let room = limit;
  for (const group of groups) {
    if (room <= 0) break;
    const passages = group.passages.slice(0, room);
    room -= passages.length;
    if (passages.length > 0) shown.push({ ...group, passages });
  }
  return shown;
}

/**
 * A passage as shown: trimmed, without the Markdown heading markers PDF and Office parsing adds,
 * and with blank lines folded so a clamped preview spends its lines on text.
 */
export function passageDisplayText(text: string): string {
  return text
    .trim()
    .replace(/^#{1,6}[ \t]+/gm, "")
    .replace(/\n[ \t]*\n\s*/g, "\n");
}

const PASSAGE_CLAMP_CHARS = 280;
const PASSAGE_CLAMP_LINES = 4;

/** Whether a passage is long enough to start clamped behind "Show more". */
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
