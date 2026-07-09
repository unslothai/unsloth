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
        else if (parsed && Array.isArray((parsed as { results?: unknown }).results)) {
          rows = (parsed as { results: unknown[] }).results;
        }
      } catch {
        rows = null;
      }
    }
  } else if (result && Array.isArray((result as { results?: unknown }).results)) {
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
