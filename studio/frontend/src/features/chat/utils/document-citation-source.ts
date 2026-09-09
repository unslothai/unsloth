// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Accept navigable http(s) source URLs. */
export function isSafeNavigableSourceUrl(raw: unknown): string {
  if (typeof raw !== "string") return "";
  const value = raw.trim();
  if (!value || /[\r\n]/.test(value)) return "";
  try {
    const parsed = new URL(value);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return value;
    }
  } catch {}
  return "";
}

/** Convert an Anthropic document citation dict into a Sources-panel source. */
export function documentCitationToSource(
  cit: Record<string, unknown>,
  fallbackIdx: number,
): {
  type: "source";
  sourceType: "url";
  id: string;
  url: string;
  title: string;
  metadata?: { description: string };
} | null {
  const source = typeof cit.source === "string" && cit.source ? cit.source : "";
  const docTitle =
    (typeof cit.document_title === "string" && cit.document_title) ||
    (typeof cit.title === "string" && cit.title) ||
    "";
  const docIndex =
    typeof cit.document_index === "number" ? cit.document_index : undefined;
  // Use document anchors for non-web sources.
  const url =
    isSafeNavigableSourceUrl(source) ||
    `#anthropic-doc-${docIndex ?? fallbackIdx}`;
  const title = docTitle || source || `Document ${fallbackIdx + 1}`;
  const cited = typeof cit.cited_text === "string" ? cit.cited_text.trim() : "";
  const description = cited.length > 240 ? `${cited.slice(0, 240)}...` : cited;
  // Keep separate footnotes for distinct citation positions.
  const citationType = typeof cit.type === "string" ? String(cit.type) : "";
  const positionParts = [
    cit.search_result_index,
    cit.start_char_index,
    cit.end_char_index,
    cit.start_page_number,
    cit.end_page_number,
    cit.start_block_index,
    cit.end_block_index,
  ]
    .filter((v) => typeof v === "number")
    .map((v) => String(v))
    .join(":");
  const idAnchor = positionParts
    ? `${citationType}:${positionParts}`
    : `${citationType}:${fallbackIdx}`;
  const id = `${url}#${idAnchor}`;
  return {
    type: "source" as const,
    sourceType: "url" as const,
    id,
    url,
    title,
    ...(description ? { metadata: { description } } : {}),
  };
}
