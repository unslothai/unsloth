// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useMessage } from "@assistant-ui/react";
import type { FC } from "react";

import { type Citation, parseCitations } from "./citation-utils";
import { CitationBadge } from "./tool-ui-knowledge-base";

/**
 * Footnote-style "Sources" row for RAG document citations, shown at the bottom of
 * an assistant message (mirrors the web-citation SourcesGroup). It reads the
 * search_knowledge_base tool-call parts off the message, parses their citations,
 * dedupes to one pill per document (keeping the best-scoring chunk as the link
 * target), and reuses CitationBadge so click-to-preview and the hovercard match
 * the rest of the app.
 */
export const RagSourcesGroup: FC = () => {
  const message = useMessage();

  const all: Citation[] = [];
  for (const part of message.content ?? []) {
    if (part.type === "tool-call" && part.toolName === "search_knowledge_base") {
      all.push(...parseCitations(part.result));
    }
  }

  // One pill per document, in first-seen order; updating a key's value keeps its
  // position, so we keep the highest-scoring chunk without reordering.
  const byDoc = new Map<string, Citation>();
  for (const c of all) {
    const key = c.documentId ?? c.filename;
    const prev = byDoc.get(key);
    if (!prev || (c.score ?? -Infinity) > (prev.score ?? -Infinity)) {
      byDoc.set(key, c);
    }
  }
  const sources = Array.from(byDoc.values());
  if (sources.length === 0) return null;

  return (
    <div className="mt-2 mb-3">
      <div className="mb-1 text-xs font-medium text-muted-foreground">Sources</div>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((citation, i) => (
          <CitationBadge key={citation.id} citation={citation} index={i} />
        ))}
      </div>
    </div>
  );
};
