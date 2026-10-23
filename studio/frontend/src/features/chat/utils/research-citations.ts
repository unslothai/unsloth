// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Citation } from "@/components/assistant-ui/citation-utils";
import type { ResearchDocumentSource } from "../types/research";

export function researchCitation(
  source: ResearchDocumentSource,
  index: number,
): Citation {
  return {
    id: source.chunkId ?? String(source.id ?? index),
    filename: source.filename,
    page: source.page,
    score: source.score,
    text: source.snippet ?? "",
    documentId: source.kind === "mcp" ? undefined : source.documentId,
    chunkId: source.chunkId,
  };
}
