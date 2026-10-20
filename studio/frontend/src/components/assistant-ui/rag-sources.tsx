// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { FC } from "react";

import { type Citation, topCitationPerDocument } from "./citation-utils";
import { CitationBadge } from "./tool-ui-knowledge-base";

export const DocumentSourcesGroup: FC<{
  sources: Citation[];
  label?: string;
}> = ({ sources: all, label = "Document Sources" }) => {
  const sources = topCitationPerDocument(all);
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 mb-3">
      <div className="mb-1 text-xs font-medium text-muted-foreground">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((citation, i) => (
          <CitationBadge key={citation.id} citation={citation} index={i} />
        ))}
      </div>
    </div>
  );
};
