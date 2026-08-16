"use client";

import { useMessage } from "@assistant-ui/react";
import type { FC } from "react";

import {
  platformChatCitations as citationsFromPlatformReference,
  mapPlatformChatReference,
} from "@/integrations/platform-backend";
import { type Citation, parseCitations } from "./citation-utils";
import { CitationBadge } from "./tool-ui-knowledge-base";

export const DocumentSourcesGroup: FC<{ sources: Citation[] }> = ({
  sources: all,
}) => {
  // Map updates keep first-seen order, so dedup to best-scoring chunk per doc.
  const byDoc = new Map<string, Citation>();
  for (const c of all) {
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
  const sources = Array.from(byDoc.values());
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 mb-3">
      <div className="mb-1 text-xs font-medium text-muted-foreground">
        Document Sources
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((citation, i) => (
          <CitationBadge key={citation.id} citation={citation} index={i} />
        ))}
      </div>
    </div>
  );
};

export const RagSourcesGroup: FC = () => {
  const message = useMessage();

  const sources: Citation[] = [];
  const metadata = message.metadata as
    | {
        custom?: {
          platformCitations?: unknown;
          platformReference?: unknown;
        };
        platformCitations?: unknown;
        platformReference?: unknown;
      }
    | undefined;
  const platformCitations =
    metadata?.custom?.platformCitations ?? metadata?.platformCitations;
  if (Array.isArray(platformCitations)) {
    sources.push(
      ...parseCitations(platformCitations).map((citation) => ({
        ...citation,
        source: "platform" as const,
      })),
    );
  } else {
    const platformReference =
      metadata?.custom?.platformReference ?? metadata?.platformReference;
    if (platformReference !== undefined && platformReference !== null) {
      sources.push(
        ...parseCitations(
          citationsFromPlatformReference(
            mapPlatformChatReference(platformReference),
          ),
        ),
      );
    }
  }
  for (const part of message.content ?? []) {
    if (
      part.type === "tool-call" &&
      part.toolName === "search_knowledge_base"
    ) {
      sources.push(...parseCitations(part.result));
    }
  }
  return <DocumentSourcesGroup sources={sources} />;
};
