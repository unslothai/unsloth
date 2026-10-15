// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { partsHaveNonEmptyText } from "@/components/assistant-ui/message-derived";
import { LibrariesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Spinner } from "@/components/ui/spinner";

import { useToolAwaitingApproval } from "@/features/chat";
import { RagDocumentTrigger } from "@/features/rag/components/rag-document-trigger";
import { FileGlyph } from "@/lib/file-icon";
import { stringifyToolResult } from "@/lib/strip-ansi";
import { type ComponentProps, memo, useMemo, useState } from "react";
import { Badge } from "./badge";
import {
  isToolCallRunning,
  knowledgeBaseToolName,
  toolArgText,
} from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";

import {
  type Citation,
  parseCitations,
  citationLabel,
  citedPagesPerDocument,
  documentKey,
  topCitationPerDocument,
} from "./citation-utils";
import { ScrollPane } from "./scroll-pane";

export function CitationBadge({
  citation,
  index,
  pages,
}: {
  citation: Citation;
  index: number;
  /** Every page cited from this document, when the badge stands for all of them. */
  pages?: readonly number[];
}) {
  const label = citationLabel(citation, pages);
  const { documentId, filename } = citation;
  const clickable = Boolean(documentId);

  const badge = (
    <Badge
      variant="outline"
      size="sm"
      className={`rounded-full inline-flex items-center gap-1.5 max-w-[calc(15rem*var(--ui-space-scale,1))] ${
        clickable
          ? "cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors"
          : "cursor-default"
      }`}
    >
      <span className="tabular-nums text-muted-foreground">{index + 1}</span>
      <FileGlyph className="size-3 shrink-0" />
      <span className="truncate" title={label}>
        {label}
      </span>
    </Badge>
  );

  if (!documentId) {
    return <span className="inline-block">{badge}</span>;
  }
  return (
    <RagDocumentTrigger
      documentId={documentId}
      filename={filename}
      page={pages?.[0] ?? citation.page ?? undefined}
    >
      <button type="button" className="inline-block rounded-full">
        {badge}
      </button>
    </RagDocumentTrigger>
  );
}

// The app's Library icon, as in the sidebar.
const LibraryGlyph = (
  props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">,
) => <HugeiconsIcon icon={LibrariesIcon} strokeWidth={1.75} {...props} />;

// Longer queries get a scroll box rather than a line of text.
const QUERY_INLINE_MAX = 80;
// The model picks top_k with no ceiling, so a huge search could cite very many documents.
const DOCUMENTS_SHOWN = 20;

const KnowledgeBaseToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const query = toolArgText((args as { query?: unknown })?.query);
  const isRunning = isToolCallRunning(status);

  const resultText = result == null ? "" : stringifyToolResult(result);
  const citations = useMemo(() => parseCitations(result), [result]);
  // The documents behind the passages, listed in the expanded card.
  const documents = useMemo(
    () => topCitationPerDocument(citations),
    [citations],
  );
  const docCount = documents.length;
  const pagesByDocument = useMemo(
    () => citedPagesPerDocument(citations),
    [citations],
  );
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? documents : documents.slice(0, DOCUMENTS_SHOWN);

  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
  );
  // What is being approved lives inside the content, while Allow/Deny render outside it.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={knowledgeBaseToolName({ isRunning, query })}
        status={status}
        icon={LibraryGlyph}
        singleLine={true}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-3.5" />
            <span>
              {query ? (
                <>Searching documents for &ldquo;{query}&rdquo;&hellip;</>
              ) : (
                <>Searching documents&hellip;</>
              )}
            </span>
          </div>
        ) : (
          // Outcome first, then the full query: the one-line header may clip it in every outcome.
          <div className="flex flex-col gap-2 text-sm text-muted-foreground">
            {citations.length > 0 ? (
              <>
                <div>
                  Retrieved {citations.length} passage
                  {citations.length === 1 ? "" : "s"} from {docCount} document
                  {docCount === 1 ? "" : "s"}.
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {shown.map((citation, i) => (
                    <CitationBadge
                      key={citation.id}
                      citation={citation}
                      index={i}
                      pages={pagesByDocument.get(documentKey(citation))}
                    />
                  ))}
                </div>
                {shown.length < documents.length ? (
                  <button
                    type="button"
                    onClick={() => setShowAll(true)}
                    className="self-start text-xs underline-offset-2 hover:text-foreground hover:underline"
                  >
                    Show {documents.length - shown.length} more
                  </button>
                ) : null}
              </>
            ) : resultText ? (
              <ScrollPane
                className="rounded bg-muted/50 p-2"
                scrollerClassName="max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs"
              >
                {resultText}
              </ScrollPane>
            ) : (
              <div>No matching passages.</div>
            )}
            {query.length > QUERY_INLINE_MAX ? (
              <ScrollPane
                data-slot="knowledge-base-query"
                className="rounded bg-muted/50 p-2"
                scrollerClassName="max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs"
              >
                {query}
              </ScrollPane>
            ) : query ? (
              <div
                data-slot="knowledge-base-query"
                className="break-words text-xs"
              >
                Query: &ldquo;{query}&rdquo;
              </div>
            ) : null}
          </div>
        )}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const KnowledgeBaseToolUI = memo(
  KnowledgeBaseToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
KnowledgeBaseToolUI.displayName = "KnowledgeBaseToolUI";
