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
import { FileGlyph } from "@/lib/file-icon";
import { stringifyToolResult } from "@/lib/strip-ansi";
import { type ComponentProps, memo, useMemo } from "react";
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
import { LocalFileDialog } from "@/components/assistant-ui/attachment-document-dialog";
import { MAX_DOCUMENT_PREVIEW_BYTES } from "@/components/file-viewer";
import {
  asViewerFile,
  fetchSourceFile,
  isViewableSource,
} from "@/features/rag/components/source-file";

import {
  type Citation,
  parseCitations,
  topCitationPerDocument,
} from "./citation-utils";
import { ScrollPane } from "./scroll-pane";

export function CitationBadge({
  citation,
  index,
}: {
  citation: Citation;
  index: number;
}) {
  const label =
    citation.page != null
      ? `${citation.filename} · p.${citation.page}`
      : citation.filename;
  const { documentId, filename } = citation;
  const clickable = Boolean(documentId) && isViewableSource(filename);

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
      <span className="truncate">{label}</span>
    </Badge>
  );

  if (!clickable || !documentId) {
    return <span className="inline-block">{badge}</span>;
  }
  // The attachment viewer, as the chat with files cards open their files.
  const load = async (): Promise<Blob> => {
    const file = await fetchSourceFile(
      { id: documentId, filename },
      MAX_DOCUMENT_PREVIEW_BYTES,
    );
    if (!file) {
      throw new Error(`Couldn't preview ${filename}`);
    }
    return asViewerFile(file);
  };
  return (
    <LocalFileDialog name={filename.split("/").pop() ?? filename} load={load}>
      <button type="button" className="inline-block">
        {badge}
      </button>
    </LocalFileDialog>
  );
}

// The app's Library icon, as in the sidebar.
const LibraryGlyph = (
  props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">,
) => <HugeiconsIcon icon={LibrariesIcon} strokeWidth={1.75} {...props} />;

// Longer queries get a scroll box rather than a line of text.
const QUERY_INLINE_MAX = 80;

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

  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
  );
  // Ask permission gates every local tool call, and what is being approved
  // lives inside the content while Allow/Deny render outside it.
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
                  {documents.map((citation, i) => (
                    <CitationBadge
                      key={citation.id}
                      citation={citation}
                      index={i}
                    />
                  ))}
                </div>
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
