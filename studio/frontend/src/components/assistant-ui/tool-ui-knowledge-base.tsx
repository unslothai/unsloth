// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { FileTextIcon, LibraryBigIcon, LoaderIcon } from "lucide-react";
import { memo, useEffect, useMemo, useState } from "react";
import { Badge } from "./badge";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { useDocumentPreviewStore } from "@/features/rag/components/preview-store";

import { type Citation, parseCitations } from "./citation-utils";

/** Citation badge: filename + page, chunk text on hover; clicking opens the source viewer when a documentId is present. */
export function CitationBadge({
  citation,
  index,
}: {
  citation: Citation;
  index: number;
}) {
  const openPreview = useDocumentPreviewStore((s) => s.openPreview);
  const clickable = Boolean(citation.documentId);
  const label =
    citation.page != null
      ? `${citation.filename} · p.${citation.page}`
      : citation.filename;

  const open = () => {
    if (!citation.documentId) return;
    openPreview({
      documentId: citation.documentId,
      chunkId: citation.chunkId,
      filename: citation.filename,
      page: citation.page,
    });
  };

  const badge = (
    <Badge
      variant="outline"
      size="sm"
      className={`rounded-full inline-flex items-center gap-1.5 max-w-[15rem] ${
        clickable
          ? "cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors"
          : "cursor-default"
      }`}
    >
      <span className="tabular-nums text-muted-foreground">{index + 1}</span>
      <FileTextIcon className="size-3 shrink-0" />
      <span className="truncate">{label}</span>
    </Badge>
  );

  return (
    <HoverCard openDelay={0} closeDelay={150}>
      <HoverCardTrigger asChild>
        {clickable ? (
          <button type="button" onClick={open} className="inline-block">
            {badge}
          </button>
        ) : (
          <span className="inline-block">{badge}</span>
        )}
      </HoverCardTrigger>
      <HoverCardContent
        side="top"
        align="start"
        className="!w-80 !p-3 !rounded-2xl"
      >
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 text-xs font-semibold">
            <FileTextIcon className="size-3.5 shrink-0" />
            <span className="truncate">{citation.filename}</span>
            {citation.page != null && (
              <span className="text-muted-foreground">· page {citation.page}</span>
            )}
          </div>
          {citation.text && (
            <p className="max-h-48 overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed text-muted-foreground">
              {citation.text}
            </p>
          )}
          <div className="flex items-center justify-between gap-2">
            {citation.score != null && (
              <span className="text-[10px] tabular-nums text-muted-foreground/70">
                score {citation.score.toFixed(3)}
              </span>
            )}
            {clickable && (
              <button
                type="button"
                onClick={open}
                className="ml-auto rounded-md px-1.5 py-0.5 text-[10px] font-medium text-primary transition-colors hover:bg-primary/10"
              >
                View source
              </button>
            )}
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

const KnowledgeBaseToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const query = (args as { query?: string })?.query ?? "";
  const isRunning = status?.type === "running";
  const citations = useMemo(() => parseCitations(result), [result]);
  // Citations themselves now render as a "Sources" list at the bottom of the
  // message (RagSourcesGroup); the block keeps only a one-line summary.
  const docCount = useMemo(
    () => new Set(citations.map((c) => c.documentId ?? c.filename)).size,
    [citations],
  );

  // Collapse once the model starts answering.
  const hasText = useAuiState(({ message }) =>
    message.content.some(
      (p) =>
        p.type === "text" &&
        "text" in p &&
        (p as { text: string }).text.length > 0,
    ),
  );
  const [open, setOpen] = useState(isRunning);
  useEffect(() => {
    if (isRunning) setOpen(true);
    else if (hasText) setOpen(false);
  }, [isRunning, hasText]);

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={query ? `Searched documents for "${query}"` : "Knowledge search"}
        status={status}
        icon={LibraryBigIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            <span>
              {query ? (
                <>Searching documents for &ldquo;{query}&rdquo;&hellip;</>
              ) : (
                <>Searching documents&hellip;</>
              )}
            </span>
          </div>
        ) : citations.length > 0 ? (
          <div className="text-sm text-muted-foreground">
            Retrieved {citations.length} passage
            {citations.length === 1 ? "" : "s"} from {docCount} document
            {docCount === 1 ? "" : "s"}. See Sources below.
          </div>
        ) : result ? (
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
            {typeof result === "string"
              ? result
              : JSON.stringify(result, null, 2)}
          </pre>
        ) : (
          <div className="text-sm text-muted-foreground">No matching passages.</div>
        )}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const KnowledgeBaseToolUI = memo(
  KnowledgeBaseToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
KnowledgeBaseToolUI.displayName = "KnowledgeBaseToolUI";
