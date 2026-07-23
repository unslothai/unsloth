// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { FileTextIcon, LibraryBigIcon } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { memo, useEffect, useMemo, useState } from "react";
import { Badge } from "./badge";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useDocumentPreviewStore } from "@/features/rag/components/preview-store";

import { type Citation, parseCitations } from "./citation-utils";

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
      className={`rounded-full inline-flex items-center gap-1.5 max-w-[240px] ${
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

  return clickable ? (
    <button type="button" onClick={open} className="inline-block">
      {badge}
    </button>
  ) : (
    <span className="inline-block">{badge}</span>
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
  // Citations render in RagSourcesGroup; this block keeps a one-line summary.
  const docCount = useMemo(
    () => new Set(citations.map((c) => c.documentId ?? c.filename)).size,
    [citations],
  );

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
            <Spinner className="size-3.5" />
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
            {docCount === 1 ? "" : "s"}. See Document Sources below.
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
