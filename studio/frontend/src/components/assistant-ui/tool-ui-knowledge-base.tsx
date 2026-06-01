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

interface Citation {
  /** Stable key; chunkId when present, else a positional fallback. */
  id: string;
  filename: string;
  page?: number | null;
  score?: number | null;
  text: string;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * The backend's search_knowledge_base tool result is provider-shaped. We
 * accept either a JSON array of result objects, a `{results:[...]}`
 * envelope, or a plain string (rendered as-is). Anything we can't map to
 * a {filename,text} pair falls through to the raw-text branch.
 */
function parseCitations(result: unknown): Citation[] {
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

/** Citation badge: filename + page, with the chunk text in a hover popover. */
function CitationBadge({ citation, index }: { citation: Citation; index: number }) {
  const label =
    citation.page != null
      ? `${citation.filename} · p.${citation.page}`
      : citation.filename;
  return (
    <HoverCard openDelay={0} closeDelay={0}>
      <HoverCardTrigger asChild>
        <span className="inline-block">
          <Badge
            variant="outline"
            size="sm"
            className="cursor-default rounded-full inline-flex items-center gap-1.5 max-w-[15rem]"
          >
            <span className="tabular-nums text-muted-foreground">
              {index + 1}
            </span>
            <FileTextIcon className="size-3 shrink-0" />
            <span className="truncate">{label}</span>
          </Badge>
        </span>
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
          {citation.score != null && (
            <p className="text-[10px] tabular-nums text-muted-foreground/70">
              score {citation.score.toFixed(3)}
            </p>
          )}
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

  // Collapse once the model starts answering, mirroring WebSearchToolUI.
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
          <div className="flex flex-wrap gap-1.5">
            {citations.map((citation, i) => (
              <CitationBadge key={citation.id} citation={citation} index={i} />
            ))}
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
