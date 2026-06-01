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

/** Sentinel before the citation source-map JSON. */
const RAG_SOURCES_SENTINEL = "__RAG_SOURCES__:";

interface Citation {
  /** Stable key; chunkId if present, else positional fallback. */
  id: string;
  filename: string;
  page?: number | null;
  score?: number | null;
  text: string;
  /** Set when the citation can open its source in the viewer. */
  documentId?: string | null;
  chunkId?: string | null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Parse the source-map after the sentinel (carries documentId/chunkId for the
 * viewer). Null if absent, so callers fall back to generic JSON shapes.
 */
function parseSentinelSources(result: unknown): Citation[] | null {
  if (typeof result !== "string") return null;
  const idx = result.indexOf(RAG_SOURCES_SENTINEL);
  if (idx < 0) return null;
  const payload = result.slice(idx + RAG_SOURCES_SENTINEL.length).trim();
  let rows: unknown;
  try {
    rows = JSON.parse(payload);
  } catch {
    return [];
  }
  if (!Array.isArray(rows)) return [];
  return rows.map((row, i) => {
    const r = (row ?? {}) as Record<string, unknown>;
    const documentId = typeof r.documentId === "string" ? r.documentId : null;
    const chunkId = typeof r.chunkId === "string" ? r.chunkId : null;
    const filename =
      typeof r.filename === "string" ? r.filename : `Source ${i + 1}`;
    return {
      id: chunkId ?? `${filename}-${i}`,
      filename,
      page: asNumber(r.page),
      score: asNumber(r.score),
      text: typeof r.text === "string" ? r.text : "",
      documentId,
      chunkId,
    };
  });
}

/**
 * Normalize a provider-shaped tool result (JSON array, `{results:[...]}`, or
 * string) to Citations; unmappable input falls through to the raw-text branch.
 */
function parseCitations(result: unknown): Citation[] {
  const sentinel = parseSentinelSources(result);
  if (sentinel !== null) return sentinel;

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

/**
 * Citation badge: filename + page, chunk text on hover. With a documentId,
 * clicking opens the source in the shared viewer (PDFs region-highlighted).
 */
function CitationBadge({ citation, index }: { citation: Citation; index: number }) {
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
    <HoverCard openDelay={0} closeDelay={0}>
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
              <span className="ml-auto text-[10px] font-medium text-primary">
                Click to view source
              </span>
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

  // Collapse once the model starts answering, like WebSearchToolUI.
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
