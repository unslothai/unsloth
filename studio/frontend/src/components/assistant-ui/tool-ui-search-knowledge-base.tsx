// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { authFetch } from "@/features/auth";
import { usePreviewStore } from "@/features/rag/stores/preview-store";
import { cn } from "@/lib/utils";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { FileTextIcon, ImageIcon, LoaderIcon } from "lucide-react";
import { memo, useCallback, useEffect, useState } from "react";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

export interface ParsedChunk {
  /** Visible citation id the model uses inside `[N]` references. Display
   *  only; never sent to the backend as a chunk_id. */
  id: string;
  source: string;
  page?: string;
  chunkIndex?: string;
  tokens?: string;
  sourcePageIndex?: string;
  pageCharStart?: string;
  pageCharEnd?: string;
  lineStart?: string;
  lineEnd?: string;
  kind?: string;
  imageUrl?: string;
  text: string;
  /** Durable `rag_documents.id`. Carries through when the tool XML
   *  includes `document_id="..."`. Absent on legacy tool output. */
  documentId?: string;
  /** Durable `rag_chunks.id`. Carries through when the tool XML
   *  includes `chunk_id="..."`. Absent on legacy tool output. The
   *  preview routing value sent as `?chunk_id=` to `/preview-target`;
   *  never the same as the visible `id`. */
  backendChunkId?: string;
}

const ATTR_RE = /(\w+)="([^"]*)"/g;
const CHUNK_RE = /<chunk\s+([^>]+)>\s*([\s\S]*?)\s*<\/chunk>/g;

function decodeXml(value: string): string {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

export function parseChunks(raw: string): ParsedChunk[] {
  if (!raw) return [];
  const out: ParsedChunk[] = [];
  let match: RegExpExecArray | null = CHUNK_RE.exec(raw);
  while (match !== null) {
    const attrBlob = match[1];
    const text = match[2];
    const attrs: Record<string, string> = {};
    let attrMatch: RegExpExecArray | null = ATTR_RE.exec(attrBlob);
    while (attrMatch !== null) {
      attrs[attrMatch[1]] = decodeXml(attrMatch[2]);
      attrMatch = ATTR_RE.exec(attrBlob);
    }
    if (attrs.id) {
      out.push({
        id: attrs.id,
        source: attrs.source ?? "unknown",
        page: attrs.page,
        chunkIndex: attrs.chunk_index,
        tokens: attrs.tokens,
        sourcePageIndex: attrs.source_page_index,
        pageCharStart: attrs.page_char_start,
        pageCharEnd: attrs.page_char_end,
        lineStart: attrs.line_start,
        lineEnd: attrs.line_end,
        kind: attrs.kind,
        imageUrl: attrs.image_url,
        text,
        // Durable backend ids (legacy XML omits both → preview gated off).
        ...(attrs.document_id ? { documentId: attrs.document_id } : {}),
        ...(attrs.chunk_id ? { backendChunkId: attrs.chunk_id } : {}),
      });
    }
    match = CHUNK_RE.exec(raw);
  }
  CHUNK_RE.lastIndex = 0;
  ATTR_RE.lastIndex = 0;
  return out;
}

/** Fetch a backend image via the bearer-authed `authFetch`, expose it
 *  as a blob URL for `<img src>`. Cleans up the object URL on unmount. */
function useAuthedImageUrl(path: string | undefined): string | undefined {
  const [url, setUrl] = useState<string | undefined>(undefined);
  useEffect(() => {
    if (!path) {
      setUrl(undefined);
      return;
    }
    let cancelled = false;
    let objectUrl: string | undefined;
    authFetch(path)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`image fetch ${response.status}`);
        }
        return response.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setUrl(undefined);
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);
  return url;
}

function ChunkImage({ url, alt }: { url: string; alt: string }) {
  const blobUrl = useAuthedImageUrl(url);
  if (!blobUrl) {
    return (
      <div className="mb-2 flex h-32 items-center justify-center rounded-md bg-muted/60 text-[10px] text-muted-foreground">
        <ImageIcon className="mr-1.5 size-3" />
        Loading image…
      </div>
    );
  }
  return (
    <img
      src={blobUrl}
      alt={alt}
      className="mb-2 max-h-64 w-full rounded-md object-contain"
    />
  );
}

function ChunkCard({ chunk }: { chunk: ParsedChunk }) {
  const openPreview = usePreviewStore((s) => s.open);
  const meta: string[] = [];
  if (chunk.page) meta.push(`page ${chunk.page}`);
  if (chunk.tokens) meta.push(`${chunk.tokens} tok`);
  if (chunk.chunkIndex) meta.push(`#${chunk.chunkIndex}`);
  if (chunk.kind && chunk.kind !== "text") meta.push(chunk.kind);

  const documentId = chunk.documentId;
  const backendChunkId = chunk.backendChunkId;
  const isPreviewable = Boolean(documentId && backendChunkId);
  const handleOpenPreview = useCallback(() => {
    if (!(documentId && backendChunkId)) {
      return;
    }
    Promise.resolve(
      openPreview({
        documentId,
        backendChunkId,
      }),
    ).catch(() => undefined);
  }, [backendChunkId, documentId, openPreview]);

  const sourceLabel = (
    <>
      <span className="rounded bg-foreground/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold">
        [{chunk.id}]
      </span>
      {chunk.kind === "image" ? (
        <ImageIcon className="size-3 shrink-0 text-muted-foreground" />
      ) : (
        <FileTextIcon className="size-3 shrink-0 text-muted-foreground" />
      )}
      <span className="truncate font-medium" title={chunk.source}>
        {chunk.source}
      </span>
    </>
  );

  return (
    <div
      data-slot="rag-chunk-card"
      className="rounded-md border border-foreground/10 bg-muted/40 p-2.5 text-xs"
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        {isPreviewable ? (
          <button
            type="button"
            className={cn(
              "flex min-w-0 cursor-pointer items-center gap-1.5 rounded-sm text-left outline-none transition-colors",
              "hover:text-primary focus-visible:ring-[3px] focus-visible:ring-ring/50",
            )}
            onClick={handleOpenPreview}
            aria-label={`Open preview of ${chunk.source}`}
            title="Open preview"
          >
            {sourceLabel}
          </button>
        ) : (
          <div className="flex min-w-0 items-center gap-1.5">{sourceLabel}</div>
        )}
        {meta.length > 0 ? (
          <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">
            {meta.join(" · ")}
          </span>
        ) : null}
      </div>
      {chunk.kind === "image" && chunk.imageUrl ? (
        <ChunkImage url={chunk.imageUrl} alt={chunk.source} />
      ) : null}
      {chunk.text ? (
        <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-relaxed text-foreground/80">
          {chunk.text}
        </pre>
      ) : null}
    </div>
  );
}

const SearchKnowledgeBaseToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const query = (args as { query?: string })?.query ?? "";
  const isRunning = status?.type === "running";
  const resultText = typeof result === "string" ? result : "";
  const chunks = parseChunks(resultText);
  const isErrorOrEmpty =
    !isRunning && chunks.length === 0 && resultText.length > 0;

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
    if (isRunning) {
      setOpen(true);
    } else if (hasText) {
      setOpen(false);
    }
  }, [isRunning, hasText]);

  const triggerLabel = query
    ? `Searched docs: "${query}"`
    : "Search knowledge base";

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={triggerLabel}
        status={status}
        icon={FileTextIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            <span>
              {query ? (
                <>Retrieving for &ldquo;{query}&rdquo;&hellip;</>
              ) : (
                <>Retrieving&hellip;</>
              )}
            </span>
          </div>
        ) : chunks.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            <div className="text-[10px] text-muted-foreground">
              {chunks.length} chunk{chunks.length === 1 ? "" : "s"} retrieved
            </div>
            {chunks.map((chunk) => (
              <ChunkCard key={chunk.id} chunk={chunk} />
            ))}
          </div>
        ) : isErrorOrEmpty ? (
          <pre
            className={cn(
              "max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs",
            )}
          >
            {resultText}
          </pre>
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const SearchKnowledgeBaseToolUI = memo(
  SearchKnowledgeBaseToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
SearchKnowledgeBaseToolUI.displayName = "SearchKnowledgeBaseToolUI";
