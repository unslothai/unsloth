// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { FileTextIcon, LoaderIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

export interface ParsedChunk {
  id: string;
  source: string;
  page?: string;
  score?: string;
  denseScore?: string;
  chunkIndex?: string;
  tokens?: string;
  kind?: string;
  text: string;
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
        score: attrs.score,
        denseScore: attrs.dense_score,
        chunkIndex: attrs.chunk_index,
        tokens: attrs.tokens,
        kind: attrs.kind,
        text,
      });
    }
    match = CHUNK_RE.exec(raw);
  }
  CHUNK_RE.lastIndex = 0;
  ATTR_RE.lastIndex = 0;
  return out;
}

function ChunkCard({ chunk }: { chunk: ParsedChunk }) {
  const meta: string[] = [];
  if (chunk.page) meta.push(`page ${chunk.page}`);
  if (chunk.score) meta.push(`score ${chunk.score}`);
  if (chunk.denseScore && chunk.denseScore !== chunk.score) {
    meta.push(`dense ${chunk.denseScore}`);
  }
  if (chunk.tokens) meta.push(`${chunk.tokens} tok`);
  if (chunk.chunkIndex) meta.push(`#${chunk.chunkIndex}`);
  if (chunk.kind && chunk.kind !== "text") meta.push(chunk.kind);

  return (
    <div
      data-slot="rag-chunk-card"
      className="rounded-md border border-foreground/10 bg-muted/40 p-2.5 text-xs"
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="rounded bg-foreground/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold">
            [{chunk.id}]
          </span>
          <FileTextIcon className="size-3 shrink-0 text-muted-foreground" />
          <span className="truncate font-medium" title={chunk.source}>
            {chunk.source}
          </span>
        </div>
        {meta.length > 0 ? (
          <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">
            {meta.join(" · ")}
          </span>
        ) : null}
      </div>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-relaxed text-foreground/80">
        {chunk.text}
      </pre>
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
