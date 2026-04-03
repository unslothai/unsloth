// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { type ToolCallMessagePartComponent, useAuiState } from "@assistant-ui/react";
import { GlobeIcon, LoaderIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
import { Source, SourceIcon, SourceTitle } from "./sources";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

interface ParsedSource {
  title: string;
  url: string;
  snippet: string;
}

const RE_BLOCK_SEP = /\n---\n/;
const RE_TITLE = /Title:\s*(.+)/;
const RE_URL = /URL:\s*(.+)/;
const RE_SNIPPET = /Snippet:\s*(.+)/s;

/** Parse the backend's "Title: ...\nURL: ...\nSnippet: ...\n---" format into structured sources. */
function parseSearchResults(raw: string): ParsedSource[] {
  if (!raw) {
    return [];
  }
  const blocks = raw.split(RE_BLOCK_SEP).filter(Boolean);
  const sources: ParsedSource[] = [];
  for (const block of blocks) {
    const titleMatch = block.match(RE_TITLE);
    const urlMatch = block.match(RE_URL);
    const snippetMatch = block.match(RE_SNIPPET);
    if (titleMatch && urlMatch) {
      sources.push({
        title: titleMatch[1].trim(),
        url: urlMatch[1].trim(),
        snippet: snippetMatch?.[1]?.trim() ?? "",
      });
    }
  }
  return sources;
}

const WebSearchToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const query = (args as { query?: string })?.query ?? "";
  const url = ((args as { url?: string })?.url ?? "").trim();
  const isUrlFetch = !!url;
  const displayDomain = (() => {
    if (!url) return "";
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
      return parsed.hostname.replace(/^www\./, "");
    } catch {
      return "";
    }
  })();
  const isRunning = status?.type === "running";
  const sources = result
    ? parseSearchResults(
        typeof result === "string" ? result : JSON.stringify(result),
      )
    : [];

  // Collapse when LLM starts generating text after the tool call
  const hasText = useAuiState(({ message }) =>
    message.content.some((p) => p.type === "text" && "text" in p && (p as { text: string }).text.length > 0),
  );
  const [open, setOpen] = useState(isRunning);
  useEffect(() => {
    if (isRunning) {
      setOpen(true);
    } else if (hasText) {
      setOpen(false);
    }
  }, [isRunning, hasText]);

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={
          isUrlFetch
            ? displayDomain ? `Read ${displayDomain}` : "Read page"
            : query
              ? `Searched "${query}"`
              : "Web Search"
        }
        status={status}
        icon={GlobeIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            <span>
              {isUrlFetch
                ? <>Reading {displayDomain || "page"}&hellip;</>
                : <>Searching for &ldquo;{query}&rdquo;&hellip;</>
              }
            </span>
          </div>
        ) : sources.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {sources.map((source, i) => (
              <Source
                key={`${source.url}-${i}`}
                href={source.url}
                variant="outline"
                size="sm"
                className="inline-flex items-center gap-1.5"
              >
                <SourceIcon url={source.url} size={3} />
                <SourceTitle>{source.title}</SourceTitle>
              </Source>
            ))}
          </div>
        ) : result ? (
          <div>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
              {typeof result === "string"
                ? result
                : JSON.stringify(result, null, 2)}
            </pre>
          </div>
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const WebSearchToolUI = memo(
  WebSearchToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
WebSearchToolUI.displayName = "WebSearchToolUI";
