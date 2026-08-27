// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { GlobeIcon } from "lucide-react";

import {
  isSearchImagesToolResult,
  useToolAwaitingApproval,
} from "@/features/chat";
import { stringifyToolResult } from "@/lib/strip-ansi";
import { memo } from "react";
import { SearchImageThumb } from "./search-image";
import { Source, SourceIcon, SourceTitle } from "./sources";
import { toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";

interface ParsedSource {
  title: string;
  url: string;
  snippet: string;
}

const RE_BLOCK_SEP = /\n---\n/;
const RE_TITLE = /Title:\s*(.+)/;
const RE_URL = /URL:\s*(.+)/;
const RE_SNIPPET = /Snippet:\s*(.+)/s;
// Mirrors _normalize_url_scheme: a dotted host, optionally followed by a port
// that may be empty ("example.com:" fetches on the default port) but otherwise
// has to be in range, so the card names a host only when the backend fetches it.
const RE_BARE_HOST =
  /^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?::(\d{0,5}))?(?:[/?#]|$)/;

function isBareHostFetchedAsHttps(value: string): boolean {
  const match = RE_BARE_HOST.exec(value);
  if (!match) return false;
  const port = match[1];
  if (!port) return true;
  return Number(port) >= 1 && Number(port) <= 65535;
}

/**
 * Reject non-http(s) URLs. Web-search/fetch output is provider-controlled,
 * so hostile `javascript:` / `data:` lines must not reach a Source <a href>.
 */
function isSafeHttpUrl(raw: string): boolean {
  const value = raw.trim();
  if (!value || /[\r\n]/.test(value)) return false;
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

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
    if (!titleMatch || !urlMatch) continue;
    const url = urlMatch[1].trim();
    if (!isSafeHttpUrl(url)) continue;
    sources.push({
      title: titleMatch[1].trim(),
      url,
      snippet: snippetMatch?.[1]?.trim() ?? "",
    });
  }
  return sources;
}

const WebSearchToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  // Coerced, like image_queries below: a local model routinely emits a number or an
  // object here, and .trim() on one crashes the card that was meant to show the call.
  const query = toolArgText((args as { query?: unknown })?.query);
  const url = toolArgText((args as { url?: unknown })?.url).trim();
  const isUrlFetch = !!url;
  const rawImageQueries = (args as { image_queries?: unknown })?.image_queries;
  const imageQueries = Array.isArray(rawImageQueries)
    ? rawImageQueries
        .map((q) => toolArgText(q).trim())
        .filter(Boolean)
        .slice(0, 5)
    : [];
  const imageLabel = imageQueries.join(", ");
  const isImageOnly = !isUrlFetch && !query.trim() && imageQueries.length > 0;
  // The header speaks for the result: a call that found nothing must not claim it did.
  const foundImages = isSearchImagesToolResult(result);
  const displayDomain = (() => {
    if (!url) return "";
    // new URL() throws on the bare hosts the backend fetches, so mirror that
    // grammar or the card names no host for exactly the URLs it does fetch.
    const bare = url.startsWith("//") ? url.slice(2) : url;
    const candidate = isBareHostFetchedAsHttps(bare) ? `https://${bare}` : url;
    try {
      const parsed = new URL(candidate);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:")
        return "";
      return parsed.hostname.replace(/^www\./, "");
    } catch {
      return "";
    }
  })();
  const isRunning = status?.type === "running";
  const withImages = isSearchImagesToolResult(result);
  const resultText =
    result == null
      ? ""
      : withImages
        ? stringifyToolResult(result.text)
        : stringifyToolResult(result);
  const images = withImages ? result.webImages : [];
  const sources = resultText ? parseSearchResults(resultText) : [];

  // Collapse when LLM starts generating text after the tool call
  const hasText = useAuiState(({ message }) =>
    message.content.some(
      (p) =>
        p.type === "text" &&
        "text" in p &&
        (p as { text: string }).text.length > 0,
    ),
  );
  // Ask permission mode gates every local tool call, and the query or code
  // being approved lives inside ToolFallbackContent while Allow/Deny render
  // outside the card, so a collapsed card asks for a decision about text the
  // trigger only shows truncated.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={
          isUrlFetch
            ? displayDomain
              ? `Read ${displayDomain}`
              : "Read page"
            : isImageOnly
              ? isRunning
                ? `Finding images for “${imageLabel}”`
                : foundImages
                  ? `Found images for “${imageLabel}”`
                  : `No images for “${imageLabel}”`
              : query
                ? imageLabel && foundImages
                  ? `Searched "${query}" · images for ${imageLabel}`
                  : `Searched "${query}"`
                : "Web Search"
        }
        status={status}
        icon={GlobeIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center text-sm text-muted-foreground">
            <span>
              {isUrlFetch ? (
                <>Reading {displayDomain || "page"}&hellip;</>
              ) : isImageOnly ? (
                <>Finding images for &ldquo;{imageLabel}&rdquo;&hellip;</>
              ) : (
                <>Searching for &ldquo;{query}&rdquo;&hellip;</>
              )}
            </span>
          </div>
        ) : sources.length === 0 && images.length > 0 ? (
          <div className="flex flex-col gap-2">
            <div
              className="flex flex-wrap gap-1.5"
              data-testid="web-search-images"
            >
              {images.map((entry) => (
                <SearchImageThumb key={entry.id} entry={entry} size="strip" />
              ))}
            </div>
            {resultText && (
              <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
                {resultText}
              </pre>
            )}
          </div>
        ) : sources.length > 0 ? (
          <div className="flex flex-col gap-2">
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
            {images.length > 0 && (
              <div
                className="flex flex-wrap gap-1.5"
                data-testid="web-search-images"
              >
                {images.map((entry) => (
                  <SearchImageThumb key={entry.id} entry={entry} size="strip" />
                ))}
              </div>
            )}
          </div>
        ) : resultText ? (
          <div>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
              {resultText}
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
