// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ToolCallMessagePartStatus } from "@assistant-ui/react";

/** Caps the JSON branch: engines' max string lengths differ wildly (V8 throws, Safari balloons). */
const MAX_SERIALISED_LENGTH = 100_000;

/** Props declared `string` are only requests to the model; a throw in a card crashes the app. */
export const toolArgText = (value: unknown): string => {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    if (typeof value === "object") {
      // Serialised, not coerced: `String({"toString":null})` throws.
      // `?? ""` covers a toJSON returning undefined.
      const json = JSON.stringify(value) ?? "";
      return json.length > MAX_SERIALISED_LENGTH
        ? `${json.slice(0, MAX_SERIALISED_LENGTH)}…`
        : json;
    }
    // Not a template literal: `String(sym)` works, `${sym}` throws on symbols.
    return String(value);
  } catch {
    // Load-bearing: Firefox and Safari throw on deep nesting.
    // Keep the binding bare: Firefox's is not a RangeError.
    return "";
  }
};

// Running until a truthy result, so a tool returning "" or 0 reads as running.
export const isToolCallRunning = (
  status?: ToolCallMessagePartStatus,
): boolean => status?.type === "running";

export const isToolCallCancelled = (
  status?: ToolCallMessagePartStatus,
): boolean => status?.type === "incomplete" && status.reason === "cancelled";

export function toolFallbackLabel(status?: ToolCallMessagePartStatus): string {
  if (isToolCallCancelled(status)) return "Cancelled tool";
  return isToolCallRunning(status) ? "Using tool" : "Used tool";
}

export interface WebSearchToolNameState {
  isRunning: boolean;
  isFindInPage: boolean;
  isUrlFetch: boolean;
  isImageOnly: boolean;
  foundImages: boolean;
  displayDomain: string;
  pattern: string;
  query: string;
  imageLabel: string;
}

// The trigger prefixes these with "Using tool" while the call runs.
export function webSearchToolName(state: WebSearchToolNameState): string {
  const {
    isRunning,
    isFindInPage,
    isUrlFetch,
    isImageOnly,
    foundImages,
    displayDomain,
    pattern,
    query,
    imageLabel,
  } = state;

  if (isFindInPage) {
    const page = displayDomain || "page";
    if (isRunning) {
      return pattern
        ? `Finding "${pattern}" in ${page}…`
        : `Searching ${page}…`;
    }
    // Neutral: the action carries no match status, so finishing is no evidence of a match.
    return pattern
      ? `Searched for "${pattern}" in ${page}`
      : `Searched ${page}`;
  }
  if (isUrlFetch) {
    if (isRunning) return `Reading ${displayDomain || "page"}…`;
    return displayDomain ? `Read ${displayDomain}` : "Read page";
  }
  if (isImageOnly) {
    if (isRunning) return `Finding images for “${imageLabel}”`;
    return foundImages
      ? `Found images for “${imageLabel}”`
      : `No images for “${imageLabel}”`;
  }
  if (!query) return isRunning ? "Searching…" : "Web Search";
  if (isRunning) return `Searching for "${query}"…`;
  return imageLabel && foundImages
    ? `Searched "${query}" · images for ${imageLabel}`
    : `Searched "${query}"`;
}

export interface KnowledgeBaseToolNameState {
  isRunning: boolean;
  query: string;
}

export function knowledgeBaseToolName(
  state: KnowledgeBaseToolNameState,
): string {
  const { isRunning, query } = state;
  if (isRunning) {
    return query
      ? `Searching documents for "${query}"…`
      : "Searching documents…";
  }
  return query ? `Searched documents for "${query}"` : "Knowledge search";
}
