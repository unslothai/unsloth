// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Separate from chat-page so /chat can validate search without loading the page.
export type ChatSearch = {
  thread?: string;
  compare?: string;
  new?: string;
  project?: string;
  // A browser search-engine shortcut's text, e.g. `/chat?q=%s`: opens a new chat with it typed in.
  q?: string;
};

// The router keeps `q` a string (router-search.ts); a blank one is no query.
export function queryParam(search: Record<string, unknown>): string | undefined {
  const { q } = search;
  return typeof q === "string" && q.trim() ? q : undefined;
}

export function validateChatSearch(
  search: Record<string, unknown>,
): ChatSearch {
  return {
    thread: typeof search.thread === "string" ? search.thread : undefined,
    compare: typeof search.compare === "string" ? search.compare : undefined,
    new: typeof search.new === "string" ? search.new : undefined,
    project: typeof search.project === "string" ? search.project : undefined,
    q: queryParam(search),
  };
}
