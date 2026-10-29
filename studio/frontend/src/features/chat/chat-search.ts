// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept apart from chat-page so the /chat route can validate its search without loading the page.
export type ChatSearch = {
  thread?: string;
  compare?: string;
  new?: string;
  project?: string;
};

export function validateChatSearch(
  search: Record<string, unknown>,
): ChatSearch {
  return {
    thread: typeof search.thread === "string" ? search.thread : undefined,
    compare: typeof search.compare === "string" ? search.compare : undefined,
    new: typeof search.new === "string" ? search.new : undefined,
    project: typeof search.project === "string" ? search.project : undefined,
  };
}
