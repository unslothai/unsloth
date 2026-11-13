// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s_.-]/g, "");
}

export function tokenizeQuery(query: string): string[] {
  return query
    .split(/\s+/)
    .map((token) => normalizeForSearch(token))
    .filter((token) => token.length > 0);
}

export interface OwnerSearchQuery {
  owner: string;
  query: string;
}

const OWNER_SEARCH_RE = /^([^/\s]+)\/([^/\s]*)$/;

/** Split an exact Hugging Face owner search, including the useful `owner/` form. */
export function parseOwnerSearchQuery(value: string): OwnerSearchQuery | null {
  const match = OWNER_SEARCH_RE.exec(value.trim());
  if (!match) return null;
  return { owner: match[1], query: match[2] };
}

export function matchTokens(haystack: string, tokens: readonly string[]): boolean {
  if (tokens.length === 0) return true;
  const normalized = normalizeForSearch(haystack);
  for (const token of tokens) {
    if (!normalized.includes(token)) return false;
  }
  return true;
}
