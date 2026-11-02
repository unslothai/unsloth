// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  defaultParseSearch,
  defaultStringifySearch,
} from "@tanstack/react-router";

// A browser search query is literal text, even when it is valid JSON such as "null" or "1.00".
export function parseSearch(search: string): Record<string, unknown> {
  const result = defaultParseSearch(search) as Record<string, unknown>;
  const query = new URLSearchParams(search).get("q");
  if (query !== null) result.q = query;
  return result;
}

export function stringifySearch(search: Record<string, unknown>): string {
  const { q, ...other } = search;
  const serialized = defaultStringifySearch(other);
  if (typeof q !== "string") return serialized;
  return `${serialized}${serialized ? "&" : "?"}q=${encodeURIComponent(q)}`;
}
