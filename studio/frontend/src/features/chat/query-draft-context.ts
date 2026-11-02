// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";
import type { QueryDraft } from "./utils/query-draft";

export const queryDrafts = new Map<string, QueryDraft>();
export const QueryDraftContext = createContext<{
  draft: QueryDraft;
  threadId: string | null;
} | null>(null);

export function getQueryDraft(
  nonce: string | undefined,
  temporary: boolean,
): QueryDraft | null {
  const draft = nonce ? queryDrafts.get(nonce) : undefined;
  return draft?.temporary === temporary ? draft : null;
}

export function isQueryDraftScope(scope: string | null): boolean {
  return scope?.startsWith("chat-query-draft:") ?? false;
}
