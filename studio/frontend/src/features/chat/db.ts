// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import Dexie, { type EntityTable, liveQuery } from "dexie";
import { useEffect, useRef, useState } from "react";
import type { MessageRecord, ThreadRecord } from "./types";

const db = new Dexie("unsloth-chat") as Dexie & {
  threads: EntityTable<ThreadRecord, "id">;
  messages: EntityTable<MessageRecord, "id">;
};

db.version(1).stores({
  threads: "id, modelType, pairId, archived, createdAt",
  messages: "id, threadId, createdAt",
});

db.version(2)
  .stores({
    threads: "id, modelType, pairId, archived, createdAt",
    messages: "id, threadId, createdAt",
  })
  .upgrade((tx) => tx.table("messages").clear());

db.version(3)
  .stores({
    threads: "id, modelType, pairId, archived, createdAt",
    messages: "id, threadId, createdAt",
  })
  .upgrade((tx) =>
    tx
      .table("threads")
      .toCollection()
      .modify((thread) => {
        if (!thread.modelId) thread.modelId = "";
      }),
  );

export { db };

/**
 * Wraps Dexie liveQuery for React state updates.
 *
 * Important: include every semantic query input in `deps` (filters, sort keys,
 * IDs, etc). `querier` identity is intentionally ignored to avoid re-subscribing
 * on every render when callers pass inline functions.
 */
export function useLiveQuery<T>(
  querier: () => Promise<T>,
  deps: unknown[] = [],
): T | undefined {
  const [value, setValue] = useState<T>();
  const querierRef = useRef(querier);
  querierRef.current = querier;

  useEffect(() => {
    const sub = liveQuery(() => querierRef.current()).subscribe({
      next: setValue,
      error: (err) => console.error("useLiveQuery:", err),
    });
    return () => sub.unsubscribe();
    // Intentionally omit `querier` from deps: inline functions would re-subscribe every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return value;
}
