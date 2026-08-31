// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import Dexie, { type EntityTable, liveQuery } from "dexie";
import { useEffect, useRef, useState } from "react";
import type { MessageRecord, ThreadRecord } from "./types";

// Legacy browser-only chat storage. Replaced by studio.db (see
// chat-history-storage.ts), kept read-only for the one-shot import path.
export const DEXIE_DB_NAME = "unsloth-chat";

const db = new Dexie(DEXIE_DB_NAME) as Dexie & {
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
 * Include every semantic query input in `deps` (filters, sort keys, IDs). `querier`
 * identity is ignored to avoid re-subscribing every render on inline functions.
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
