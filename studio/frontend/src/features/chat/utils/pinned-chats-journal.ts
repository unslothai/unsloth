// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type PinOperation,
  applyPinOperation,
  validatePinOperation,
  validatePinnedIds,
} from "./pinned-chats-sync.ts";
import type { PinnedChatsFallbackState } from "./pinned-chats-transport.ts";
import { PIN_JOURNAL_PREFIX } from "../stores/pinned-chats-store.ts";
import { createNavigationNonce } from "../../../lib/navigation-nonce.ts";

export { PIN_JOURNAL_PREFIX };

interface Entry {
  operation?: PinOperation;
  baseIds?: string[];
  order: number;
}

/** One immutable key per intent prevents tabs from overwriting each other's queues. */
export function createPinnedChatsJournal(storage: Storage, scope: string) {
  const prefix = `${PIN_JOURNAL_PREFIX}${encodeURIComponent(scope)}:`;
  const seedKey = `${prefix}fallback:seed:${createNavigationNonce()}`;
  const observed = { pending: new Set<string>(), fallback: new Set<string>() };
  let lastOrder = 0;

  function keys(kind: "pending" | "fallback"): string[] {
    const start = `${prefix}${kind}:`;
    const result: string[] = [];
    for (let i = 0; i < storage.length; i += 1) {
      const key = storage.key(i);
      if (key?.startsWith(start)) {
        result.push(key);
      }
    }
    return result;
  }

  function read(kind: "pending" | "fallback"): [string, Entry][] {
    const result: [string, Entry][] = [];
    for (const key of keys(kind)) {
      const raw = storage.getItem(key);
      if (raw === null) {
        continue;
      }
      const value = JSON.parse(raw) as Entry;
      if (!(value && Number.isFinite(value.order))) {
        throw new Error("Invalid saved pin change");
      }
      if (kind === "pending" || value.operation !== undefined) {
        value.operation = validatePinOperation(value.operation);
      }
      if (kind === "fallback") {
        value.baseIds = validatePinnedIds(value.baseIds);
        if (!value.baseIds) {
          throw new Error("Invalid saved pin state");
        }
      }
      observed[kind].add(key);
      lastOrder = Math.max(lastOrder, value.order);
      result.push([key, value]);
    }
    return result.sort(
      ([a, x], [b, y]) => x.order - y.order || a.localeCompare(b),
    );
  }

  function write(
    kind: "pending" | "fallback",
    operations: PinOperation[],
    baseIds?: string[],
  ): void {
    const keep = new Set<string>();
    const keyOf = (operation: PinOperation) =>
      `${prefix}${kind}:${operation.operationId}`;
    const storedOrder = (key: string): number | undefined => {
      const raw = storage.getItem(key);
      if (raw === null) return undefined;
      const order = (JSON.parse(raw) as Entry).order;
      return Number.isFinite(order) ? order : undefined;
    };
    let previous: number | undefined;
    operations.forEach((operation, index) => {
      validatePinOperation(operation);
      const key = keyOf(operation);
      let order = storedOrder(key);
      if (order === undefined) {
        // Seen here and gone now: another tab acknowledged or cancelled it. Never recreate it.
        if (observed[kind].has(key)) return;
        // A trimmed pin keeps its slot: it must replay before the unpin fencing its server ID.
        let next: number | undefined;
        for (const later of operations.slice(index + 1)) {
          next = storedOrder(keyOf(later));
          if (next !== undefined) break;
        }
        order =
          next === undefined
            ? Math.max(Date.now(), lastOrder + 1, (previous ?? 0) + 1)
            : previous === undefined
              ? next - 1
              : (previous + next) / 2;
        lastOrder = Math.max(lastOrder, order);
        storage.setItem(key, JSON.stringify({ operation, baseIds, order }));
      }
      previous = order;
      keep.add(key);
    });
    // A fallback seed is needed even before the first offline mutation.
    if (
      kind === "fallback" &&
      baseIds !== undefined &&
      operations.length === 0
    ) {
      const key = seedKey;
      keep.add(key);
      lastOrder = Math.max(Date.now(), lastOrder + 1);
      storage.setItem(key, JSON.stringify({ baseIds, order: lastOrder }));
    }
    // Only acknowledge entries seen here: another tab may have appended one mid-request.
    for (const key of observed[kind]) {
      if (!keep.has(key)) {
        storage.removeItem(key);
      }
    }
    observed[kind] = keep;
  }

  return {
    readPending(): PinOperation[] {
      return read("pending").flatMap(([, value]) =>
        value.operation ? [value.operation] : [],
      );
    },
    writePending(operations: PinOperation[]): void {
      write("pending", operations);
    },
    readFallback(): PinnedChatsFallbackState | undefined {
      const entries = read("fallback");
      if (entries.length === 0) {
        return undefined;
      }
      const baseIds = [
        ...new Set(entries.flatMap(([, value]) => value.baseIds ?? [])),
      ];
      validatePinnedIds(baseIds);
      const operations = entries.flatMap(([, value]) =>
        value.operation ? [value.operation] : [],
      );
      return {
        baseIds,
        operations,
        pinnedIds: operations.reduce(applyPinOperation, baseIds),
      };
    },
    writeFallback(state: PinnedChatsFallbackState | undefined): void {
      write("fallback", state?.operations ?? [], state?.baseIds);
    },
    clear(): void {
      for (const kind of ["pending", "fallback"] as const) {
        for (const key of keys(kind)) {
          storage.removeItem(key);
        }
        observed[kind].clear();
      }
    },
  };
}
