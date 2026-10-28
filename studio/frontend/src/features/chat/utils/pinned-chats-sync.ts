// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createNavigationNonce } from "../../../lib/navigation-nonce.ts";

export type PinOperation = {
  operationId: string;
  kind: "pin";
  ids: string[];
  pinned: boolean;
  /** Queued pins this unpin superseded; the server acknowledges them without applying. */
  cancels?: string[];
  /** Server replay ID when it differs from the journal's: a trimmed intent keeps its original's. */
  serverId?: string;
};

export interface PinnedChatsSyncOptions {
  load: () => Promise<string[] | undefined>;
  compareAndSet: (
    expected: string[] | undefined,
    next: string[],
    operation?: PinOperation,
  ) => Promise<{ pinnedIds: string[] | undefined; applied: boolean }>;
  readLegacy: () => string[];
  migrateLegacy?: (ids: string[]) => Promise<string[] | undefined>;
  readPending?: () => PinOperation[];
  writePending?: (operations: PinOperation[]) => void;
  isRetryable?: (error: unknown) => boolean;
  publish: (ids: string[]) => void;
  onError?: (error: unknown) => void;
}

export class PinnedChatsValidationError extends Error {}

export function isValidPinId(value: unknown): value is string {
  if (typeof value !== "string" || value.length === 0 || value.length > 512) {
    return false;
  }
  // JSON can contain lone UTF-16 surrogates, which cannot be stored as UTF-8.
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        return false;
      }
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }
  return true;
}

export function validatePinnedIds(value: unknown): string[] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (
    !Array.isArray(value) ||
    value.length > 10_000 ||
    !value.every(isValidPinId) ||
    new Set(value).size !== value.length
  ) {
    throw new PinnedChatsValidationError("Invalid pinned chats response");
  }
  return value.slice();
}

export function validatePinOperation(value: unknown): PinOperation {
  if (!value || typeof value !== "object") {
    throw new PinnedChatsValidationError("Invalid pending pin operation");
  }
  const operation = value as Partial<PinOperation>;
  if (!isValidPinId(operation.operationId)) {
    throw new PinnedChatsValidationError("Invalid pending pin operation");
  }
  if (operation.kind === "pin" && typeof operation.pinned === "boolean") {
    const ids = validatePinnedIds(operation.ids);
    const cancels = validatePinnedIds(operation.cancels);
    const serverId =
      operation.serverId === undefined || isValidPinId(operation.serverId)
        ? operation.serverId
        : null;
    if (ids && serverId !== null) {
      return {
        kind: "pin",
        operationId: operation.operationId,
        ids,
        pinned: operation.pinned,
        ...(cancels?.length ? { cancels } : {}),
        ...(serverId ? { serverId } : {}),
      };
    }
  }
  throw new PinnedChatsValidationError("Invalid pending pin operation");
}

/** Apply relative intent to the latest state, without reviving missing rows. */
export function applyPinOperation(
  ids: string[],
  operation: PinOperation,
): string[] {
  const selected = new Set(operation.ids);
  if (!operation.pinned) {
    return ids.filter((id) => !selected.has(id));
  }
  const present = new Set(ids);
  return [...operation.ids.filter((id) => !present.has(id)), ...ids];
}

/** A durable FIFO of pin intents, rebased when another client writes first. */
export function createPinnedChatsSync(options: PinnedChatsSyncOptions) {
  let generation = 0;
  let loaded = false;
  let confirmed: string[] | undefined;
  let pending: PinOperation[] = [];
  let queue: Promise<void> = Promise.resolve();
  let refreshing: Promise<void> | undefined;
  const scheduled = new Map<string, Promise<void>>();
  // A response arrived, but removing its journal entry may still need retrying.
  const acknowledged = new Set<string>();

  function publish() {
    options.publish(pending.reduce(applyPinOperation, confirmed ?? []));
  }

  function persist(operations: PinOperation[]): void {
    options.writePending?.(operations.map(validatePinOperation));
  }

  function restore(): void {
    if (!options.readPending) {
      return;
    }
    const saved = options.readPending();
    if (!Array.isArray(saved) || saved.length > 10_000) {
      throw new PinnedChatsValidationError("Invalid pending pin operations");
    }
    const operations = saved.map(validatePinOperation);
    // The journal is the queue, in its order: a missing intent was acknowledged/cancelled elsewhere.
    const next = [
      ...new Map(operations.map((entry) => [entry.operationId, entry])).values(),
    ];
    const changed =
      next.length !== pending.length ||
      next.some((entry, index) => entry.operationId !== pending[index]?.operationId);
    if (changed) {
      pending = next;
      publish();
    }
  }

  function rejected(error: unknown): Promise<void> {
    options.onError?.(error);
    return Promise.reject(error);
  }

  function enqueue(task: (epoch: number) => Promise<void>): Promise<void> {
    const epoch = generation;
    const result = queue.then(async () => {
      if (epoch !== generation) {
        return;
      }
      try {
        await task(epoch);
      } catch (error) {
        if (epoch !== generation) {
          return;
        }
        options.onError?.(error);
        throw error;
      }
    });
    queue = result.catch(() => undefined);
    return result;
  }

  async function mergeLegacy(
    ids: string[] | undefined,
    epoch: number,
  ): Promise<string[] | undefined> {
    const legacy = validatePinnedIds([...new Set(options.readLegacy())]) ?? [];
    if (legacy.length === 0) {
      return ids;
    }
    let result: string[] | undefined;
    if (options.migrateLegacy) {
      result = await options.migrateLegacy(legacy);
    } else if (ids === undefined) {
      // No merge endpoint: never claim an empty setting before importing legacy pins.
      result = (await options.compareAndSet(undefined, legacy)).pinnedIds;
    } else {
      return ids;
    }
    if (epoch !== generation) {
      return ids;
    }
    // The backend can leave the setting absent when every legacy ID is orphaned.
    return validatePinnedIds(result);
  }

  async function hydrate(epoch: number, force = false): Promise<void> {
    if (loaded && !force) {
      return;
    }
    const ids = validatePinnedIds(await options.load());
    if (epoch !== generation) {
      return;
    }
    const migrated = await mergeLegacy(ids, epoch);
    if (epoch !== generation) {
      return;
    }
    confirmed = migrated;
    loaded = true;
    publish();
  }

  function finish(operation: PinOperation): void {
    const remaining = pending.filter(
      (entry) => entry.operationId !== operation.operationId,
    );
    persist(remaining);
    pending = remaining;
    acknowledged.delete(operation.operationId);
    publish();
  }

  async function writeOperation(
    operation: PinOperation,
    epoch: number,
  ): Promise<void> {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      if (epoch !== generation) {
        return;
      }
      try {
        // A rebase must not resend an intent another tab cancelled during the last attempt.
        restore();
      } catch {
        // An unreadable journal keeps the in-memory queue.
      }
      if (!pending.some((entry) => entry.operationId === operation.operationId)) {
        return;
      }
      const next =
        validatePinnedIds(applyPinOperation(confirmed ?? [], operation)) ?? [];
      const result = await options.compareAndSet(
        confirmed?.slice(),
        next,
        operation,
      );
      if (epoch !== generation) {
        return;
      }
      if (!result || typeof result.applied !== "boolean") {
        throw new PinnedChatsValidationError("Invalid pinned chats response");
      }
      const ids = validatePinnedIds(result.pinnedIds);
      if (result.applied && ids === undefined) {
        throw new PinnedChatsValidationError("Invalid pinned chats response");
      }
      confirmed = ids;
      if (result.applied) {
        acknowledged.add(operation.operationId);
        finish(operation);
        return;
      }
      publish();
    }
    throw new Error(
      "Pinned chats changed in another browser. Please try again.",
    );
  }

  async function recover(
    operation: PinOperation,
    error: unknown,
    epoch: number,
  ): Promise<void> {
    const retryable =
      !(error instanceof PinnedChatsValidationError) &&
      (options.isRetryable?.(error) ?? true);
    if (!(retryable || acknowledged.has(operation.operationId))) {
      // A rejected intent cannot succeed on retry; later valid intents may proceed.
      acknowledged.add(operation.operationId);
      finish(operation);
    }
    loaded = false;
    try {
      const ids = validatePinnedIds(await options.load());
      if (epoch !== generation) {
        return;
      }
      confirmed = ids;
      loaded = ids !== undefined;
    } catch {
      // Keep the last confirmed state and the durable optimistic intents.
    }
    if (epoch === generation) {
      publish();
    }
  }

  async function execute(
    operation: PinOperation,
    epoch: number,
  ): Promise<void> {
    try {
      // Another tab may have cancelled this intent while an earlier one was in flight.
      restore();
    } catch {
      // An unreadable journal keeps the in-memory queue.
    }
    if (!pending.some((entry) => entry.operationId === operation.operationId)) {
      return;
    }
    if (pending[0]?.operationId !== operation.operationId) {
      throw new Error("An earlier pin change is still waiting to save.");
    }
    try {
      if (acknowledged.has(operation.operationId)) {
        finish(operation);
        return;
      }
      await hydrate(epoch);
      await writeOperation(operation, epoch);
    } catch (error) {
      if (epoch !== generation) {
        return;
      }
      await recover(operation, error, epoch);
      throw error;
    }
  }

  function schedulePending(): Promise<void>[] {
    return pending.map((operation) => {
      const existing = scheduled.get(operation.operationId);
      if (existing) {
        return existing;
      }
      const result = enqueue((epoch) => execute(operation, epoch));
      scheduled.set(operation.operationId, result);
      const clear = () => {
        if (scheduled.get(operation.operationId) === result) {
          scheduled.delete(operation.operationId);
        }
      };
      void result.then(clear, clear);
      return result;
    });
  }

  function mutate(intent: Omit<PinOperation, "operationId">): Promise<void> {
    try {
      restore();
      let operation = validatePinOperation({
        ...intent,
        operationId: createNavigationNonce(),
      });
      // An unpin strips its chats from earlier queued pins, so an unplaceable pin cannot block.
      const cancelled = new Set(operation.pinned ? [] : operation.ids);
      const superseded: string[] = [];
      const kept = pending.flatMap((entry) => {
        if (!entry.pinned) return [entry];
        const ids = entry.ids.filter((id) => !cancelled.has(id));
        if (ids.length === entry.ids.length) return [entry];
        const serverId = entry.serverId ?? entry.operationId;
        // Another tab may already have sent it: the unpin fences its ID on the server.
        superseded.push(serverId);
        // New journal ID (entries are immutable), same server ID; it runs before the fencing unpin.
        return ids.length === 0
          ? []
          : [{ ...entry, ids, operationId: createNavigationNonce(), serverId }];
      });
      if (superseded.length > 0) {
        // An original and its trimmed replacement can coexist after a partial journal write.
        operation = validatePinOperation({ ...operation, cancels: [...new Set(superseded)] });
      }
      const next = [...kept, operation];
      if (next.length > 10_000) {
        throw new PinnedChatsValidationError("Too many pending pin changes");
      }
      // Persist before presenting success in the UI; navigation may happen immediately.
      persist(next);
      pending = next;
      publish();
      schedulePending();
      const result = scheduled.get(operation.operationId);
      if (!result) {
        throw new Error("Pin operation was not scheduled");
      }
      return result;
    } catch (error) {
      return rejected(error);
    }
  }

  return {
    refresh(): Promise<void> {
      if (refreshing) {
        return refreshing;
      }
      try {
        restore();
      } catch (error) {
        return rejected(error);
      }
      const hydration = enqueue((epoch) => hydrate(epoch, true));
      const operations = schedulePending();
      const result = Promise.allSettled([hydration, ...operations]).then(
        (results) => {
          const failure = results.find((entry) => entry.status === "rejected");
          if (failure?.status === "rejected") {
            throw failure.reason;
          }
        },
      );
      refreshing = result;
      const clear = () => {
        if (refreshing === result) {
          refreshing = undefined;
        }
      };
      void result.then(clear, clear);
      return result;
    },
    setPinned(ids: string[], pinned: boolean): Promise<void> {
      return mutate({ kind: "pin", ids: [...new Set(ids)], pinned });
    },
    reset({ clearPending = false }: { clearPending?: boolean } = {}): void {
      generation += 1;
      loaded = false;
      confirmed = undefined;
      pending = [];
      queue = Promise.resolve();
      refreshing = undefined;
      scheduled.clear();
      acknowledged.clear();
      publish();
      if (clearPending) {
        persist([]);
      }
    },
  };
}
