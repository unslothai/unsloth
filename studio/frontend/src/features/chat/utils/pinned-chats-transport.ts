// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChatSettingsRequestError } from "./settings-retry.ts";

import {
  applyPinOperation,
  PinnedChatsValidationError,
  validatePinnedIds,
  validatePinOperation,
  type PinOperation,
} from "./pinned-chats-sync.ts";

export interface PinnedChatsFallbackState {
  baseIds: string[];
  pinnedIds: string[];
  operations: PinOperation[];
}

interface PinnedChatsTransportOptions {
  load: () => Promise<string[] | undefined>;
  compareAndSet: (
    expected: string[] | undefined,
    next: string[],
    operation?: PinOperation,
  ) => Promise<{ pinnedIds: string[] | undefined; applied: boolean }>;
  migrateLegacy?: (ids: string[]) => Promise<string[] | undefined>;
  readLegacy: () => string[];
  readFallback: () => PinnedChatsFallbackState | undefined;
  writeFallback: (state: PinnedChatsFallbackState | undefined) => void;
  onUnsupported?: () => void;
}

class ReplayConflictError extends Error {}

const PIN_OPERATION_FIELDS = new Set([
  "operationId",
  "unpinnedChatIds",
  "cancelledOperationIds",
]);

function isUnsupported(error: unknown): boolean {
  if (!(error instanceof ChatSettingsRequestError)) return false;
  if (error.status === 404 || error.status === 405) return true;
  if (
    error.status !== 400 ||
    !Array.isArray(error.detail) ||
    !error.detail.length
  ) {
    return false;
  }
  let rejectsPinField = false;
  const onlyUnsupportedPins = error.detail.every((entry: unknown) => {
    if (!entry || typeof entry !== "object") return false;
    const { type, loc, msg } = entry as {
      type?: unknown;
      loc?: unknown;
      msg?: unknown;
    };
    if (!Array.isArray(loc)) return false;
    if (type === "extra_forbidden" && loc.at(-1) === "pinnedChatIds") {
      rejectsPinField = true;
      return true;
    }
    // A backend downgraded since support was probed also rejects this client's operation metadata.
    if (
      type === "extra_forbidden" &&
      loc.length === 1 &&
      PIN_OPERATION_FIELDS.has(String(loc[0]))
    ) {
      return true;
    }
    // Older backends also reject the migration's expectedAbsent naming the new field.
    return (
      type === "value_error" &&
      loc.length === 1 &&
      loc[0] === "expectedAbsent" &&
      msg === "Value error, Unknown chat settings field(s): pinnedChatIds"
    );
  });
  return onlyUnsupportedPins && rejectsPinField;
}

/** Drop pins a later entry unpins; that unpin fences each dropped pin's server ID. */
function compactReplay(operations: PinOperation[]): PinOperation[] {
  const unpinFor = new Map<string, PinOperation & { cancels: string[] }>();
  const kept: PinOperation[] = [];
  for (const operation of [...operations].reverse()) {
    if (!operation.pinned) {
      const fence = { ...operation, cancels: [...(operation.cancels ?? [])] };
      operation.ids.forEach((id) => unpinFor.set(id, fence));
      kept.push(fence);
      continue;
    }
    const fenceId = operation.ids.find((id) => unpinFor.has(id));
    if (fenceId === undefined) {
      kept.push(operation);
      continue;
    }
    const serverId = operation.serverId ?? operation.operationId;
    unpinFor.get(fenceId)!.cancels.push(serverId);
    const ids = operation.ids.filter((id) => !unpinFor.has(id));
    // Same server ID, replayed before the fence: an already-committed copy is just acknowledged.
    if (ids.length > 0) kept.push({ ...operation, ids, serverId });
  }
  return kept
    .reverse()
    .map(({ cancels, ...operation }) =>
      cancels?.length ? { ...operation, cancels: [...new Set(cancels)] } : operation,
    );
}

function sameIds(left: string[] | undefined, right: string[]): boolean {
  return left !== undefined && left.length === right.length &&
    left.every((id, index) => id === right[index]);
}

/** Older servers use a durable local journal which is replayed after upgrade. */
export function createPinnedChatsTransport(options: PinnedChatsTransportOptions) {
  let localOnly = false;
  let generation = 0;
  let lastRemote: string[] | undefined;
  let recoveredOnLoad = false;

  function assertCurrent(epoch: number): void {
    if (epoch !== generation) throw new Error("Pinned chat session changed");
  }

  function readFallback(): PinnedChatsFallbackState | undefined {
    const state = options.readFallback();
    if (state === undefined) return undefined;
    if (!state || !Array.isArray(state.operations) || state.operations.length > 10_000)
      throw new PinnedChatsValidationError("Invalid local pin journal");
    const baseIds = validatePinnedIds(state.baseIds);
    const pinnedIds = validatePinnedIds(state.pinnedIds);
    const operations = state.operations.map(validatePinOperation);
    if (!baseIds || !pinnedIds || new Set(operations.map((op) => op.operationId)).size !== operations.length)
      throw new PinnedChatsValidationError("Invalid local pin journal");
    return { baseIds, pinnedIds, operations };
  }

  function fallback(error: unknown, epoch: number, expected?: string[]): PinnedChatsFallbackState {
    assertCurrent(epoch);
    if (!isUnsupported(error)) throw error;
    let state = readFallback();
    if (!state) {
      const baseIds = validatePinnedIds(expected ?? lastRemote ?? options.readLegacy())!;
      state = { baseIds, pinnedIds: baseIds.slice(), operations: [] };
      // Failure to save the recovery record must not be reported as a local save.
      options.writeFallback(state);
    }
    if (!localOnly) {
      localOnly = true;
      options.onUnsupported?.();
    }
    return state;
  }

  function validateResult(result: { pinnedIds: string[] | undefined; applied: boolean }) {
    if (!result || typeof result.applied !== "boolean")
      throw new PinnedChatsValidationError("Invalid pinned chats response");
    const pinnedIds = validatePinnedIds(result.pinnedIds);
    if (result.applied && pinnedIds === undefined)
      throw new PinnedChatsValidationError("Invalid pinned chats response");
    return { pinnedIds, applied: result.applied };
  }

  async function remoteMigration(ids: string[], current: string[] | undefined, epoch: number) {
    if (options.migrateLegacy) {
      const migrated = validatePinnedIds(await options.migrateLegacy(ids));
      assertCurrent(epoch);
      // No valid legacy IDs may remain after pruning deleted chats.
      return migrated;
    }
    if (current !== undefined) return current;
    if (!ids.length) return current;
    const result = validateResult(await options.compareAndSet(undefined, ids));
    assertCurrent(epoch);
    if (result.pinnedIds === undefined)
      throw new PinnedChatsValidationError("Invalid pin migration response");
    return result.pinnedIds;
  }

  async function recover(state: PinnedChatsFallbackState, current: string[] | undefined, epoch: number) {
    current = await remoteMigration(state.baseIds, current, epoch);
    for (const operation of compactReplay(state.operations)) {
      let applied = false;
      for (let attempt = 0; attempt < 8; attempt += 1) {
        assertCurrent(epoch);
        const next = validatePinnedIds(applyPinOperation(current ?? [], operation))!;
        const result = validateResult(await options.compareAndSet(current, next, operation));
        assertCurrent(epoch);
        current = result.pinnedIds;
        if (result.applied) {
          applied = true;
          break;
        }
      }
      if (!applied) throw new ReplayConflictError("Pinned chats changed in another browser. Please try again.");
    }
    assertCurrent(epoch);
    // Cleared only once every write is acknowledged; the server deduplicates a retried replay.
    options.writeFallback(undefined);
    localOnly = false;
    lastRemote = current?.slice();
    recoveredOnLoad = true;
    return current;
  }

  function localCompareAndSet(expected: string[] | undefined, next: string[], operation?: PinOperation) {
    const state = readFallback();
    if (!state) throw new Error("Local pinned chats are unavailable");
    // A pin another tab's unpin already fenced arrived late: acknowledge it, never apply it.
    const replayId = operation && (operation.serverId ?? operation.operationId);
    if (replayId && state.operations.some((entry) => entry.cancels?.includes(replayId)))
      return { pinnedIds: state.pinnedIds, applied: true };
    const existing = operation && state.operations.some((entry) => entry.operationId === operation.operationId);
    if (!existing && !sameIds(expected, state.pinnedIds))
      return { pinnedIds: state.pinnedIds, applied: false };
    if (!existing && !operation && !sameIds(next, state.pinnedIds))
      throw new PinnedChatsValidationError("A local pin change requires an operation");
    const operations = existing || !operation
      ? state.operations : [...state.operations, validatePinOperation(operation)];
    if (operations.length > 10_000)
      throw new PinnedChatsValidationError("Too many pending pin changes");
    const pinnedIds = existing || !operation
      ? state.pinnedIds : validatePinnedIds(applyPinOperation(state.pinnedIds, operation))!;
    options.writeFallback({ ...state, pinnedIds, operations });
    return { pinnedIds, applied: true };
  }

  return {
    async load(): Promise<string[] | undefined> {
      const epoch = generation;
      recoveredOnLoad = false;
      const state = readFallback();
      try {
        const current = validatePinnedIds(await options.load());
        assertCurrent(epoch);
        if (current !== undefined) lastRemote = current.slice();
        if (state) return await recover(state, current, epoch);
        return current;
      } catch (error) {
        assertCurrent(epoch);
        const transient = error instanceof TypeError ||
          (error as { unslothTransportFailure?: unknown })?.unslothTransportFailure === true ||
          (error instanceof Error && ["AbortError", "TimeoutError"].includes(error.name)) ||
          (error instanceof ChatSettingsRequestError &&
            (error.status >= 500 || error.status === 429 || error.status === 408));
        // A refused replay keeps the saved view, so a later unpin can compact that pin away.
        if (state && (transient || error instanceof ReplayConflictError)) return state.pinnedIds;
        return fallback(error, epoch).pinnedIds;
      }
    },
    async migrateLegacy(ids: string[]): Promise<string[] | undefined> {
      const epoch = generation;
      const state = readFallback();
      if (state) return state.pinnedIds;
      // load() just replayed the baseline and offline changes; reimporting would duplicate them.
      if (recoveredOnLoad) return lastRemote?.slice();
      try {
        const current = await remoteMigration(validatePinnedIds(ids)!, lastRemote, epoch);
        assertCurrent(epoch);
        lastRemote = current?.slice();
        return current;
      } catch (error) {
        return fallback(error, epoch).pinnedIds;
      }
    },
    async compareAndSet(expected: string[] | undefined, next: string[], operation?: PinOperation) {
      validatePinnedIds(expected);
      validatePinnedIds(next);
      if (operation) validatePinOperation(operation);
      if (readFallback()) return localCompareAndSet(expected, next, operation);
      const epoch = generation;
      try {
        const result = validateResult(await options.compareAndSet(expected, next, operation));
        assertCurrent(epoch);
        lastRemote = result.pinnedIds?.slice();
        return result;
      } catch (error) {
        fallback(error, epoch, expected);
        return localCompareAndSet(expected, next, operation);
      }
    },
    reset(): void {
      generation += 1;
      localOnly = false;
      lastRemote = undefined;
      recoveredOnLoad = false;
    },
  };
}
