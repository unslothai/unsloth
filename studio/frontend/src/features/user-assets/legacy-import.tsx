// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSubjectKey, subscribeAuthSubject } from "@/features/auth";
import { toastError } from "@/shared/toast";
import {
  type ReactElement,
  useEffect,
  useRef,
  useSyncExternalStore,
} from "react";
import {
  type LegacyImportItemResult,
  type UserAssetsBootstrap,
  bootstrapUserAssets,
  importLegacyUserAssets,
} from "./api";
import { MAX_LEGACY_BATCH_JSON_BYTES } from "./persistence-policy";

const SOURCE = "recipe-indexeddb-v1";
const DEVICE_IMPORT_OWNER_KEY = "user-assets:recipe-indexeddb-v1:owner";
const DEVICE_IMPORT_LOCK_NAME = "user-assets:recipe-indexeddb-v1:claim";
const DEVICE_IMPORT_CLAIM_DB = "unsloth-user-assets-migration-claims";
const DEVICE_IMPORT_CLAIM_STORE = "claims";
const LEGACY_PAGE_SIZE = 100;
const utf8Encoder = new TextEncoder();

type LegacyPage<T> = {
  items: T[];
  nextCursor: string | null;
};

type LegacyPageReader<T> = (
  cursor?: string | null,
  limit?: number,
) => Promise<LegacyPage<T>>;

type LegacyRecipe = { id: string; revision?: number } & object;
type LegacyExecution = { id: string } & object;

type LegacyImportCoordinatorProps = {
  onImported: () => void;
  readRecipes: LegacyPageReader<LegacyRecipe>;
  readExecutions: LegacyPageReader<LegacyExecution>;
};

export type LegacyImportOptions = Omit<
  LegacyImportCoordinatorProps,
  "onImported"
> & {
  signal: AbortSignal;
  expectedSubjectKey?: string;
  onProgress?: () => void;
};

type LegacyBatchPlan<T> = {
  batches: T[][];
  rejected: LegacyImportItemResult[];
};

function withoutRevision(recipe: LegacyRecipe) {
  return { ...recipe, revision: undefined };
}

function splitByLegacyBatchBytes<T extends { id: string }>(
  items: T[],
  kind: "recipes" | "executions",
  confirmSubject: string,
): LegacyBatchPlan<T> {
  const batches: T[][] = [];
  const rejected: LegacyImportItemResult[] = [];
  let current: T[] = [];
  const emptyEnvelope = {
    source: SOURCE,
    confirmSubject,
    recipes: [] as T[],
    executions: [] as T[],
  };
  const baseBytes = utf8Encoder.encode(
    JSON.stringify(emptyEnvelope),
  ).byteLength;
  let currentBytes = baseBytes;

  for (const item of items) {
    const itemBytes = utf8Encoder.encode(JSON.stringify(item)).byteLength;
    const singleItemBytes = baseBytes + itemBytes;
    if (singleItemBytes > MAX_LEGACY_BATCH_JSON_BYTES) {
      rejected.push({
        id: item.id,
        outcome: "rejected",
        reason: "legacy_batch_limit_exceeded",
      });
      continue;
    }
    const addedBytes = itemBytes + (current.length > 0 ? 1 : 0);
    if (
      current.length > 0 &&
      currentBytes + addedBytes > MAX_LEGACY_BATCH_JSON_BYTES
    ) {
      batches.push(current);
      current = [item];
      currentBytes = singleItemBytes;
    } else {
      current.push(item);
      currentBytes += addedBytes;
    }
  }
  if (current.length > 0) {
    batches.push(current);
  }

  for (const batch of batches) {
    const envelope = {
      ...emptyEnvelope,
      recipes: kind === "recipes" ? batch : [],
      executions: kind === "executions" ? batch : [],
    };
    if (
      utf8Encoder.encode(JSON.stringify(envelope)).byteLength >
      MAX_LEGACY_BATCH_JSON_BYTES
    ) {
      throw new Error(
        "Legacy import batch accounting exceeded its byte limit.",
      );
    }
  }
  return { batches, rejected };
}

function reportRejectedItems(
  kind: "recipes" | "executions",
  rejected: LegacyImportItemResult[],
): void {
  if (rejected.length === 0) return;
  console.error(`Legacy ${kind} import rejected items.`, rejected);
}

function reportRedactedItems(
  kind: "recipes" | "executions",
  redacted: LegacyImportItemResult[],
): void {
  if (redacted.length === 0) return;
  console.warn(
    `Legacy ${kind} import removed saved credentials. Configure replacement environment variables before running them.`,
    redacted,
  );
  toastError(
    "Credentials removed from browser migration",
    `${redacted.length} ${kind === "recipes" ? "recipe" : "execution"}${redacted.length === 1 ? " had" : "s had"} saved credentials removed. Configure replacement environment variables before running them.`,
  );
}

function rejectedItems(
  results: LegacyImportItemResult[],
): LegacyImportItemResult[] {
  return results.filter((item) => item.outcome === "rejected");
}

async function claimLegacyBrowserData(owner: string): Promise<boolean> {
  if (typeof window === "undefined") return true;
  const claimDurably = async (): Promise<boolean> => {
    const legacyOwner = window.localStorage.getItem(DEVICE_IMPORT_OWNER_KEY);
    if (legacyOwner && legacyOwner !== owner) return false;
    const claimed = await claimLegacyBrowserDataWithIndexedDb(owner);
    if (claimed && !legacyOwner) {
      window.localStorage.setItem(DEVICE_IMPORT_OWNER_KEY, owner);
    }
    return claimed;
  };
  if (typeof navigator.locks === "undefined") {
    return claimDurably();
  }
  try {
    return await navigator.locks.request(
      DEVICE_IMPORT_LOCK_NAME,
      { mode: "exclusive" },
      claimDurably,
    );
  } catch {
    throw new Error("Could not confirm ownership of browser-saved recipes.");
  }
}

function openDeviceImportClaimDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DEVICE_IMPORT_CLAIM_DB, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(DEVICE_IMPORT_CLAIM_STORE)) {
        database.createObjectStore(DEVICE_IMPORT_CLAIM_STORE, {
          keyPath: "key",
        });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(
        request.error ??
          new Error("Could not open the migration claim database."),
      );
    request.onblocked = () =>
      reject(new Error("The migration claim database is blocked."));
  });
}

async function claimLegacyBrowserDataWithIndexedDb(
  owner: string,
): Promise<boolean> {
  if (typeof window.indexedDB === "undefined") {
    throw new Error("Browser storage migration ownership could not be locked.");
  }
  const database = await openDeviceImportClaimDb();
  return new Promise((resolve, reject) => {
    let claimed = false;
    const transaction = database.transaction(
      DEVICE_IMPORT_CLAIM_STORE,
      "readwrite",
    );
    transaction.oncomplete = () => {
      database.close();
      resolve(claimed);
    };
    transaction.onerror = () => {
      database.close();
      reject(
        transaction.error ??
          new Error("Could not claim browser-saved recipes."),
      );
    };
    transaction.onabort = transaction.onerror;

    const store = transaction.objectStore(DEVICE_IMPORT_CLAIM_STORE);
    const request = store.get(DEVICE_IMPORT_OWNER_KEY);
    request.onsuccess = () => {
      const existing = request.result as
        | { key: string; owner: string }
        | undefined;
      if (existing) {
        claimed = existing.owner === owner;
        return;
      }
      const addRequest = store.add({ key: DEVICE_IMPORT_OWNER_KEY, owner });
      addRequest.onsuccess = () => {
        claimed = true;
      };
    };
  });
}

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return;
  if (signal.reason instanceof Error) throw signal.reason;
  throw new DOMException("The legacy import was cancelled.", "AbortError");
}

function readableError(error: unknown): string {
  const detail =
    error instanceof Error && error.message.trim()
      ? error.message
      : "The saved browser data could not be imported.";
  const sentence = /[.!?]$/.test(detail) ? detail : `${detail}.`;
  return `${sentence} Refresh the page to retry; items already imported will not be duplicated.`;
}

async function importRecipePages(
  bootstrap: UserAssetsBootstrap,
  readRecipes: LegacyPageReader<LegacyRecipe>,
  signal: AbortSignal,
  expectedSubjectKey: string,
  onProgress?: () => void,
): Promise<number> {
  const importedIds = new Set(bootstrap.importLedger.recipes);
  let rejectedCount = 0;
  let cursor: string | null = null;
  do {
    throwIfAborted(signal);
    const page = await readRecipes(cursor, LEGACY_PAGE_SIZE);
    throwIfAborted(signal);
    const recipes = page.items
      .filter((item) => !importedIds.has(item.id))
      .map(withoutRevision);
    const plan = splitByLegacyBatchBytes(recipes, "recipes", bootstrap.subject);
    reportRejectedItems("recipes", plan.rejected);
    rejectedCount += plan.rejected.length;
    for (const recipeBatch of plan.batches) {
      const result = await importLegacyUserAssets(
        {
          source: SOURCE,
          confirmSubject: bootstrap.subject,
          recipes: recipeBatch,
          executions: [],
        },
        { signal, expectedSubjectKey },
      );
      const rejected = rejectedItems(result.recipes);
      reportRejectedItems("recipes", rejected);
      rejectedCount += rejected.length;
      reportRedactedItems(
        "recipes",
        result.recipes.filter((item) => item.outcome === "redacted"),
      );
      onProgress?.();
    }
    if (cursor !== null && page.nextCursor === cursor) {
      throw new Error("Legacy recipe cursor did not advance.");
    }
    cursor = page.nextCursor;
  } while (cursor);
  return rejectedCount;
}

function effectiveRetryIds(results: LegacyImportItemResult[]): Set<string> {
  return new Set(
    results
      .filter((item) => item.outcome === "missing_parent" && item.id)
      .map((item) => item.id as string),
  );
}

async function importExecutionPages(
  bootstrap: UserAssetsBootstrap,
  readExecutions: LegacyPageReader<LegacyExecution>,
  signal: AbortSignal,
  expectedSubjectKey: string,
  onProgress?: () => void,
): Promise<number> {
  const importedIds = new Set(bootstrap.importLedger.executions);
  let rejectedCount = 0;
  let cursor: string | null = null;
  do {
    throwIfAborted(signal);
    const page = await readExecutions(cursor, LEGACY_PAGE_SIZE);
    throwIfAborted(signal);
    const executions = page.items.filter((item) => !importedIds.has(item.id));
    const plan = splitByLegacyBatchBytes(
      executions,
      "executions",
      bootstrap.subject,
    );
    reportRejectedItems("executions", plan.rejected);
    rejectedCount += plan.rejected.length;
    for (const executionBatch of plan.batches) {
      const result = await importLegacyUserAssets(
        {
          source: SOURCE,
          confirmSubject: bootstrap.subject,
          recipes: [],
          executions: executionBatch,
        },
        { signal, expectedSubjectKey },
      );
      onProgress?.();
      const rejected = rejectedItems(result.executions);
      reportRejectedItems("executions", rejected);
      rejectedCount += rejected.length;
      reportRedactedItems(
        "executions",
        result.executions.filter((item) => item.outcome === "redacted"),
      );
      const retryIds = effectiveRetryIds(result.executions);
      if (retryIds.size > 0) {
        const retryResult = await importLegacyUserAssets(
          {
            source: SOURCE,
            confirmSubject: bootstrap.subject,
            recipes: [],
            executions: executionBatch.filter((item) => retryIds.has(item.id)),
          },
          { signal, expectedSubjectKey },
        );
        const retryRejected = rejectedItems(retryResult.executions);
        reportRejectedItems("executions", retryRejected);
        rejectedCount += retryRejected.length;
        const remainingMissingParents = retryResult.executions.filter(
          (item) => item.outcome === "missing_parent",
        );
        reportRejectedItems("executions", remainingMissingParents);
        rejectedCount += remainingMissingParents.length;
        onProgress?.();
      }
    }
    if (cursor !== null && page.nextCursor === cursor) {
      throw new Error("Legacy execution cursor did not advance.");
    }
    cursor = page.nextCursor;
  } while (cursor);
  return rejectedCount;
}

// The editor also imports before declaring a legacy recipe missing.
// eslint-disable-next-line react-refresh/only-export-components
export async function importLegacyUserAssetsFromIndexedDb({
  readRecipes,
  readExecutions,
  signal,
  expectedSubjectKey = getAuthSubjectKey(),
  onProgress,
}: LegacyImportOptions): Promise<void> {
  let requestSubjectKey = expectedSubjectKey;
  const bootstrap = await bootstrapUserAssets({
    signal,
    expectedSubjectKey: requestSubjectKey,
  });
  throwIfAborted(signal);
  // Anonymous Tauri bootstrap may establish an account; bind later mutations to it.
  if (requestSubjectKey === "anonymous") {
    requestSubjectKey = getAuthSubjectKey();
  }
  if (!(await claimLegacyBrowserData(bootstrap.subject))) return;
  const rejectedRecipes = await importRecipePages(
    bootstrap,
    readRecipes,
    signal,
    requestSubjectKey,
    onProgress,
  );
  const rejectedExecutions = await importExecutionPages(
    bootstrap,
    readExecutions,
    signal,
    requestSubjectKey,
    onProgress,
  );
  const rejectedCount = rejectedRecipes + rejectedExecutions;
  if (rejectedCount > 0) {
    throw new Error(
      `${rejectedCount} browser-saved item${rejectedCount === 1 ? " could" : "s could"} not be imported. Correct or remove those items, then retry.`,
    );
  }
}

export function LegacyImportCoordinator({
  onImported,
  readRecipes,
  readExecutions,
}: LegacyImportCoordinatorProps): ReactElement | null {
  const authSubjectKey = useSyncExternalStore(
    subscribeAuthSubject,
    getAuthSubjectKey,
    getAuthSubjectKey,
  );
  const onImportedRef = useRef(onImported);
  useEffect(() => {
    onImportedRef.current = onImported;
  }, [onImported]);

  useEffect(() => {
    const controller = new AbortController();
    let madeProgress = false;
    let completed = false;

    importLegacyUserAssetsFromIndexedDb({
      readRecipes,
      readExecutions,
      signal: controller.signal,
      expectedSubjectKey: authSubjectKey,
      onProgress: () => {
        madeProgress = true;
      },
    })
      .then(() => {
        completed = true;
      })
      .catch((error: unknown) => {
        if (
          controller.signal.aborted ||
          getAuthSubjectKey() !== authSubjectKey
        ) {
          return;
        }
        toastError("Could not import browser recipes", readableError(error));
      })
      .finally(() => {
        if (
          !controller.signal.aborted &&
          getAuthSubjectKey() === authSubjectKey &&
          (completed || madeProgress)
        ) {
          onImportedRef.current();
        }
      });

    return () => controller.abort();
  }, [authSubjectKey, readExecutions, readRecipes]);

  return null;
}
