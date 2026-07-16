// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const LEGACY_OPEN_TIMEOUT_MS = 3_000;

export class LegacyDatabaseUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LegacyDatabaseUnavailableError";
  }
}

async function discoverKnownDatabase(name: string): Promise<boolean | null> {
  if (typeof indexedDB === "undefined") return false;
  if (!("databases" in indexedDB)) return null;
  try {
    const databases = await indexedDB.databases();
    return databases.some((database) => database.name === name);
  } catch {
    return null;
  }
}

async function openKnownDatabase(name: string): Promise<IDBDatabase | null> {
  const discovered = await discoverKnownDatabase(name);
  if (discovered === false) return null;

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name);
    let probingMissingDatabase = false;
    const timeout = setTimeout(() => {
      request.transaction?.abort();
      reject(
        new LegacyDatabaseUnavailableError(
          `Opening legacy database ${name} timed out. Close older Studio tabs and retry.`,
        ),
      );
    }, LEGACY_OPEN_TIMEOUT_MS);
    const finish = (callback: () => void) => {
      clearTimeout(timeout);
      callback();
    };

    request.onupgradeneeded = (event) => {
      if ((event as IDBVersionChangeEvent).oldVersion === 0) {
        // Abort missing-database creation so capability probing stays read-only.
        probingMissingDatabase = true;
        request.transaction?.abort();
      }
    };
    request.onblocked = () =>
      finish(() =>
        reject(
          new LegacyDatabaseUnavailableError(
            `Legacy database ${name} is blocked by another tab. Close it and retry.`,
          ),
        ),
      );
    request.onerror = () =>
      finish(() => {
        if (probingMissingDatabase && request.error?.name === "AbortError") {
          resolve(null);
          return;
        }
        reject(
          request.error ??
            new LegacyDatabaseUnavailableError(
              `Could not open legacy database ${name}.`,
            ),
        );
      });
    request.onsuccess = () => finish(() => resolve(request.result));
  });
}

export async function readLegacyStorePage<T extends { id: string }>(input: {
  databaseName: string;
  storeName: string;
  cursor: string | null;
  limit: number;
}): Promise<{ items: T[]; nextCursor: string | null }> {
  const database = await openKnownDatabase(input.databaseName);
  if (!database) return { items: [], nextCursor: null };
  try {
    if (!database.objectStoreNames.contains(input.storeName)) {
      return { items: [], nextCursor: null };
    }
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(input.storeName, "readonly");
      const store = transaction.objectStore(input.storeName);
      const range = input.cursor
        ? IDBKeyRange.lowerBound(input.cursor, true)
        : undefined;
      const request = store.openCursor(range, "next");
      const rows: T[] = [];
      const timeout = setTimeout(() => {
        transaction.abort();
        reject(
          new LegacyDatabaseUnavailableError(
            `Reading legacy database ${input.databaseName} timed out. Close older Studio tabs and retry.`,
          ),
        );
      }, LEGACY_OPEN_TIMEOUT_MS);
      const finish = (callback: () => void) => {
        clearTimeout(timeout);
        callback();
      };
      request.onerror = () =>
        finish(() => reject(request.error ?? new Error("Legacy cursor failed.")));
      transaction.onerror = () =>
        finish(() =>
          reject(transaction.error ?? new Error("Legacy transaction failed.")),
        );
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor || rows.length >= input.limit + 1) {
          finish(() => {
            const items = rows.slice(0, input.limit);
            resolve({
              items,
              nextCursor:
                rows.length > input.limit ? (items.at(-1)?.id ?? null) : null,
            });
          });
          return;
        }
        rows.push(cursor.value as T);
        cursor.continue();
      };
    });
  } finally {
    database.close();
  }
}
