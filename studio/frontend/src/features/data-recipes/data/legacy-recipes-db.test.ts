// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import Dexie from "dexie";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { readLegacyRecipes } from "./legacy-recipes-db";

const DATABASE_NAME = "unsloth-data-recipes";

async function seedRecipes(count: number): Promise<void> {
  const db = new Dexie(DATABASE_NAME);
  db.version(1).stores({ recipes: "id, name, updatedAt, createdAt" });
  await db.table("recipes").bulkPut(
    Array.from({ length: count }, (_, index) => ({
      id: `recipe-${index}`,
      name: `Recipe ${index}`,
      payload: {},
      createdAt: index,
      updatedAt: index,
    })),
  );
  db.close();
}

describe("legacy recipe reader", () => {
  beforeEach(() => Dexie.delete(DATABASE_NAME));
  afterEach(() => {
    vi.restoreAllMocks();
    return Dexie.delete(DATABASE_NAME);
  });

  it("does not create an absent database", async () => {
    const before = await indexedDB.databases();
    await expect(readLegacyRecipes()).resolves.toEqual({
      items: [],
      nextCursor: null,
    });
    const after = await indexedDB.databases();
    expect(after.map((item) => item.name)).toEqual(
      before.map((item) => item.name),
    );
  });

  it("falls back to a non-creating known-name probe when enumeration is unavailable", async () => {
    await seedRecipes(1);
    const original = indexedDB;
    vi.stubGlobal(
      "indexedDB",
      new Proxy(original, {
        has: (target, property) =>
          property === "databases" ? false : Reflect.has(target, property),
        get: (target, property) => {
          const value = Reflect.get(target, property);
          return typeof value === "function" ? value.bind(target) : value;
        },
      }),
    );
    await expect(readLegacyRecipes()).resolves.toMatchObject({
      items: [expect.objectContaining({ id: "recipe-0" })],
    });
    vi.stubGlobal("indexedDB", original);
  });

  it("uses the same fallback when database enumeration rejects", async () => {
    await seedRecipes(1);
    vi.spyOn(indexedDB, "databases").mockRejectedValueOnce(
      new DOMException("denied", "SecurityError"),
    );
    await expect(readLegacyRecipes()).resolves.toMatchObject({
      items: [expect.objectContaining({ id: "recipe-0" })],
    });
  });

  it("reads without upgrading while a raw legacy owner remains open", async () => {
    await seedRecipes(1);
    const owner = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(DATABASE_NAME);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    await expect(readLegacyRecipes()).resolves.toMatchObject({
      items: [expect.objectContaining({ id: "recipe-0" })],
    });
    owner.close();
  });

  it("walks bounded cursor pages through all overflow rows, backfills revision, and closes", async () => {
    await seedRecipes(105);
    const close = vi.spyOn(IDBDatabase.prototype, "close");
    const first = await readLegacyRecipes(null, 100);
    expect(first.items).toHaveLength(100);
    expect(first.nextCursor).not.toBeNull();
    expect(first.items.every((record) => record.revision === 0)).toBe(true);
    const second = await readLegacyRecipes(first.nextCursor, 100);
    expect(second.items).toHaveLength(5);
    expect(second.nextCursor).toBeNull();
    expect(
      new Set([...first.items, ...second.items].map((record) => record.id))
        .size,
    ).toBe(105);
    expect(close).toHaveBeenCalledTimes(2);
  });
});
