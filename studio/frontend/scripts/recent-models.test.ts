// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  removeRecentModel,
  touchRecentModel,
} from "../src/features/chat/model-config/recent-models.ts";

const STORAGE_KEY = "unsloth_recent_models";

class MemoryStorage implements Storage {
  private readonly map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }

  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }

  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

function installStorage(): MemoryStorage {
  const storage = new MemoryStorage();
  Object.defineProperty(globalThis, "window", {
    value: {},
    configurable: true,
  });
  Object.defineProperty(globalThis, "localStorage", {
    value: storage,
    configurable: true,
  });
  return storage;
}

function variants(storage: Storage): string[] {
  const raw = storage.getItem(STORAGE_KEY);
  assert.ok(raw);
  const parsed = JSON.parse(raw) as Array<{ id: string; variant: string }>;
  return parsed.map((entry) => entry.variant).sort();
}

test("removeRecentModel all removes every variant for a model", () => {
  const storage = installStorage();
  touchRecentModel({ id: "Org/Foo", ggufVariant: null });
  touchRecentModel({ id: "Org/Foo", ggufVariant: "Q4_K_M" });
  touchRecentModel({ id: "Org/Bar", ggufVariant: null });

  removeRecentModel("org/foo", "all");

  const raw = storage.getItem(STORAGE_KEY);
  assert.ok(raw);
  const parsed = JSON.parse(raw) as Array<{
    id: string;
    variant: string;
    ts: number;
  }>;
  assert.equal(parsed.length, 1);
  assert.equal(parsed[0].id, "org/bar");
  assert.equal(parsed[0].variant, "");
  assert.equal(Number.isFinite(parsed[0].ts), true);
});

test("removeRecentModel variant null removes only the safetensors entry", () => {
  const storage = installStorage();
  touchRecentModel({ id: "Org/Foo", ggufVariant: null });
  touchRecentModel({ id: "Org/Foo", ggufVariant: "Q4_K_M" });

  removeRecentModel("Org/Foo", { variant: null });

  assert.deepEqual(variants(storage), ["Q4_K_M"]);
});

test("removeRecentModel variant removes only the matching GGUF entry", () => {
  const storage = installStorage();
  touchRecentModel({ id: "Org/Foo", ggufVariant: null });
  touchRecentModel({ id: "Org/Foo", ggufVariant: "Q4_K_M" });
  touchRecentModel({ id: "Org/Foo", ggufVariant: "Q8_0" });

  removeRecentModel("Org/Foo", { variant: "q4_k_m" });

  assert.deepEqual(variants(storage), ["", "Q8_0"]);
});
