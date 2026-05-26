// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const srcRoot = resolve(import.meta.dirname, "../src");
const fetchSymbol = Symbol.for("unsloth.deviceInventoryTestFetch");

function dataModule(source: string): string {
  return `data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`;
}

const mockModules = new Map<string, string>([
  [
    resolve(srcRoot, "features/inventory/api.ts"),
    dataModule(`
      const fetchSymbol = Symbol.for("unsloth.deviceInventoryTestFetch");
      function fetchSource(source, token) {
        return globalThis[fetchSymbol](source, token);
      }
      export const listCachedGguf = (token) => fetchSource("cachedGguf", token);
      export const listCachedModels = (token) => fetchSource("cachedModels", token);
      export const listCachedDatasets = () => fetchSource("cachedDatasets");
      export const listLocalModels = () => fetchSource("localModels");
      export const listLocalDatasets = () => fetchSource("localDatasets");
    `),
  ],
  [
    "@/hooks/use-debounced-value",
    dataModule(`
      export function useDebouncedValue(value) { return value; }
    `),
  ],
  [
    "@/stores/inventory-events",
    dataModule(`
      export function useInventoryVersion() { return 0; }
    `),
  ],
]);

function resolveLocalModule(base: string, specifier: string): string | null {
  const target = resolve(base, specifier);
  const candidates = [
    target,
    `${target}.ts`,
    `${target}.tsx`,
    resolve(target, "index.ts"),
    resolve(target, "index.tsx"),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return pathToFileURL(candidate).href;
    }
  }
  return null;
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith("@/")) {
      const localPath = resolve(srcRoot, specifier.slice(2));
      const mock = mockModules.get(localPath) ?? mockModules.get(specifier);
      if (mock) return { shortCircuit: true, url: mock };
      const url = resolveLocalModule(srcRoot, specifier.slice(2));
      if (url) return { shortCircuit: true, url };
    }
    if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
      const localPath = resolve(
        dirname(fileURLToPath(context.parentURL)),
        specifier,
      );
      const candidates = [
        localPath,
        `${localPath}.ts`,
        `${localPath}.tsx`,
        resolve(localPath, "index.ts"),
        resolve(localPath, "index.tsx"),
      ];
      for (const candidate of candidates) {
        const mock = mockModules.get(candidate);
        if (mock) return { shortCircuit: true, url: mock };
      }
      const url = resolveLocalModule(
        dirname(fileURLToPath(context.parentURL)),
        specifier,
      );
      if (url) return { shortCircuit: true, url };
    }
    return nextResolve(specifier, context);
  },
});

type PendingFetch = {
  source: string;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
};

function emptySource() {
  return { rows: [], loading: false, ready: false, error: null, key: null };
}

test("forced inventory refresh waits for an in-flight request and re-fetches", async () => {
  const pending: PendingFetch[] = [];
  Object.defineProperty(globalThis, fetchSymbol, {
    value(source: string) {
      return new Promise((resolveRequest, rejectRequest) => {
        pending.push({
          source,
          resolve: resolveRequest,
          reject: rejectRequest,
        });
      });
    },
    configurable: true,
  });

  const {
    fetchInventorySource,
    useDeviceInventoryStore,
  } = await import("../src/features/inventory/use-device-inventory.ts");

  useDeviceInventoryStore.setState({
    cachedGguf: emptySource(),
    cachedModels: emptySource(),
    cachedDatasets: emptySource(),
    localModels: emptySource(),
    localDatasets: emptySource(),
  });

  const first = fetchInventorySource("cachedDatasets", {
    inventoryVersion: 1,
  });
  const forced = fetchInventorySource("cachedDatasets", {
    inventoryVersion: 1,
    force: true,
  });

  assert.equal(pending.length, 1);
  assert.equal(pending[0].source, "cachedDatasets");
  pending[0].resolve([{ repo_id: "Org/Stale", size_bytes: 1 }]);
  assert.deepEqual(await first, [{ repo_id: "Org/Stale", size_bytes: 1 }]);

  await Promise.resolve();
  assert.equal(pending.length, 2);
  assert.equal(pending[1].source, "cachedDatasets");
  pending[1].resolve([{ repo_id: "Org/Fresh", size_bytes: 2 }]);

  assert.deepEqual(await forced, [{ repo_id: "Org/Fresh", size_bytes: 2 }]);
  assert.deepEqual(useDeviceInventoryStore.getState().cachedDatasets.rows, [
    { repo_id: "Org/Fresh", size_bytes: 2 },
  ]);
});
