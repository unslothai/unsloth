// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const srcRoot = resolve(import.meta.dirname, "../src");
const fetchSymbol = Symbol.for("unsloth.ggufVariantsApiTestFetch");
const offlineSymbol = Symbol.for("unsloth.ggufVariantsApiTestOffline");
const inventoryVersionSymbol = Symbol.for(
  "unsloth.ggufVariantsApiTestInventoryVersion",
);

function dataModule(source: string): string {
  return `data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`;
}

const mockModules = new Map<string, string>([
  [
    "@/features/auth",
    dataModule(`
      const fetchSymbol = Symbol.for("unsloth.ggufVariantsApiTestFetch");
      export function hfTokenHeader(token) {
        return token ? { "X-HF-Token": token } : {};
      }
      export async function authFetch(input, init) {
        return globalThis[fetchSymbol](input, init);
      }
    `),
  ],
  [
    "@/lib/network",
    dataModule(`
      const offlineSymbol = Symbol.for("unsloth.ggufVariantsApiTestOffline");
      export function isHuggingFaceOffline() {
        return Boolean(globalThis[offlineSymbol]);
      }
    `),
  ],
  [
    "@/stores/inventory-events",
    dataModule(`
      const inventoryVersionSymbol = Symbol.for(
        "unsloth.ggufVariantsApiTestInventoryVersion",
      );
      export function getInventoryVersion() {
        return globalThis[inventoryVersionSymbol] || 0;
      }
      export function notifyInventoryChanged() {}
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
    if (existsSync(candidate)) return pathToFileURL(candidate).href;
  }
  return null;
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    const mock = mockModules.get(specifier);
    if (mock) return { shortCircuit: true, url: mock };
    if (specifier.startsWith("@/")) {
      const url = resolveLocalModule(srcRoot, specifier.slice(2));
      if (url) return { shortCircuit: true, url };
    }
    if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
      const url = resolveLocalModule(
        dirname(fileURLToPath(context.parentURL)),
        specifier,
      );
      if (url) return { shortCircuit: true, url };
    }
    return nextResolve(specifier, context);
  },
});

function variantsResponse(): Response {
  return new Response(
    JSON.stringify({
      repo_id: "Org/Model-GGUF",
      variants: [],
      has_vision: false,
      default_variant: null,
    }),
    {
      status: 200,
      headers: { "content-type": "application/json" },
    },
  );
}

test("GGUF variant API keeps remote catalog lookup when a cache path is known", async () => {
  const requests: Array<{ url: URL; init: RequestInit | undefined }> = [];
  Object.defineProperty(globalThis, offlineSymbol, {
    value: false,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, fetchSymbol, {
    value(input: string, init?: RequestInit) {
      requests.push({ url: new URL(input, "http://studio.test"), init });
      return Promise.resolve(variantsResponse());
    },
    configurable: true,
  });

  const { listGgufVariants, invalidateGgufVariantsCache } = await import(
    "../src/features/inventory/api.ts"
  );
  invalidateGgufVariantsCache();

  await listGgufVariants("Org/Model-GGUF", "hf-token", {
    preferLocalCache: false,
    localPath: "/cache/models--Org--Model-GGUF",
  });

  assert.equal(requests.length, 1);
  const params = requests[0].url.searchParams;
  assert.equal(params.get("repo_id"), "Org/Model-GGUF");
  assert.equal(params.get("local_path"), "/cache/models--Org--Model-GGUF");
  assert.equal(params.has("prefer_local_cache"), false);
  assert.equal(params.has("offline"), false);
});

test("GGUF variant API switches to local lookup while Hugging Face is offline", async () => {
  const requests: URL[] = [];
  Object.defineProperty(globalThis, offlineSymbol, {
    value: true,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, fetchSymbol, {
    value(input: string) {
      requests.push(new URL(input, "http://studio.test"));
      return Promise.resolve(variantsResponse());
    },
    configurable: true,
  });

  const { listGgufVariants, invalidateGgufVariantsCache } = await import(
    "../src/features/inventory/api.ts"
  );
  invalidateGgufVariantsCache();

  await listGgufVariants("Org/Model-GGUF", "hf-token", {
    preferLocalCache: false,
    localPath: "/cache/models--Org--Model-GGUF",
  });

  assert.equal(requests.length, 1);
  const params = requests[0].searchParams;
  assert.equal(params.get("prefer_local_cache"), "true");
  assert.equal(params.get("offline"), "true");
});

test("GGUF variant API cache is invalidated by repo instead of global inventory version", async () => {
  const requests: URL[] = [];
  Object.defineProperty(globalThis, offlineSymbol, {
    value: false,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, inventoryVersionSymbol, {
    value: 1,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, fetchSymbol, {
    value(input: string) {
      requests.push(new URL(input, "http://studio.test"));
      return Promise.resolve(variantsResponse());
    },
    configurable: true,
  });

  const { listGgufVariants, invalidateGgufVariantsCache } = await import(
    "../src/features/inventory/api.ts"
  );
  invalidateGgufVariantsCache();

  await listGgufVariants("Org/Model-GGUF", "hf-token", {
    preferLocalCache: true,
  });
  await listGgufVariants("Org/Model-GGUF", "hf-token", {
    preferLocalCache: true,
  });
  assert.equal(requests.length, 1);

  Reflect.set(globalThis, inventoryVersionSymbol, 2);
  await listGgufVariants("Org/Model-GGUF", "hf-token", {
    preferLocalCache: true,
  });
  assert.equal(requests.length, 1);

  invalidateGgufVariantsCache("org/model-gguf");
  await listGgufVariants("Org/Model-GGUF", "hf-token", {
    preferLocalCache: true,
  });

  assert.equal(requests.length, 2);
});
