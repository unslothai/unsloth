// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const srcRoot = resolve(import.meta.dirname, "../src");
const fetchSymbol = Symbol.for("unsloth.downloadJobsApiTestFetch");

function dataModule(source: string): string {
  return `data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`;
}

const mockModules = new Map<string, string>([
  [
    "@/features/auth",
    dataModule(`
      const fetchSymbol = Symbol.for("unsloth.downloadJobsApiTestFetch");
      export function hfTokenHeader(token) {
        return token ? { "X-HF-Token": token } : {};
      }
      export async function authFetch(input, init) {
        return globalThis[fetchSymbol](input, init);
      }
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

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function headerValue(
  init: RequestInit | undefined,
  name: string,
): string | null {
  const headers = init?.headers;
  if (!headers) return null;
  if (headers instanceof Headers) return headers.get(name);
  if (Array.isArray(headers)) {
    const pair = headers.find(
      ([key]) => key.toLowerCase() === name.toLowerCase(),
    );
    return pair?.[1] ?? null;
  }
  return (headers as Record<string, string>)[name] ?? null;
}

test("progress APIs forward token, expected bytes, signal, and query params", async () => {
  const requests: Array<{ url: URL; init: RequestInit | undefined }> = [];
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    requests.push({ url: new URL(raw, "http://studio.test"), init });
    return Promise.resolve(
      jsonResponse({
        downloaded_bytes: 12,
        expected_bytes: 24,
        progress: 0.5,
        cache_path: null,
      }),
    );
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const {
    getDatasetDownloadProgress,
    getDownloadProgress,
    getGgufDownloadProgress,
  } = await import("../src/features/download-jobs/api.ts");

  const modelAbort = new AbortController();
  const datasetAbort = new AbortController();
  const ggufAbort = new AbortController();

  await getDownloadProgress("Org/Model", {
    hfToken: "hf-model",
    expectedBytes: 123,
    signal: modelAbort.signal,
  });
  await getDatasetDownloadProgress("Org/Data", {
    hfToken: "hf-dataset",
    expectedBytes: 456,
    signal: datasetAbort.signal,
  });
  await getGgufDownloadProgress("Org/GGUF", {
    variant: "Q4_K_M",
    hfToken: "hf-gguf",
    expectedBytes: 789,
    signal: ggufAbort.signal,
  });

  assert.equal(requests.length, 3);

  assert.equal(requests[0].url.pathname, "/api/models/download-progress");
  assert.equal(requests[0].url.searchParams.get("repo_id"), "Org/Model");
  assert.equal(requests[0].url.searchParams.get("expected_bytes"), "123");
  assert.equal(headerValue(requests[0].init, "X-HF-Token"), "hf-model");
  assert.equal(requests[0].init?.signal, modelAbort.signal);

  assert.equal(requests[1].url.pathname, "/api/datasets/download-progress");
  assert.equal(requests[1].url.searchParams.get("repo_id"), "Org/Data");
  assert.equal(requests[1].url.searchParams.get("expected_bytes"), "456");
  assert.equal(headerValue(requests[1].init, "X-HF-Token"), "hf-dataset");
  assert.equal(requests[1].init?.signal, datasetAbort.signal);

  assert.equal(requests[2].url.pathname, "/api/models/gguf-download-progress");
  assert.equal(requests[2].url.searchParams.get("repo_id"), "Org/GGUF");
  assert.equal(requests[2].url.searchParams.get("variant"), "Q4_K_M");
  assert.equal(requests[2].url.searchParams.get("expected_bytes"), "789");
  assert.equal(headerValue(requests[2].init, "X-HF-Token"), "hf-gguf");
  assert.equal(requests[2].init?.signal, ggufAbort.signal);
});
