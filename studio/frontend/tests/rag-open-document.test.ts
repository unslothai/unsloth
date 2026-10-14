// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type * as OpenModule from "../src/features/rag/lib/open-document.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const LIMIT = 100;
const toasts: string[] = [];
const opened: { name: string; size: number; page?: number }[] = [];

const { openRagDocumentInBrowser, openRagDocumentExternally } = loadWithStubs<typeof OpenModule>(
  new URL("../src/features/rag/lib/open-document.ts", import.meta.url),
  {
    "@/components/file-viewer": { MAX_DOCUMENT_PREVIEW_BYTES: LIMIT },
    "@/features/browser": {
      openFileInBrowser: ({ name, blob, page }: { name: string; blob: Blob; page?: number }) =>
        opened.push({ name, size: blob.size, ...(page ? { page } : {}) }),
    },
    "@/lib/api-base": { isTauri: false },
    "@/lib/toast": {
      toast: {
        error: (title: string) => toasts.push(`error:${title}`),
        message: (title: string) => toasts.push(`message:${title}`),
      },
    },
    "../api/rag-api": { getDocumentFileUrl: async () => "http://studio.test/file" },
  },
);

const settle = () => new Promise((resolve) => setTimeout(resolve, 10));

function serve(size: number, contentLength = String(size)) {
  let cancelled = false;
  globalThis.fetch = (async () => ({
    ok: true,
    headers: new Headers(contentLength ? { "content-length": contentLength } : {}),
    body: { cancel: async () => void (cancelled = true) },
    blob: async () => new Blob([new Uint8Array(size)]),
  })) as unknown as typeof fetch;
  return () => cancelled;
}

test("a file within the preview limit opens in the browser", async () => {
  toasts.length = 0;
  opened.length = 0;
  serve(LIMIT);
  openRagDocumentInBrowser("doc-1", "handbook.pdf");
  await settle();
  assert.deepEqual(opened, [{ name: "handbook.pdf", size: LIMIT }]);
  assert.deepEqual(toasts, []);
});

test("a cited page reaches the browser's file view", async () => {
  toasts.length = 0;
  opened.length = 0;
  serve(LIMIT);
  openRagDocumentInBrowser("doc-1", "handbook.pdf", 3);
  await settle();
  assert.deepEqual(opened, [{ name: "handbook.pdf", size: LIMIT, page: 3 }]);
});

test("two quick clicks on one document open the later one, whichever download finishes first", async () => {
  toasts.length = 0;
  opened.length = 0;
  const delays = [40, 5];
  globalThis.fetch = (async () => {
    await new Promise((resolve) => setTimeout(resolve, delays.shift()));
    return { ok: true, headers: new Headers(), blob: async () => new Blob([new Uint8Array(LIMIT)]) };
  }) as unknown as typeof fetch;
  openRagDocumentInBrowser("doc-1", "handbook.pdf", 3);
  openRagDocumentInBrowser("doc-1", "handbook.pdf", 5);
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.deepEqual(opened, [{ name: "handbook.pdf", size: LIMIT, page: 5 }]);
});

test("a superseded click that fails says nothing once a later click opened the file", async () => {
  toasts.length = 0;
  opened.length = 0;
  const outcomes = [{ delay: 40, ok: false }, { delay: 5, ok: true }];
  globalThis.fetch = (async () => {
    const { delay, ok } = outcomes.shift() ?? { delay: 0, ok: false };
    await new Promise((resolve) => setTimeout(resolve, delay));
    if (!ok) throw new Error("network");
    return { ok: true, headers: new Headers(), blob: async () => new Blob([new Uint8Array(LIMIT)]) };
  }) as unknown as typeof fetch;
  openRagDocumentInBrowser("doc-1", "handbook.pdf", 3);
  openRagDocumentInBrowser("doc-1", "handbook.pdf", 5);
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.deepEqual(opened, [{ name: "handbook.pdf", size: LIMIT, page: 5 }]);
  assert.deepEqual(toasts, []);
});

test("past the limit the body is never buffered and the file is offered outside Studio", async () => {
  toasts.length = 0;
  opened.length = 0;
  const cancelled = serve(LIMIT + 1);
  openRagDocumentInBrowser("doc-1", "scan.pdf");
  await settle();
  assert.equal(cancelled(), true);
  assert.deepEqual(opened, []);
  assert.deepEqual(toasts, ["message:scan.pdf is too large to preview in Studio"]);
  // No Content-Length: the size is still checked once read.
  toasts.length = 0;
  serve(LIMIT + 1, "");
  openRagDocumentInBrowser("doc-1", "scan.pdf");
  await settle();
  assert.deepEqual(opened, []);
  assert.equal(toasts.length, 1);
});

test("a blocked new tab says so instead of doing nothing", () => {
  toasts.length = 0;
  (globalThis as { window?: unknown }).window = { open: () => null };
  try {
    openRagDocumentExternally("doc-1", "book.epub");
  } finally {
    (globalThis as { window?: unknown }).window = undefined;
  }
  assert.deepEqual(toasts, ["error:Your browser blocked the new tab for book.epub"]);
});
