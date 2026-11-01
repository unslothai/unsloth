// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Studio's CSP (backend and tauri.conf.json) has no data: in connect-src, so fetch() on a sent
// attachment's data URL is refused outside `tauri dev`: "Could not open clip.mp4" (#16248).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type * as OpenModule from "../src/components/assistant-ui/attachment-browser-open.tsx";
import type * as NativeFilesModule from "../src/lib/native-files.ts";
import * as dataUri from "../src/lib/data-uri.ts";
import { loadWithStubs, type StubElement, stubJsxRuntime } from "./helpers/module-stubs.ts";

const CLIP = Uint8Array.of(0, 0, 0, 32, 102, 116, 121, 112);
const CLIP_B64 = Buffer.from(CLIP).toString("base64");

function load() {
  const fetched: string[] = [];
  const opened: { blob: Blob; name: string }[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = ((input: unknown) => {
    fetched.push(String(input));
    return Promise.reject(new TypeError("Failed to fetch"));
  }) as typeof fetch;
  const nativeFiles = loadWithStubs<typeof NativeFilesModule>(new URL("../src/lib/native-files.ts", import.meta.url), {
    "@/lib/api-base": { isTauri: false },
    "@/lib/data-uri": dataUri,
    "@tauri-apps/api/core": {},
  });
  const mod = loadWithStubs<typeof OpenModule>(
    new URL("../src/components/assistant-ui/attachment-browser-open.tsx", import.meta.url),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: { useCallback: (fn: unknown) => fn, useLayoutEffect: () => {}, useRef: (v: unknown) => ({ current: v }) },
      "@/features/auth": { authFetch: () => Promise.reject(new Error("authFetch")) },
      "@/features/chat": {},
      "@/features/browser": { openFileInBrowser: (o: { blob: Blob; name: string }) => opened.push(o) },
      "@/lib/api-base": { isStudioUrl: () => false },
      "@/lib/data-uri": dataUri,
      "@/lib/native-files": nativeFiles,
      "@/lib/toast": { toast: { error: (m: string) => opened.push({ blob: new Blob(), name: `toast:${m}` }) } },
      "@assistant-ui/react": { useAuiState: () => "id" },
      "radix-ui": { Slot: { Root: "Slot" } },
      "./attachment-browser-open-context": { AttachmentBrowserOpenContext: { Provider: "Provider" } },
      "./link-context-menu": { FileContextMenu: "FileContextMenu" },
    },
  );
  return { mod, fetched, opened, restore: () => (globalThis.fetch = realFetch) };
}

const source = (over: Record<string, unknown>) =>
  ({
    kind: "video",
    name: "clip.mp4",
    contentType: "video/mp4",
    file: undefined,
    src: undefined,
    audio: undefined,
    video: undefined,
    text: undefined,
    hasOriginal: false,
    ...over,
  }) as Parameters<typeof OpenModule.AttachmentBrowserOpenProvider>[0]["source"];

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

for (const data of [CLIP_B64, `data:video/mp4;base64,${CLIP_B64}`]) {
  test(`a sent video opens without fetching its data URL (${data.slice(0, 5)})`, async () => {
    const { mod, fetched, opened, restore } = load();
    try {
      const element = mod.AttachmentBrowserOpenProvider({
        source: source({ video: { data, mimeType: "video/mp4" } }),
        children: null,
      }) as unknown as StubElement;
      (element.props.open as () => void)();
      await flush();
      assert.deepEqual(fetched, []);
      assert.equal(opened.length, 1);
      assert.equal(opened[0].name, "clip.mp4");
      assert.equal(opened[0].blob.type, "video/mp4");
      assert.deepEqual(new Uint8Array(await opened[0].blob.arrayBuffer()), CLIP);
    } finally {
      restore();
    }
  });
}

test("a sent image's context menu loads its data URL without fetching", async () => {
  const { mod, fetched, restore } = load();
  try {
    const element = mod.AttachmentFileContextMenu({
      source: source({ kind: "image", name: "a.png", contentType: "image/png", src: `data:image/png;base64,${CLIP_B64}` }),
      children: null as never,
    }) as unknown as StubElement;
    const blob = await (element.props.load as () => Promise<Blob>)();
    assert.deepEqual(fetched, []);
    assert.equal(blob.type, "image/png");
    assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), CLIP);
  } finally {
    restore();
  }
});

test("attachment viewers never fetch() a sent attachment's data URL", () => {
  const preview = readFileSync(new URL("../src/components/assistant-ui/attachment-preview.tsx", import.meta.url), "utf8");
  assert.ok(!/\bfetch\(/.test(preview), "attachment-preview.tsx calls fetch(); load data URLs with urlToBlob");
});
