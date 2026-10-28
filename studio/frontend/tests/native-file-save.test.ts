// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type * as NativeFilesModule from "../src/lib/native-files.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Invoke = (
  command: string,
  args?: unknown,
  options?: { headers?: Record<string, string> },
) => Promise<unknown>;

function loadNativeFiles(
  isTauri: boolean,
  invoke: Invoke,
): typeof NativeFilesModule {
  return loadWithStubs<typeof NativeFilesModule>(
    new URL("../src/lib/native-files.ts", import.meta.url),
    {
      "@/lib/api-base": { isTauri },
      "@/lib/data-uri": {},
      "@tauri-apps/api/core": { invoke },
    },
  );
}

for (const savedFileName of [
  "renamed-image.png",
  "suggested.png",
  "caf\u00e9.png",
]) {
  test(`native save reports the selected basename: ${savedFileName}`, async () => {
    const bytes = Uint8Array.of(137, 80, 78, 71);
    let calls = 0;
    const files = loadNativeFiles(true, (command, args, options) => {
      calls += 1;
      assert.equal(command, "save_native_file");
      assert.deepEqual(args, bytes);
      assert.equal(
        Buffer.from(
          options?.headers?.["x-unsloth-default-name"] ?? "",
          "base64",
        ).toString("utf8"),
        "suggested.png",
      );
      return Promise.resolve(savedFileName);
    });

    const result = await files.downloadFile(
      new Blob([bytes], { type: "image/png" }),
      "suggested.png",
    );
    assert.equal(result, savedFileName);
    assert.equal(calls, 1);
  });
}

test("native cancellation rejects instead of reporting a saved filename", async () => {
  const files = loadNativeFiles(true, () => Promise.resolve(null));
  await assert.rejects(
    files.downloadFile("contents", "suggested.txt"),
    (error: unknown) => files.isDownloadCancelled(error),
  );
});

test("a failed native save propagates its error instead of reporting success", async () => {
  const failure = new Error("Disk full");
  const files = loadNativeFiles(true, () => Promise.reject(failure));
  await assert.rejects(
    files.downloadFile("contents", "suggested.txt"),
    (error: unknown) => error === failure,
  );
});

test("browser download suggests a name but does not report an unknown final filename", async (t) => {
  const originalDocument = Object.getOwnPropertyDescriptor(
    globalThis,
    "document",
  );
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  t.after(() => {
    for (const [key, descriptor] of [
      ["document", originalDocument],
      ["window", originalWindow],
    ] as const) {
      if (descriptor) {
        Object.defineProperty(globalThis, key, descriptor);
      } else {
        Reflect.deleteProperty(globalThis, key);
      }
    }
  });

  let downloadedBlob: Blob | undefined;
  let clicked = false;
  let removed = false;
  const anchor = {
    href: "",
    download: "",
    click: () => {
      clicked = true;
    },
    remove: () => {
      removed = true;
    },
  };
  const objectUrl = "blob:browser-download";
  t.mock.method(URL, "createObjectURL", (blob: Blob) => {
    downloadedBlob = blob;
    return objectUrl;
  });
  const revoke = t.mock.method(URL, "revokeObjectURL", () => undefined);
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: {
      createElement: (tag: string) => {
        assert.equal(tag, "a");
        return anchor;
      },
      body: { appendChild: (child: unknown) => assert.equal(child, anchor) },
    },
  });
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      setTimeout: (callback: () => void) => {
        callback();
        return 1;
      },
    },
  });
  const files = loadNativeFiles(false, () =>
    Promise.reject(new Error("Browser must not invoke native save")),
  );

  const result = await files.downloadFile(
    "contents",
    "suggested.txt",
    "text/plain",
  );
  assert.equal(result, undefined);
  assert.equal(anchor.download, "suggested.txt");
  assert.equal(anchor.href, objectUrl);
  assert.equal(clicked, true);
  assert.equal(removed, true);
  assert.equal(downloadedBlob?.type, "text/plain");
  assert.equal(await downloadedBlob?.text(), "contents");
  assert.deepEqual(revoke.mock.calls[0]?.arguments, [objectUrl]);
});

test("a chunked native save reports the basename the dialog wrote", async () => {
  const files = loadNativeFiles(true, (command) => {
    if (command === "begin_native_file_save") return Promise.resolve("token");
    if (command === "append_native_file_save_chunk") return Promise.resolve(undefined);
    assert.equal(command, "finish_native_file_save");
    return Promise.resolve("renamed-stems.zip");
  });
  assert.equal(
    await files.downloadBlobStreaming(new Blob(["stems"]), "stems.zip"),
    "renamed-stems.zip",
  );
});

test("a streamed URL save reports the basename the dialog wrote", async () => {
  const files = loadNativeFiles(true, (command, args) => {
    assert.equal(command, "save_native_file_from_url");
    assert.deepEqual(args, { url: "/clip.mp4", fileName: "clip.mp4" });
    return Promise.resolve("renamed-clip.mp4");
  });
  assert.equal(await files.downloadUrlStreaming("/clip.mp4", "clip.mp4"), "renamed-clip.mp4");
});

test("a native URL save reports the basename the dialog wrote", async (t) => {
  t.mock.method(globalThis, "fetch", () =>
    Promise.resolve(new Response("wav", { headers: { "content-type": "audio/wav" } })),
  );
  const files = loadWithStubs<typeof NativeFilesModule>(
    new URL("../src/lib/native-files.ts", import.meta.url),
    {
      "@/lib/api-base": { isTauri: true },
      "@/lib/data-uri": { isDataUri: () => false },
      "@tauri-apps/api/core": {
        invoke: (command: string) => {
          assert.equal(command, "save_native_file");
          return Promise.resolve("renamed-clip.wav");
        },
      },
    },
  );
  assert.equal(await files.downloadUrl("/clip.wav", "clip.wav"), "renamed-clip.wav");
});
