// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Invoke = (
  command: string,
  args?: unknown,
  options?: { headers?: Record<string, string> },
) => Promise<unknown>;

const DISK_FULL = /disk full/;

type NativeFiles = {
  downloadBlobStreaming: (content: Blob, filename: string) => Promise<void>;
  isDownloadCancelled: (error: unknown) => boolean;
};

function loadNativeFiles(invoke: Invoke): NativeFiles {
  return loadWithStubs<NativeFiles>(
    new URL("../src/lib/native-files.ts", import.meta.url),
    {
      "@/lib/api-base": { isTauri: true },
      "@/lib/data-uri": {
        decodeDataUri: () => {
          throw new Error("unexpected data URI decode");
        },
        isDataUri: () => false,
      },
      "@tauri-apps/api/core": { invoke },
    },
  );
}

test("a native Blob save crosses IPC in bounded chunks", async () => {
  const chunkSizes: number[] = [];
  const headers: Array<Record<string, string> | undefined> = [];
  const commands: string[] = [];
  const nativeFiles = loadNativeFiles((command, args, options) => {
    commands.push(command);
    if (command === "begin_native_file_save") {
      assert.deepEqual(args, { fileName: "stems.zip" });
      return Promise.resolve("opaque-save-token");
    }
    if (command === "append_native_file_save_chunk") {
      assert.ok(args instanceof Uint8Array);
      chunkSizes.push(args.byteLength);
      headers.push(options?.headers);
      return Promise.resolve(undefined);
    }
    if (command === "finish_native_file_save") {
      assert.deepEqual(args, { token: "opaque-save-token" });
      return Promise.resolve("stems.zip");
    }
    throw new Error(`unexpected command ${command}`);
  });

  await nativeFiles.downloadBlobStreaming(
    new Blob([new Uint8Array(8 * 1024 * 1024 + 17)]),
    "stems.zip",
  );

  assert.deepEqual(commands, [
    "begin_native_file_save",
    "append_native_file_save_chunk",
    "append_native_file_save_chunk",
    "finish_native_file_save",
  ]);
  assert.deepEqual(chunkSizes, [8 * 1024 * 1024, 17]);
  assert.deepEqual(headers, [
    { "x-unsloth-save-token": "opaque-save-token" },
    { "x-unsloth-save-token": "opaque-save-token" },
  ]);
});

test("a cancelled native Blob save has the normal cancellation error", async () => {
  const nativeFiles = loadNativeFiles((command) => {
    assert.equal(command, "begin_native_file_save");
    return Promise.resolve(null);
  });

  const error = await nativeFiles
    .downloadBlobStreaming(new Blob(), "stems.zip")
    .then(
      () => null,
      (rejection: unknown) => rejection,
    );
  assert.equal(nativeFiles.isDownloadCancelled(error), true);
});

test("a failed native Blob save drops its staged file", async () => {
  const commands: string[] = [];
  const nativeFiles = loadNativeFiles((command, args) => {
    commands.push(command);
    if (command === "begin_native_file_save") {
      return Promise.resolve("opaque-save-token");
    }
    if (command === "append_native_file_save_chunk") {
      return Promise.reject(new Error("disk full"));
    }
    if (command === "cancel_native_file_save") {
      assert.deepEqual(args, { token: "opaque-save-token" });
      return Promise.resolve(undefined);
    }
    throw new Error(`unexpected command ${command}`);
  });

  await assert.rejects(
    nativeFiles.downloadBlobStreaming(new Blob(["audio"]), "stems.zip"),
    DISK_FULL,
  );
  assert.deepEqual(commands, [
    "begin_native_file_save",
    "append_native_file_save_chunk",
    "cancel_native_file_save",
  ]);
});
