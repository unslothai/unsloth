// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const onDisk = new Set<string>();
const calls: string[] = [];
(globalThis as { window?: unknown }).window = {
  __TAURI_INTERNALS__: {
    invoke: async (command: string, args: { id?: string }) => {
      calls.push(command);
      if (command === "browser_file_local_copy") {
        onDisk.add("copy");
        return { id: "copy", path: "/cache/open-with/copy/clip.mp4" };
      }
      if (!onDisk.has(args.id ?? "")) throw "The file is no longer there.";
    },
  },
};

const { openLocalCopy, withLocalCopy } = await import("../src/features/browser/open-with.ts");

test("a copy moved away after Show in Finder is written again, so Open still opens the tab", async () => {
  const blob = new Blob(["video"]);
  const open = () => withLocalCopy(blob, "clip.mp4", (copy) => openLocalCopy(copy.id));
  await open();
  await open();
  assert.deepEqual(calls, ["browser_file_local_copy", "browser_file_open", "browser_file_open"]);
  onDisk.clear();
  await open();
  assert.deepEqual(calls.slice(3), ["browser_file_open", "browser_file_local_copy", "browser_file_open"]);
});
