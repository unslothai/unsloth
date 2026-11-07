// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";
import type { BrowserTab } from "../src/features/browser/store.ts";

register("./helpers/browser-capture-resolver.mjs", import.meta.url);
const { captureHiddenPage, screenshotPage } = await import("../src/features/browser/capture.ts");
const tab = (id: string) => ({ id, index: 0, history: [{ kind: "web", url: `https://${id}.example/` }] }) as BrowserTab;
const tick = () => new Promise((resolve) => setImmediate(resolve));

test("observer/tool captures serialize per page while other chats capture concurrently", async () => {
  const calls: string[] = [];
  const pending: { tabId: string; resolve: (value: ArrayBuffer) => void }[] = [];
  (globalThis as Record<string, unknown>).captureNativeCall = (command: string, args: { tabId: string }) => {
    assert.equal(command, "browser_capture");
    calls.push(args.tabId);
    return new Promise<ArrayBuffer>((resolve) => pending.push({ tabId: args.tabId, resolve }));
  };
  const a = captureHiddenPage(tab("a"));
  const aTool = screenshotPage(tab("a"), {} as HTMLElement);
  const b = captureHiddenPage(tab("b"));
  await tick();
  assert.deepEqual(calls, ["a", "b"]);
  pending.find((item) => item.tabId === "a")!.resolve(new Uint8Array([1]).buffer);
  assert.equal((await a)?.size, 1);
  await tick();
  assert.deepEqual(calls, ["a", "b", "a"]);
  pending.find((item) => item.tabId === "b")!.resolve(new Uint8Array([2]).buffer);
  pending.at(-1)!.resolve(new Uint8Array([3]).buffer);
  assert.deepEqual(await Promise.all([aTool, b]).then((values) => values.map((value) => value?.size)), [1, 1]);
});

test("a failed capture releases its page queue", async () => {
  let calls = 0;
  (globalThis as Record<string, unknown>).captureNativeCall = async () => {
    if (++calls === 1) throw new Error("capture failed");
    return new Uint8Array([1]).buffer;
  };
  const failed = captureHiddenPage(tab("failure"));
  const next = captureHiddenPage(tab("failure"));
  await assert.rejects(failed, /capture failed/);
  assert.equal((await next)?.size, 1);
  assert.equal(calls, 2);
});
