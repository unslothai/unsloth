// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/browser-local-resolver.mjs", import.meta.url);
const { grantLocalOrigin, localOriginGranted } = await import("../src/features/browser/native-support.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const globals = globalThis as Record<string, unknown>;

test("local grants belong to one tab and close even when the native view was pruned", async () => {
  const calls: { command: string; args: Record<string, unknown> }[] = [];
  globals.__localInvoke = async (command: string, args: Record<string, unknown>) => { calls.push({ command, args }); };
  const store = useBrowserStore.getState();
  const owner = store.openTabIn(store.chat, "http://localhost:7860/")!;
  const publicTab = store.openTabIn(store.chat, "https://example.com/")!;
  await grantLocalOrigin(owner, "http://localhost:7860/");
  assert.equal(localOriginGranted(owner, "http://localhost:7860/app"), true);
  assert.equal(localOriginGranted(publicTab, "http://localhost:7860/app"), false);
  assert.equal(localOriginGranted(owner, "http://localhost:7861/"), false);
  store.closeTab(owner);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(localOriginGranted(owner, "http://localhost:7860/"), false);
  assert.deepEqual(calls.at(-1), { command: "browser_view_close", args: { tabId: owner, forgetLocal: true } });
  store.closeTab(publicTab);
});

test("a tab closed while a grant was pending cannot regain local access", async () => {
  let release!: () => void;
  const calls: string[] = [];
  globals.__localInvoke = async (command: string) => {
    calls.push(command);
    if (command === "browser_view_grant") await new Promise<void>((resolve) => { release = resolve; });
  };
  const store = useBrowserStore.getState();
  const tab = store.openTabIn(store.chat, "http://localhost:7860/")!;
  const granting = grantLocalOrigin(tab, "http://localhost:7860/");
  await new Promise((resolve) => setTimeout(resolve, 0));
  store.closeTab(tab);
  release();
  await granting;
  assert.equal(localOriginGranted(tab, "http://localhost:7860/"), false);
  assert.deepEqual(calls, ["browser_view_grant", "browser_view_close"]);
});
