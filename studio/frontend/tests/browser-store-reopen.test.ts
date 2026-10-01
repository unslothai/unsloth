// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { browserFile, currentEntry, setNativeWebHistory, useBrowserStore } = await import(
  "../src/features/browser/store.ts"
);

test("reopening a rewritten file shows its new bytes; an unchanged one keeps its viewer", async () => {
  const store = useBrowserStore.getState();
  const open = (text: string) => store.openFile({ blob: new Blob([text]), name: "a.txt", key: "sandbox/a.txt" });
  const shown = () => {
    const tab = useBrowserStore.getState().tabs[0];
    const entry = tab && currentEntry(tab);
    return entry?.kind === "file" ? entry.fileId : null;
  };
  open("version 1");
  const first = shown();
  open("version 1");
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(shown(), first);
  open("version 2");
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.notEqual(shown(), first);
  assert.equal(await browserFile(shown() ?? "")?.text(), "version 2");
  // Two quick reopens: the last one wins.
  open("version 3");
  open("version 4");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(await browserFile(shown() ?? "")?.text(), "version 4");
  assert.equal(useBrowserStore.getState().tabs.length, 1);
  // Closing another tab while a reopen is queued keeps the queued bytes.
  store.openFile({ blob: new Blob(["other"]), name: "b.txt", key: "sandbox/b.txt" });
  open("version 5");
  store.closeTab(useBrowserStore.getState().tabs[1]?.id ?? "");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(await browserFile(shown() ?? "")?.text(), "version 5");
});

test("a native tab's navigations replace its entry unless asked to keep it", () => {
  setNativeWebHistory(true);
  const store = useBrowserStore.getState();
  store.openUrl("https://a.example/", { newTab: true });
  const tab = () => useBrowserStore.getState().tabs.find((candidate) => candidate.id === id);
  const id = useBrowserStore.getState().activeTabId ?? "";
  store.navigate(id, { url: "https://b.example/" });
  assert.equal(tab()?.history.length, 1);
  store.navigate(id, { url: "https://c.example/" }, { replace: false });
  assert.deepEqual(
    tab()?.history.map((entry) => (entry.kind === "web" ? entry.url : "")),
    ["https://b.example/", "https://c.example/"],
  );
  setNativeWebHistory(false);
});
