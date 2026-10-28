// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
register("./helpers/file-kind-resolver.mjs", import.meta.url);
const { currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");
const { browserTabType, fileBarKind, textFileKind } = await import("../src/features/browser/file-kind.ts");
const { REACT_PREVIEW_KEY_PREFIX, REACT_PREVIEW_TYPE } = await import("../src/features/browser/react-preview-type.ts");

const read = (path: string) => readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");

test("only the private type is a React preview; an uploaded .tsx is code", () => {
  assert.equal(textFileKind("dashboard.tsx", REACT_PREVIEW_TYPE), "react");
  assert.equal(textFileKind("dashboard.tsx", REACT_PREVIEW_TYPE, true), "text");
  assert.equal(textFileKind("dashboard.tsx", ""), "code");
  assert.equal(textFileKind("dashboard.tsx", "text/plain"), "code");
  assert.equal(textFileKind("dashboard.tsx", `${REACT_PREVIEW_TYPE}; charset=utf-8`), "code");
  // The tab gets the HTML page's bar, and opens outside Studio as text, never as a page.
  assert.equal(fileBarKind("dashboard.tsx", REACT_PREVIEW_TYPE), "html");
  assert.equal(fileBarKind("dashboard.tsx", ""), "code");
  assert.equal(browserTabType("dashboard.tsx", REACT_PREVIEW_TYPE), "text/plain");
});

test("a file that claims the private type opens as text unless chat opened it", () => {
  const store = useBrowserStore.getState();
  const typeOf = () => {
    const state = useBrowserStore.getState();
    const tab = state.tabs.find((candidate) => candidate.id === state.activeTabId);
    const entry = tab && currentEntry(tab);
    return entry?.kind === "file" ? entry.contentType : null;
  };
  store.openFile({ blob: new Blob(["x"], { type: REACT_PREVIEW_TYPE }), name: "a.tsx", key: "download:1" });
  assert.equal(typeOf(), "text/plain");
  store.openFile({ blob: new Blob(["x"]), name: "b.tsx", contentType: REACT_PREVIEW_TYPE });
  assert.equal(typeOf(), "text/plain");
  store.openFile({ blob: new Blob(["x"]), name: "c.tsx", contentType: REACT_PREVIEW_TYPE, key: `${REACT_PREVIEW_KEY_PREFIX}1` });
  assert.equal(typeOf(), REACT_PREVIEW_TYPE);
});

test("a React preview is one of the chat's pages: reopening focuses it, a chat switch closes it", () => {
  const store = useBrowserStore.getState();
  for (const tab of useBrowserStore.getState().tabs) store.closeTab(tab.id);
  const open = () =>
    store.openFile({
      blob: new Blob(["export default () => null"], { type: REACT_PREVIEW_TYPE }),
      name: "app.tsx",
      contentType: REACT_PREVIEW_TYPE,
      key: `${REACT_PREVIEW_KEY_PREFIX}fence:t:m:abc`,
    });
  open();
  open();
  assert.equal(useBrowserStore.getState().tabs.length, 1);
  assert.equal(useBrowserStore.getState().tabs[0]?.openKey, "file:html:react:fence:t:m:abc");
  store.openFile({ blob: new Blob(["x"]), name: "notes.txt", key: "sandbox:s:notes.txt" });
  store.closeChatPages();
  assert.deepEqual(
    useBrowserStore.getState().tabs.map((tab) => tab.openKey),
    ["file:sandbox:s:notes.txt"],
  );
});

test("openReactInBrowser uses the private type and the chat-page key prefix", () => {
  const index = read("../src/features/browser/index.ts");
  assert.match(index, /blob: new Blob\(\[code\], \{ type: REACT_PREVIEW_TYPE \}\),/);
  assert.match(index, /contentType: REACT_PREVIEW_TYPE,/);
  assert.match(index, /key: `\$\{REACT_PREVIEW_KEY_PREFIX\}\$\{key\}`,/);
  assert.ok(REACT_PREVIEW_KEY_PREFIX.startsWith("html:"));
  // A React tab shows outside a tab of its own only as its code.
  assert.match(read("../src/features/browser/file-view.tsx"), /const kind = fileKind === "react" && !tabId \? "code" : fileKind;/);
});
