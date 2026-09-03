// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { useChatPreferencesStore } from "../src/features/chat/stores/chat-preferences-store.ts";

test("deleting a chat leaves its files alone until asked", () => {
  // Files are the destructive half of a delete, so an install that never saw
  // this setting must not start removing them.
  const fresh = useChatPreferencesStore.getInitialState();
  assert.equal(fresh.alwaysDeleteChatFiles, false);
});

test("a saved payload without the key still defaults to off", async () => {
  // Preferences written before this setting existed rehydrate through merge,
  // and a missing key there must not read as "yes, delete the files".
  const source = await readFile(
    new URL("../src/features/chat/stores/chat-preferences-store.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /alwaysDeleteChatFiles: saved\?\.alwaysDeleteChatFiles \?\? false/,
  );
});

test("every chat delete path honours the preference", async () => {
  // Five places delete chats. One skipping the setting would leave sandbox
  // folders behind depending on which button was used, and after a clear-all
  // there is no chat row left to reach them from.
  const files = [
    "../src/components/app-sidebar.tsx",
    "../src/features/chat/chat-page.tsx",
    "../src/features/settings/components/archived-chats-dialog.tsx",
    "../src/features/settings/components/recent-dictations-view.tsx",
    "../src/features/settings/tabs/data-tab.tsx",
  ];
  for (const file of files) {
    const source = await readFile(new URL(file, import.meta.url), "utf8");
    assert.match(
      source,
      /alwaysDeleteChatFiles/,
      `${file} deletes chats without reading the preference`,
    );
  }
});

test("every confirmation discloses the file deletion and can undo it", async () => {
  // The preference makes a delete destructive past the chat itself. A dialog
  // that neither says so nor offers the switch removes a sandbox the user was
  // never shown, and there is no undo.
  // Every dialog that can reach a delete, with the state each one names.
  const dialogs: Array<[string, string]> = [
    ["../src/components/app-sidebar.tsx", "deleteFilesOnDelete"],
    ["../src/features/chat/chat-page.tsx", "deleteFilesOnDelete"],
    [
      "../src/features/settings/components/archived-chats-dialog.tsx",
      "deleteFilesOnDelete",
    ],
    [
      "../src/features/settings/components/recent-dictations-view.tsx",
      "deleteFilesOnDelete",
    ],
    ["../src/features/settings/tabs/data-tab.tsx", "deleteFilesOnClear"],
  ];
  for (const [file, state] of dialogs) {
    const source = await readFile(new URL(file, import.meta.url), "utf8");
    // The rendered control, not just the state behind it.
    assert.ok(
      source.includes(`checked={${state}}`),
      `${file} deletes files without a switch to turn it off`,
    );
    // Preselected from the preference: the switch has to show what will happen,
    // not sit off while the delete removes the files anyway.
    const setter = state.replace(/^delete/, "setDelete");
    assert.match(
      source,
      new RegExp(`${setter}\\([\\s\\S]{0,40}alwaysDeleteChatFiles`),
      `${file} does not preselect the switch from the preference`,
    );
  }
});

test("the confirmed delete follows the switch, not the preference", async () => {
  // Reading the preference at delete time would ignore a switch the user just
  // turned off, which is the whole point of showing it.
  for (const [file, signature] of [
    ["../src/features/chat/chat-page.tsx", /item: SidebarItem,\s*deleteFiles: boolean/],
    [
      "../src/features/settings/components/archived-chats-dialog.tsx",
      /item: SidebarItem,\s*deleteFiles: boolean/,
    ],
    [
      "../src/features/settings/components/recent-dictations-view.tsx",
      /confirmDeleteWithChat\(deleteFiles: boolean\)/,
    ],
    ["../src/features/settings/tabs/data-tab.tsx", /deleteFiles: deleteFilesOnClear/],
  ] as Array<[string, RegExp]>) {
    const source = await readFile(new URL(file, import.meta.url), "utf8");
    // The executor is handed the answer instead of reading the store, so the
    // confirmed value is the one that reaches the request.
    assert.match(source, signature, `${file} does not take the confirmed value`);
    assert.equal(
      /deleteFiles: alwaysDeleteChatFiles/.test(source),
      false,
      `${file} still reads the preference past its own switch`,
    );
  }
});

test("the clear-all chain carries deleteFiles to the request", async () => {
  // Three hops: data-tab -> clearAllChats -> clearStoredChats -> the DELETE.
  // Any hop dropping the option loses it silently, since each still compiles.
  const read = (path: string) =>
    readFile(new URL(path, import.meta.url), "utf8");

  const clearAll = await read("../src/features/chat/utils/clear-all-chats.ts");
  assert.match(clearAll, /clearStoredChats\(options\)/);

  const storage = await read(
    "../src/features/chat/utils/chat-history-storage.ts",
  );
  assert.match(storage, /deleteFiles: options\.deleteFiles/);

  const api = await read("../src/features/chat/api/chat-api.ts");
  assert.match(api, /options\.deleteFiles \? "\?delete_files=true" : ""/);
});
