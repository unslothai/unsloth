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
  // Three places delete a chat. One of them skipping the setting would leave
  // sandbox folders behind depending on which button was used.
  const files = [
    "../src/components/app-sidebar.tsx",
    "../src/features/chat/chat-page.tsx",
    "../src/features/settings/components/archived-chats-dialog.tsx",
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
