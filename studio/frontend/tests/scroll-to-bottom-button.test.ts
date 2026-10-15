// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// thread.tsx cannot be imported by node type stripping, so its shape is pinned from source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);
const { SETTINGS_SEARCH_INDEX } = await import(
  "../src/features/settings/settings-search.ts"
);

const THREAD = readSrc("components/assistant-ui/thread.tsx");
const STORE = readSrc("features/chat/stores/chat-preferences-store.ts");

function button(): string {
  const start = THREAD.indexOf("const ThreadScrollToBottom: FC");
  assert.notEqual(
    start,
    -1,
    "ThreadScrollToBottom is gone; this test needs rewriting",
  );
  return THREAD.slice(start, THREAD.indexOf("\n};", start));
}

test("the button shows by default, including installs that never saw the setting", () => {
  assert.equal(
    useChatPreferencesStore.getInitialState().showScrollToBottomButton,
    true,
  );
  assert.match(
    STORE,
    /showScrollToBottomButton: saved\?\.showScrollToBottomButton \?\? true/,
  );
});

test("turning it off hides the button without unmounting it", () => {
  const source = button();
  assert.match(source, /state\.showScrollToBottomButton/);
  // Unmounting would read as a content change to the autoscroll observer.
  assert.match(
    source,
    /\(isAtBottom \|\| !enabled\) && "invisible pointer-events-none"/,
  );
});

test("a smaller button with a larger arrow, lifted off the dark background", () => {
  const source = button();
  assert.match(source, /p-0 size-\[calc\(28px\*var\(--ui-space-scale,1\)\)\]/);
  assert.match(
    source,
    /className="size-\[calc\(var\(--ui-icon-size\)\*1\.125\)\]"/,
  );
  assert.match(source, /dark:bg-muted dark:hover:bg-accent/);
  assert.doesNotMatch(source, /dark:bg-background/);
});

test("the setting is a searchable switch in Chat settings", () => {
  const tab = readSrc("features/settings/tabs/chat-tab.tsx");
  const at = tab.indexOf('label={t("settings.chat.scrollToBottomButton")}');
  assert.notEqual(at, -1);
  const row = tab.slice(at, tab.indexOf("</SettingsRow>", at));
  assert.match(row, /<Switch/);
  assert.match(row, /checked=\{showScrollToBottomButton\}/);
  assert.match(row, /onCheckedChange=\{setShowScrollToBottomButton\}/);
  assert.ok(
    SETTINGS_SEARCH_INDEX.chat.includes("settings.chat.scrollToBottomButton"),
  );
});
