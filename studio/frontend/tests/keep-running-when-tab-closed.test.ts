// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_KEEP_RUNNING_WHEN_TAB_CLOSED,
  tabCloseRunFields,
} from "../src/features/chat/utils/keep-running-when-tab-closed.ts";
import {
  installLocalStorageFake,
  readSrc,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { sanitizeChatSettings, isEmptyChatSettings } = await import(
  "../src/features/chat/utils/chat-settings-storage.ts"
);
const { snapshotQueuedChatRunSettings } = await import(
  "../src/features/chat/utils/queued-chat-run-settings.ts"
);

test("runs keep going after the tab closes unless the setting says otherwise", () => {
  assert.equal(DEFAULT_KEEP_RUNNING_WHEN_TAB_CLOSED, true);
  // On sends nothing, so admission is what it was before the setting existed.
  assert.deepEqual(tabCloseRunFields(true), {});
  assert.deepEqual(tabCloseRunFields(false), { stopWhenUnattended: true });
});

test("the setting survives the round trip through stored settings", () => {
  assert.deepEqual(sanitizeChatSettings({ keepRunningWhenTabClosed: false }), {
    keepRunningWhenTabClosed: false,
  });
  assert.deepEqual(sanitizeChatSettings({ keepRunningWhenTabClosed: "no" }), {});
  assert.equal(isEmptyChatSettings({ keepRunningWhenTabClosed: false }), false);
});

test("both durable admissions carry the choice", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const uses = adapter.match(
    /\.\.\.tabCloseRunFields\(\s*runtime\.keepRunningWhenTabClosed,?\s*\)/g,
  );
  // One for createResearchRun, one for createChatGenerationRunUntilAbort.
  assert.equal(uses?.length, 2);
});

test("a queued send keeps the choice it was sent with", () => {
  const state = {
    params: {},
    keepRunningWhenTabClosed: false,
  } as unknown as Parameters<typeof snapshotQueuedChatRunSettings>[0];
  assert.equal(
    snapshotQueuedChatRunSettings(state).keepRunningWhenTabClosed,
    false,
  );
  // Flipping it while a send waits must count as a settings change, like the settings beside it.
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  const setter = store.slice(
    store.indexOf("setKeepRunningWhenTabClosed: (keepRunningWhenTabClosed) =>"),
  );
  assert.match(
    setter.slice(0, setter.indexOf("}),")),
    /queuedSettingsEpoch: state\.queuedSettingsEpoch \+ 1/,
  );
});
