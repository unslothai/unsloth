// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_COMPOSER_RICH_TEXT,
  migrateComposerRichText,
} from "../src/features/chat/utils/composer-rich-text.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

test("the old rich paste switch maps onto the three-way setting", () => {
  assert.equal(migrateComposerRichText(undefined, true), "paste");
  assert.equal(migrateComposerRichText(undefined, false), "off");
  assert.equal(migrateComposerRichText(undefined, undefined), "paste");
  assert.equal(DEFAULT_COMPOSER_RICH_TEXT, "paste");
});

test("a saved three-way value wins over a leftover switch", () => {
  assert.equal(migrateComposerRichText("full", false), "full");
  assert.equal(migrateComposerRichText("off", true), "off");
  assert.equal(migrateComposerRichText("paste", false), "paste");
});

test("an unknown stored value falls back instead of leaving the picker blank", () => {
  assert.equal(migrateComposerRichText("rich", undefined), "paste");
  assert.equal(migrateComposerRichText(1, false), "off");
  assert.equal(migrateComposerRichText(null, null), "paste");
});

test("a saved rich paste switch is read into the new setting", async () => {
  const read = async (state: object) => {
    store.set(
      "unsloth_chat_preferences",
      JSON.stringify({ state, version: 1 }),
    );
    await useChatPreferencesStore.persist.rehydrate();
    return useChatPreferencesStore.getState();
  };
  assert.equal(
    useChatPreferencesStore.getInitialState().composerRichText,
    "paste",
  );
  assert.equal((await read({ richTextPaste: false })).composerRichText, "off");
  assert.equal((await read({ richTextPaste: true })).composerRichText, "paste");
  assert.equal((await read({})).composerRichText, "paste");
  assert.equal(
    (await read({ composerRichText: "full", richTextPaste: false }))
      .composerRichText,
    "full",
  );
  useChatPreferencesStore.getState().setComposerRichText("off");
  const written = JSON.parse(
    store.get("unsloth_chat_preferences") ?? "{}",
  ).state;
  assert.equal(written.composerRichText, "off");
  assert.equal("richTextPaste" in written, false);
});
