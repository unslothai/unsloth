// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { en: upstreamEnglish } = await import("../src/i18n/locales/en.ts");
const { mergeMessageTrees } = await import(
  "../src/features/unforgettable/i18n/merge-message-trees.ts"
);
const messagesModule = await import("../src/i18n/messages.ts");
const { unforgettableMessages } = await import(
  "../src/features/unforgettable/i18n/en.ts"
);

test("upstream English has no Unforgettable keys", () => {
  assert.equal("unforgettable" in upstreamEnglish, false);
  assert.equal("unforgettable" in upstreamEnglish.shell.navigation, false);
  assert.equal("unforgettable" in upstreamEnglish.settings.tabs, false);
  assert.equal("unforgettable" in upstreamEnglish.settings, false);
});

test("merged English serves Unforgettable keys without dropping upstream ones", () => {
  assert.equal(
    messagesModule.translate("unforgettable.page.title"),
    "Unforgettable",
  );
  assert.equal(
    messagesModule.translate("shell.navigation.unforgettable"),
    "Unforgettable",
  );
  assert.equal(
    messagesModule.translate("settings.tabs.unforgettable"),
    "Unforgettable",
  );
  assert.equal(
    messagesModule.translate("settings.unforgettable.store.path"),
    "memory.db",
  );
  assert.equal(messagesModule.translate("common.cancel"), "Cancel");
  assert.ok(
    messagesModule
      .translate("modelMemory.readout", {
        model: "1G",
        context: "2G",
        total: "3G",
        budget: "8G",
      })
      .includes("1G"),
  );
});

test("a lazy catalog keeps its own strings and receives the overlay", async () => {
  const load = messagesModule.loadLocaleMessages("de", async () => ({
    de: { common: { cancel: "Abbrechen" } },
  }));
  assert.ok(load);
  await load;
  assert.equal(
    messagesModule.translate("common.cancel", undefined, "de"),
    "Abbrechen",
  );
  assert.equal(
    messagesModule.translate("unforgettable.page.title", undefined, "de"),
    "Unforgettable",
  );
});

test("mergeMessageTrees nests overlay keys without replacing siblings", () => {
  const merged = mergeMessageTrees(
    { shell: { navigation: { chat: "Chat" } }, common: { cancel: "Cancel" } },
    unforgettableMessages,
  );
  assert.equal(merged.shell.navigation.chat, "Chat");
  assert.equal(merged.shell.navigation.unforgettable, "Unforgettable");
  assert.equal(merged.common.cancel, "Cancel");
  assert.equal(merged.unforgettable.page.title, "Unforgettable");
});
