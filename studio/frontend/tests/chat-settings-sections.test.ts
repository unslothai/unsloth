// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Chat: Chat defaults keeps the message box and limits; reply and history behaviour sits in Conversations.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const tab = readSrc("features/settings/tabs/chat-tab.tsx");

function section(titleKey: string): string {
  const start = tab.indexOf(`<SettingsSection title={t("${titleKey}")}>`);
  assert.ok(start !== -1, `${titleKey} section exists`);
  return tab.slice(start, tab.indexOf("</SettingsSection>", start));
}

const labels = (body: string): string[] =>
  [...body.matchAll(/^\s+label=\{t\("([^"]+)"\)\}/gm)].map((match) => match[1]);

test("Chat defaults holds the message box, pastes, speed and context stats, and the owner limits only", () => {
  const defaults = section("settings.general.chatDefaults");
  assert.match(defaults, /<ComposerSettings embedded=\{true\} \/>/);
  assert.deepEqual(labels(defaults), ["settings.chat.pastedTextThreshold"]);
  assert.match(
    defaults,
    /<GenerationStatsUpdatesRow \/>\s+\{isOwner \? <ToolResultLimitRow \/> : null\}\s+\{isOwner \? <FirstTokenTimeoutRow \/> : null\}/,
  );
});

test("Conversations takes compaction, background runs, scrolling and reply notifications", () => {
  assert.deepEqual(
    labels(section("settings.chat.groups.conversations.title")),
    [
      "settings.chat.rememberParamsPerModel",
      "settings.chat.currentDate.label",
      "settings.chat.projectAttachments",
      "settings.general.autoTitleNewChats",
      "settings.chat.autoCompact",
      "settings.chat.keepRunningWhenTabClosed",
      "settings.chat.autoScroll",
      "settings.chat.replyNotification",
    ],
  );
});
