// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Chat owns composer preferences, followed by conversation and reply settings.

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

const sectionOrder = (keys: string[]): number[] =>
  keys.map((key) => tab.indexOf(`<SettingsSection title={t("${key}")}>`));

const composer = readSrc("features/settings/components/composer-settings.tsx");
const composerBody = (name: string): string => {
  const start = composer.indexOf(`export function ${name}(`);
  assert.ok(start !== -1, `${name} exists`);
  const next = composer.indexOf("\nexport function ", start + 1);
  return composer.slice(start, next === -1 ? undefined : next);
};
const composerLabels = labels;

test("Chat defaults keeps scrolling and reply stats", () => {
  const defaults = section("settings.general.chatDefaults");
  assert.deepEqual(labels(defaults), ["settings.chat.autoScroll"]);
  assert.match(defaults, /<GenerationStatsUpdatesRow \/>\s+$/);
  assert.doesNotMatch(defaults, /isOwner/, "the owner limits sit in Replies");
});

test("Composer groups behavior, pasting, context and appearance in Chat", () => {
  const settings = section("composerSettings.title");
  assert.match(settings, /<ComposerSettings embedded=\{true\} \/>/);
  assert.deepEqual(labels(settings), [
    "settings.chat.pastedTextThreshold",
    "settings.appearance.custom.chatWidth.label",
    "settings.chat.composerPosition",
    "settings.appearance.custom.sentAttachments.label",
  ]);
  assert.deepEqual(composerLabels(composerBody("ComposerSettings")), [
    "composerSettings.sendShortcut",
    "composerSettings.followUp",
    "composerSettings.shareDraft",
  ]);
  assert.deepEqual(composerLabels(composerBody("ComposerFormatRows")), [
    "composerSettings.plainText",
    "composerSettings.richPaste",
  ]);
  assert.match(
    composerBody("ContextWindowUsageRow"),
    /<SettingsRow label=\{t\("composerSettings\.showContext"\)\}>/,
  );
  assert.match(settings, /<ContextWindowUsageRow \/>/);
  assert.match(settings, /<ComposerFormatRows \/>/);
  const display = section("settings.chat.groups.display.title");
  assert.doesNotMatch(display, /<ContextWindowUsageRow \/>|<ComposerFormatRows \/>/);
  assert.equal(
    tab.match(/<ContextWindowUsageRow \/>|<ComposerFormatRows \/>/g)?.length,
    2,
  );
});

test("Conversations keeps per-chat setup and compaction", () => {
  assert.deepEqual(
    labels(section("settings.chat.groups.conversations.title")),
    [
      "settings.chat.rememberParamsPerModel",
      "settings.chat.currentDate.label",
      "settings.chat.projectAttachments",
      "settings.general.autoTitleNewChats",
      "settings.chat.autoCompact",
    ],
  );
});

test("Replies holds background runs and notifications, owner limits last", () => {
  const replies = section("settings.chat.groups.replies.title");
  assert.deepEqual(labels(replies), [
    "settings.chat.keepRunningWhenTabClosed",
    "settings.chat.replyNotification",
  ]);
  assert.match(
    replies,
    /\{isOwner \? <ToolResultLimitRow \/> : null\}\s+\{isOwner \? <FirstTokenTimeoutRow \/> : null\}\s+$/,
  );
});

test("Thinking and tools groups the visibility rows out of Display", () => {
  assert.deepEqual(
    labels(section("settings.chat.groups.thinkingTools.title")),
    [
      "settings.chat.thinking.visibility",
      "settings.chat.tools.visibility",
      "settings.chat.tools.foldIntoThinking",
      "settings.chat.webSearch.images",
    ],
  );
  assert.deepEqual(labels(section("settings.chat.groups.display.title")), [
    "settings.chat.showLinkPreviews",
    "settings.chat.showResponseModel",
    "settings.chat.showReplyTimes",
    "settings.chat.showGenerationDetails",
    "settings.chat.inlineEditResponse",
    "settings.chat.turnNavigation",
  ]);
});

test("sections run composer, defaults, memory, conversations, replies, thinking and tools, display", () => {
  const [composer, defaults, conversations, replies, thinkingTools, display, models] =
    sectionOrder([
      "composerSettings.title",
      "settings.general.chatDefaults",
      "settings.chat.groups.conversations.title",
      "settings.chat.groups.replies.title",
      "settings.chat.groups.thinkingTools.title",
      "settings.chat.groups.display.title",
      "settings.chat.modelSelection.title",
    ]);
  const memory = tab.indexOf("<ChatMemorySection />");
  assert.ok(
    composer < defaults &&
      defaults < memory &&
      memory < conversations &&
      conversations < replies &&
      replies < thinkingTools &&
      thinkingTools < display &&
      display < models,
  );
  const search = readSrc("features/settings/settings-search.ts");
  for (const key of ["replies", "thinkingTools"]) {
    assert.match(
      search,
      new RegExp(`"settings\\.chat\\.groups\\.${key}\\.title"`),
    );
  }
});

test("section-level resets use the Appearance reset: an outline small button, greyed out with nothing to undo", () => {
  const appearance = readSrc(
    "features/settings/components/appearance-custom-controls.tsx",
  );
  const resetAll = appearance.slice(
    appearance.indexOf("export function ResetCustomizationButton"),
  );
  assert.match(
    resetAll,
    /variant="outline"\s+size="sm"\s+disabled=\{pristine\}/,
  );

  const menu = tab.slice(tab.indexOf('{t("settings.chat.groups.menu.title")}'));
  assert.match(
    menu,
    /<Button\s+type="button"\s+variant="outline"\s+size="sm"\s+disabled=\{plusMenuPristine\}\s+onClick=\{resetPlusMenu\}\s*>\s*\{t\("settings\.chat\.menu\.reset"\)\}/,
  );
  assert.match(tab, /plusMenuAtDefaults\(state\.pins, state\.order\)/);

  const shortcuts = readSrc(
    "features/settings/tabs/keyboard-shortcuts-tab.tsx",
  );
  // Appearance's footer: a full-width rule above, the button on the right, after the last section.
  const RULED_FOOTER =
    /className="(?:mt-auto )?flex justify-end border-t border-border\/60 pt-4">\s+<(?:Button|ResetCustomizationButton)\b/;
  assert.match(
    readSrc("features/settings/tabs/appearance-tab.tsx"),
    RULED_FOOTER,
  );
  assert.match(tab, RULED_FOOTER);
  assert.match(shortcuts, RULED_FOOTER);
  assert.ok(
    tab.lastIndexOf("</SettingsSection>") <
      tab.indexOf('t("settings.chat.menu.reset")'),
    "the Chat menu reset sits below the last section, like Appearance's",
  );
  const footer = shortcuts.slice(
    shortcuts.lastIndexOf(
      "<Button",
      shortcuts.indexOf('t("settings.keyboardShortcuts.resetAll")'),
    ),
  );
  assert.match(
    footer,
    /^<Button\s+type="button"\s+variant="outline"\s+size="sm"\s+disabled=/,
  );
  assert.doesNotMatch(shortcuts, /variant="dark"/);

  for (const [file, key] of [
    ["features/settings/tabs/browser-tab.tsx", "browser.downloadLocationReset"],
    [
      "features/settings/tabs/resources-tab.tsx",
      "settings.resources.storage.resetAction",
    ],
  ] as const) {
    const source = readSrc(file);
    const button = source.slice(
      source.lastIndexOf("<Button", source.indexOf(`t("${key}")`)),
      source.indexOf(`t("${key}")`),
    );
    assert.match(button, /variant="outline"/, `${key} is an outline button`);
  }
});
