// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

const CHAT_TAB = readSrc("features/settings/tabs/chat-tab.tsx");
const SIDEBAR = readSrc("components/app-sidebar.tsx");
const SEARCH = readSrc("features/settings/settings-search.ts");

test("project grouping is set from the sidebar, not Chat settings", () => {
  assert.doesNotMatch(
    CHAT_TAB,
    /setOrganizeBy|settings\.chat\.projectsSection/,
  );
  assert.doesNotMatch(SEARCH, /settings\.chat\.projectsSection/);
  assert.equal("projectsSection" in en.settings.chat, false);
  assert.match(
    SIDEBAR,
    /\{ value: "project", key: "shell\.organize\.byProject"/,
  );
  assert.match(
    SIDEBAR,
    /onValueChange=\{\(value\) => setOrganizeBy\(value as SidebarOrganizeBy\)\}/,
  );
});
