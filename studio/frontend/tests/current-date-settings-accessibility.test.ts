// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const chatTab = await readSrcAsync("features/settings/tabs/chat-tab.tsx");

test("the current date switch has an accessible name", () => {
  assert.match(
    chatTab,
    /aria-label=\{t\("settings\.chat\.currentDate\.label"\)\}/,
  );
});

test("current date setting errors are announced", () => {
  assert.match(chatTab, /role="alert"/);
});
