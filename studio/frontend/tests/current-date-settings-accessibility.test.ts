// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const chatTab = await readFile(
  new URL("../src/features/settings/tabs/chat-tab.tsx", import.meta.url),
  "utf8",
);

test("the current date switch has an accessible name", () => {
  assert.match(
    chatTab,
    /aria-label=\{t\("settings\.chat\.currentDate\.label"\)\}/,
  );
});

test("current date setting errors are announced", () => {
  assert.match(chatTab, /role="alert"/);
});
