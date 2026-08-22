// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { en } from "../src/i18n/locales/en.ts";

// agents-tab.tsx reaches the chat barrel and cannot be imported here, so this
// asserts on source, like ~50 sibling tests.
const TAB = readFileSync(
  fileURLToPath(
    new URL("../src/features/settings/tabs/agents-tab.tsx", import.meta.url),
  ),
  "utf-8",
);

test("--reasoning is listed with the other start flags", () => {
  const start = TAB.indexOf("const OPTION_ROWS");
  const rows = TAB.slice(start, TAB.indexOf("];", start));
  assert.ok(
    rows.includes('flag: "--reasoning"'),
    "the flag row is in the options table",
  );
  assert.ok(rows.includes("settings.agents.options.reasoning"));
  // Next to --serve: both only mean anything for a server this command starts.
  assert.ok(
    rows.indexOf("--serve") < rows.indexOf("--reasoning"),
    "the row follows --serve",
  );
});

test("the row carries the effort recipe, which is a different flag", () => {
  const row = en.settings.agents.options.reasoning;
  // --reasoning is the on/off/auto mode. An effort level is a chat template
  // kwarg on the server, and unsloth start forwards unknown flags to the
  // agent, so it has to be set when the model is served.
  assert.ok(row.includes("off, auto, on"));
  assert.ok(row.includes('--chat-template-kwargs \'{"reasoning_effort":"medium"}\''));
  assert.ok(row.includes("unsloth run"));
});
