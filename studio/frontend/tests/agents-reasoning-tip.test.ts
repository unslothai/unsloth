// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createSettingsSearchIndex } from "../src/features/settings/settings-search.ts";
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

test("the reasoning example carries the agent and model that are picked", () => {
  // A fixed `unsloth start claude` example would tell the user to serve a model
  // other than the one the builder above just resolved.
  assert.match(TAB, /command=\{`\$\{command\} \$\{REASONING_FLAGS\}`\}/);
  assert.match(TAB, /const REASONING_FLAGS = "--reasoning auto"/);
});

test("the tip is findable from settings search", () => {
  const index = createSettingsSearchIndex({
    desktop: false,
    closeToTray: false,
  });
  // SettingsSection renders the title as its data-settings-label, so a hit here
  // has something to scroll to.
  assert.ok(index.agents.includes("settings.agents.reasoning.title"));
  assert.ok(en.settings.agents.reasoning.title.length > 0);
  assert.ok(en.settings.agents.reasoning.description.includes("--reasoning"));
  assert.ok(en.settings.agents.options.reasoning.includes("off"));
});
