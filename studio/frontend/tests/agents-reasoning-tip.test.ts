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
  assert.ok(rows.includes('flag: "--reasoning-effort"'));
  assert.ok(rows.includes("settings.agents.options.reasoningEffort"));
  // Next to --serve: both only mean anything for a server this command starts.
  assert.ok(
    rows.indexOf("--serve") < rows.indexOf("--reasoning"),
    "the row follows --serve",
  );
});

test("both reasoning rows are there, and say what each one does", () => {
  const { reasoning, reasoningEffort } = en.settings.agents.options;
  // --reasoning is the on/off/auto switch, and auto (the model's template)
  // is what an agent session gets unless it is set.
  assert.ok(reasoning.includes("on, off, or auto"));
  assert.ok(reasoning.includes("chat template"));
  // The level is a separate flag. The accepted values are the model's own, so
  // the row gives an example rather than a list that would be wrong elsewhere.
  assert.ok(reasoningEffort.includes("e.g. medium"));
  assert.ok(reasoningEffort.includes("per model"));
  assert.ok(reasoningEffort.includes("chat template"));
});
