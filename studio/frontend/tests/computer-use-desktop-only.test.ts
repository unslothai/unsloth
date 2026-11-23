// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { en } from "../src/i18n/locales/en.ts";

const tab = readFileSync(new URL("../src/features/settings/tabs/computer-use-tab.tsx", import.meta.url), "utf8");

test("browser Studio fills the empty Control section with a Desktop app only row", () => {
	assert.match(tab, /\{!isTauri && !status\?\.available \? \(\s*<SettingsRow/);
	const row = tab.slice(tab.indexOf("{!isTauri && !status?.available"), tab.indexOf(") : status ? ("));
	assert.match(row, /description=\{t\("computerUse\.settings\.desktopOnlyDescription"\)\}/);
	assert.match(row, /\{t\("settings\.keyboardShortcuts\.desktopOnly"\)\}/);
	// The Desktop app keeps its own row, reason and switch.
	assert.match(tab, /\) : status \? \(\s*<SettingsRow[\s\S]*?description=\{status\.available \? bold/);
	assert.ok(en.computerUse.settings.desktopOnlyDescription.includes("Unsloth Desktop app"));
});
