// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("reduce motion defaults to off, and System and On stay selectable", () => {
	const store = readFileSync(new URL("../src/features/settings/stores/appearance-custom-store.ts", import.meta.url), "utf8");
	assert.match(store, /reduceMotion: "off",\n/);
	assert.match(store, /source\.reduceMotion === "on" \|\| source\.reduceMotion === "system"\n\s*\? source\.reduceMotion\n\s*: "off",/);
});
