// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("reduce motion defaults to on, and System and Off stay selectable", () => {
	const store = readFileSync(new URL("../src/features/settings/lib/appearance-customization.ts", import.meta.url), "utf8");
	assert.match(store, /reduceMotion: "on",\n/);
	assert.match(store, /reduceMotion: readChoice\(\s*source\.reduceMotion,\s*\["system", "on", "off"\],\s*"on",/);
});
