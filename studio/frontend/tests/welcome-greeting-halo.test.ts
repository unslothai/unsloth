// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { WALLPAPER_HOST_FLAGS } from "../src/features/chat-background/host-flags.ts";
import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");

test("the welcome greeting's halo is drawn over a real wallpaper only, not the plain Glass backdrop", () => {
	// Over the Glass backdrop (no image) the faint radial halo bands into concentric rings around the
	// logo and greeting; it only exists to lift the greeting off a photo.
	const rules = [...CSS.matchAll(/([^{}]*\.unsloth-welcome-greeting(?:::before)?)\s*\{/g)].map((m) => m[1].trim());
	const halo = rules.filter((sel) => /\[data-wallpaper-(?:ready|image)\]/.test(sel));
	assert.equal(halo.length, 2, `expected the greeting and its ::before halo rule, got ${JSON.stringify(halo)}`);
	for (const sel of halo) {
		assert.match(sel, /\[data-wallpaper-image\]/, `halo rule must need a real image: ${sel}`);
		assert.doesNotMatch(sel, /\[data-wallpaper-ready\]\[data-wallpaper-ready\]/, sel);
	}
	// The flag is set only for a ready wallpaper that is not the Glass backdrop.
	const image = WALLPAPER_HOST_FLAGS.find(([name]) => name === "data-wallpaper-image");
	assert.ok(image);
	assert.match(image[1], /:not\(\[data-glass-backdrop\]\)/);
});
