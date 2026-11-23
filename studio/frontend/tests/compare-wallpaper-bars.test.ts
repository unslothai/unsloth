// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("over a wallpaper, Compare's model bars take the thread wash and the pane fade hides", () => {
	assert.match(readSrc("features/chat/chat-page.tsx"), /pt-\[var\(--studio-chat-header-padding-top,11px\)\] compare-pane-header"/);
	const css = readSrc("index.css");
	assert.match(css, /\[data-wallpaper-ready\]\[data-wallpaper-ready\] \.compare-pane-header \{\n  background: color-mix\(/);
	assert.match(css, /\[data-wallpaper-ready\]\[data-wallpaper-ready\] \.compare-pane-fade \{\n  visibility: hidden;/);
});
