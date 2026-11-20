// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Compare composer's textarea carries role="combobox"; it must keep the text cursor, not the hand.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("a text field with role=combobox is left out of the pointer cursor rule", () => {
	const css = readSrc("index.css");
	assert.match(
		css,
		/\[role="combobox"\]:not\(textarea, input, \[contenteditable="true"\]\),/,
	);
	assert.doesNotMatch(css, /^\s*\[role="combobox"\],$/m);
});
