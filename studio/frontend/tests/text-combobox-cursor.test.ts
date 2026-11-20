// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Compare composer's textarea has role="combobox" and must keep the text cursor.

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
