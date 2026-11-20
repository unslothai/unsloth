// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("the Translate language list is 28rem tall and fades while more languages are below", () => {
	const src = readSrc("features/translate/translate-page.tsx");
	assert.match(src, /const listFade = useScrollFades\(\);/);
	assert.match(src, /ref=\{listFade\.attach\}\n\s+onScroll=\{listFade\.onScroll\}/);
	assert.match(src, /"hover-scrollbar panel-scroll-fade-action max-h-\[min\(28rem,/);
});
