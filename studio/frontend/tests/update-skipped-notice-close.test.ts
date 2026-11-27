// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("the Skipped update notice has a close button that dismisses it", () => {
	const banner = readSrc("components/llama-update-banner.tsx");
	assert.match(banner, /onClick=\{onExpire\}\n\s+aria-label=\{t\("common\.close"\)\}/);
	assert.match(banner, /data-testid="llama-update-skipped-close"/);
});
