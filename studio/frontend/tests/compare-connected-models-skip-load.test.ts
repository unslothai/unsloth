// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("Compare selects a connected model instead of loading it", () => {
	const src = readSrc("features/chat/shared-composer.tsx");
	const start = src.indexOf("async function ensureModelLoaded(");
	const body = src.slice(start, start + 1500);
	const guard = body.indexOf("if (isExternalModelId(selectedId)) {");
	assert.ok(guard !== -1, "ensureModelLoaded checks for a connected model");
	assert.ok(guard < body.indexOf("resolveResidentInitialConfig"), "before any load setup");
	assert.match(body, /live\.setCheckpoint\(selectedId, null\);\n\s+return "connected";/);
	assert.match(
		src,
		/const compareSelectionNeedsLoad = \(sel: CompareModelSelection\) => \{\n\s+if \(isExternalModelId\(sel\.id\)\) return false;/,
	);
});
