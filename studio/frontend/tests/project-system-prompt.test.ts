// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { composeProjectSystemPrompt } from "../src/features/chat/utils/project-system-prompt.ts";

test("existing stored instructions retain their original bytes and envelope", () => {
	assert.equal(
		composeProjectSystemPrompt(
			"Keep <user> data intact.\nRésumé — 中文",
			"  Answer briefly.\n ",
		),
		"<project_instructions>\nKeep <user> data intact.\nRésumé — 中文\n</project_instructions>\n\nAnswer briefly.",
	);
});

test("empty projects preserve user-only and empty system prompts", () => {
	assert.equal(
		composeProjectSystemPrompt("", "  Keep spacing\ninside.  "),
		"Keep spacing\ninside.",
	);
	assert.equal(composeProjectSystemPrompt("", " \n"), "");
	assert.equal(
		composeProjectSystemPrompt("Stored rules", ""),
		"<project_instructions>\nStored rules\n</project_instructions>",
	);
});
