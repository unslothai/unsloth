// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Preserve Studio's existing stored-project/user-system prompt serialization. */
export function composeProjectSystemPrompt(
	projectInstructions: string,
	userSystemPrompt: string,
): string {
	return [
		projectInstructions
			? `<project_instructions>\n${projectInstructions}\n</project_instructions>`
			: "",
		userSystemPrompt.trim(),
	]
		.filter(Boolean)
		.join("\n\n");
}
