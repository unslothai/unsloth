// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The global then project AGENTS.md the backend found, each under a `# Source:` line. */
export type AgentsMdRecord = { text: string };

export const EMPTY_AGENTS_MD: AgentsMdRecord = { text: "" };

/** System prompt for one request. AGENTS.md first, so project instructions and the typed system prompt
 *  have the last word; with no AGENTS.md this is byte-identical to the prompt before AGENTS.md support. */
export function composeChatInstructions(parts: {
  agentsMd?: AgentsMdRecord;
  projectInstructions: string;
  systemPrompt: string;
}): string {
  const agentsMd = parts.agentsMd?.text.trim() ?? "";
  return [
    agentsMd ? `<agents_md>\n${agentsMd}\n</agents_md>` : "",
    parts.projectInstructions
      ? `<project_instructions>\n${parts.projectInstructions}\n</project_instructions>`
      : "",
    parts.systemPrompt.trim(),
  ]
    .filter(Boolean)
    .join("\n\n");
}
