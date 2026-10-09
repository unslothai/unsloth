// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The coding agents Studio can set up, shared by Settings > Agents and Settings > API.
// Pure data so node tests can import it.

export const UNSLOTH_START_DOCS_URL =
  "https://unsloth.ai/docs/integrations/unsloth-start";

export type AgentDetails = {
  id: string;
  name: string;
  docsUrl: string;
  logo?: string;
  icon?: string;
  darkIcon?: string;
  color?: string;
  mark?: string;
};

// Names are untranslated, so `settings.agents.intro` lists them all to keep them searchable.
export const SUPPORTED_AGENTS: AgentDetails[] = [
  {
    id: "claude",
    name: "Claude Code",
    docsUrl: "https://unsloth.ai/docs/basics/claude-code",
    logo: "anthropic",
  },
  {
    id: "codex",
    name: "OpenAI Codex",
    docsUrl: "https://unsloth.ai/docs/basics/codex",
    logo: "openai",
  },
  {
    id: "hermes",
    name: "Hermes Agent",
    docsUrl: "https://unsloth.ai/docs/integrations/hermes-agent",
    // hermes.png is the desktop app icon from NousResearch/hermes-agent (apps/desktop/assets/icon.png)
    icon: "hermes.png",
  },
  {
    id: "openclaw",
    name: "OpenClaw",
    docsUrl: "https://unsloth.ai/docs/integrations/openclaw",
    icon: "openclaw.svg",
  },
  {
    id: "opencode",
    name: "OpenCode",
    docsUrl: "https://unsloth.ai/docs/integrations/opencode",
    icon: "opencode-light.svg",
    darkIcon: "opencode-dark.svg",
  },
  {
    id: "dsh",
    name: "DeepSeek Harness",
    docsUrl: "https://github.com/deepseek-ai/deepseek-harness",
    logo: "deepseek",
  },
  {
    id: "vibe",
    name: "Mistral Vibe",
    docsUrl: "https://github.com/mistralai/mistral-vibe",
    logo: "mistral",
  },
];

export function detailsFor(agentId: string): AgentDetails {
  return (
    SUPPORTED_AGENTS.find((agent) => agent.id === agentId) ?? {
      id: agentId,
      name: agentId,
      docsUrl: UNSLOTH_START_DOCS_URL,
      color: "#64748B",
      mark: agentId.slice(0, 2),
    }
  );
}
