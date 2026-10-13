// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// shared by Settings > Agents and Settings > API; kept data-only for Node.js tests.

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

// names stay untranslated; `settings.agents.intro` includes them for search.
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
    id: "vscode",
    name: "VS Code",
    docsUrl:
      "https://code.visualstudio.com/docs/copilot/customization/language-models",
    // vscode.svg is the stable icon from code.visualstudio.com/brand.
    icon: "vscode.svg",
  },
  {
    id: "dsh",
    name: "DeepSeek Harness",
    docsUrl: "https://github.com/deepseek-ai/deepseek-harness",
    logo: "deepseek",
  },
  {
    id: "opencode",
    name: "OpenCode",
    docsUrl: "https://unsloth.ai/docs/integrations/opencode",
    icon: "opencode-light.svg",
    darkIcon: "opencode-dark.svg",
  },
  {
    id: "hermes",
    name: "Hermes Agent",
    docsUrl: "https://unsloth.ai/docs/integrations/hermes-agent",
    // hermes.png comes from the NousResearch/hermes-agent desktop assets.
    icon: "hermes.png",
  },
  {
    id: "openclaw",
    name: "OpenClaw",
    docsUrl: "https://unsloth.ai/docs/integrations/openclaw",
    icon: "openclaw.svg",
  },
  {
    id: "pi",
    name: "Pi",
    docsUrl: "https://pi.dev",
    // pi.svg is https://pi.dev/logo-auto.svg, the logo the earendil-works/pi README uses.
    icon: "pi.svg",
  },
  {
    id: "vibe",
    name: "Mistral Vibe",
    docsUrl: "https://github.com/mistralai/mistral-vibe",
    logo: "mistral",
  },
];

// VS Code is detected by its `code` launcher, which most developers have. It only adds a model to
// Copilot Chat, so it is offered but never picked over a terminal agent on its own.
export const NOT_AUTO_PICKED_AGENTS: ReadonlySet<string> = new Set(["vscode"]);

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
