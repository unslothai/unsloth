// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type McpPreset = {
  id: string;
  displayName: string;
  url: string;
  label?: string;
  hint?: string;
  disablesWebSearch?: boolean;
};

const REMOTE_MCP_PRESETS: readonly McpPreset[] = [
  {
    id: "unsloth-docs",
    displayName: "Unsloth Docs",
    url: "https://unsloth.ai/docs/~gitbook/mcp",
  },
  {
    id: "context7",
    displayName: "Context7",
    url: "https://mcp.context7.com/mcp",
    label: "Context7 (Realtime Docs)",
  },
  {
    id: "exa",
    displayName: "Exa",
    url: "https://mcp.exa.ai/mcp",
    label: "Exa (Semantic Search)",
    hint: "Enabling Exa will disable default search",
    disablesWebSearch: true,
  },
  {
    id: "huggingface",
    displayName: "Hugging Face",
    url: "https://huggingface.co/mcp",
  },
] as const;

const CUA_DRIVER_PRESET: McpPreset = {
  id: "cua-driver",
  displayName: "Cua Driver",
  url: "cua-driver mcp",
  label: "Cua Driver (Computer Use)",
  hint: "Requires Cua Driver on PATH. Install from cua.ai/docs/cua-driver",
};

export function getMcpPresets(isDesktop: boolean): readonly McpPreset[] {
  return isDesktop
    ? [CUA_DRIVER_PRESET, ...REMOTE_MCP_PRESETS]
    : REMOTE_MCP_PRESETS;
}
