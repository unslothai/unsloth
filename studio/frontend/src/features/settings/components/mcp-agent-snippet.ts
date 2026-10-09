// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Setup text that points a coding agent at Studio's MCP endpoint. Pure so node tests
// can pin every agent's format. Each agent reads the key from UNSLOTH_API_KEY unless
// the caller passes a key it just created.

import { type AgentCommandOs, quoteShellArg } from "./agent-command.ts";

export const MCP_SERVER_NAME = "unsloth-studio";
export const MCP_API_KEY_ENV = "UNSLOTH_API_KEY";

// Only these two are shell commands, so only they need the Unix/PowerShell choice.
export const MCP_SHELL_AGENT_IDS: ReadonlySet<string> = new Set([
  "claude",
  "codex",
]);

export type McpSnippet = {
  text: string;
  /** set for agents configured by a file the user edits */
  configPath: string | null;
  /** the text reads UNSLOTH_API_KEY, so it has to be set before the agent starts */
  readsKeyEnv: boolean;
};

/** `<origin>/mcp/` for any Studio base, or null when the base is not a URL. */
export function mcpEndpointUrl(base: string | null | undefined): string | null {
  if (!base) {
    return null;
  }
  try {
    const url = new URL(base);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return null;
    }
    // Studio answers the bare /mcp with a redirect that clients may not follow on a POST.
    return `${url.origin}/mcp/`;
  } catch {
    return null;
  }
}

// JSON strings are valid YAML double-quoted scalars and TOML basic strings.
const quoted = (value: string): string => JSON.stringify(value);

function claudeSnippet(
  url: string,
  os: AgentCommandOs,
  apiKey: string | null,
): string {
  const command = `claude mcp add --transport http ${MCP_SERVER_NAME} ${quoteShellArg(url, os)}`;
  if (apiKey) {
    return `${command} --header ${quoteShellArg(`Authorization: Bearer ${apiKey}`, os)}`;
  }
  // In PowerShell $UNSLOTH_API_KEY is an unset shell variable, not the environment.
  const envRef =
    os === "windows" ? `$env:${MCP_API_KEY_ENV}` : `$${MCP_API_KEY_ENV}`;
  return `${command} --header "Authorization: Bearer ${envRef}"`;
}

function codexSnippet(url: string, os: AgentCommandOs): string {
  // Codex only takes the name of an environment variable, never the key itself.
  return [
    `codex mcp add ${MCP_SERVER_NAME} --url ${quoteShellArg(url, os)} --bearer-token-env-var ${MCP_API_KEY_ENV}`,
    `# Optional, for long jobs, in ~/.codex/config.toml under [mcp_servers.${MCP_SERVER_NAME}]:`,
    "# tool_timeout_sec = 300",
  ].join("\n");
}

function opencodeSnippet(url: string, apiKey: string | null): string {
  const config = {
    mcp: {
      [MCP_SERVER_NAME]: {
        type: "remote",
        url,
        oauth: false,
        headers: {
          Authorization: `Bearer ${apiKey ?? `{env:${MCP_API_KEY_ENV}}`}`,
        },
      },
    },
  };
  return JSON.stringify(config, null, 2);
}

function hermesSnippet(url: string, apiKey: string | null): string {
  const token = apiKey ?? `\${${MCP_API_KEY_ENV}}`;
  return [
    "mcp_servers:",
    "  unsloth_studio:",
    `    url: ${quoted(url)}`,
    "    headers:",
    `      Authorization: ${quoted(`Bearer ${token}`)}`,
    "# Then run /reload-mcp in Hermes",
  ].join("\n");
}

function openclawSnippet(url: string, apiKey: string | null): string {
  const config = {
    mcp: {
      servers: {
        [MCP_SERVER_NAME]: {
          url,
          // Without it a url server defaults to SSE.
          transport: "streamable-http",
          headers: {
            Authorization: `Bearer ${apiKey ?? `\${${MCP_API_KEY_ENV}}`}`,
          },
        },
      },
    },
  };
  return JSON.stringify(config, null, 2);
}

function vibeSnippet(url: string): string {
  // Vibe's static auth reads the key from a named variable only.
  return [
    "[[mcp_servers]]",
    'name = "unsloth_studio"',
    'transport = "streamable-http"',
    `url = ${quoted(url)}`,
    "",
    "[mcp_servers.auth]",
    'type = "static"',
    `api_key_env = "${MCP_API_KEY_ENV}"`,
    'api_key_header = "Authorization"',
    'api_key_format = "Bearer {token}"',
  ].join("\n");
}

function dshSnippet(url: string, apiKey: string | null): string {
  // dsh has no ${VAR} substitution. Its !!js tag evaluates the template instead.
  const authorization = apiKey
    ? quoted(`Bearer ${apiKey}`)
    : `!!js '\`Bearer \${process.env.${MCP_API_KEY_ENV}}\`'`;
  return [
    "- insert:",
    "    - id: mcp-unsloth-studio",
    "      name: '@deepseek-ai/dsh-mcp-client'",
    "      config:",
    `        serverName: ${MCP_SERVER_NAME}`,
    "        transport: streamable-http",
    `        url: ${quoted(url)}`,
    "        headers:",
    `          Authorization: ${authorization}`,
  ].join("\n");
}

type Builder = (url: string, os: AgentCommandOs, key: string | null) => string;

const BUILDERS = new Map<string, Builder>([
  ["claude", claudeSnippet],
  ["codex", codexSnippet],
  ["opencode", (url, _os, apiKey) => opencodeSnippet(url, apiKey)],
  ["hermes", (url, _os, apiKey) => hermesSnippet(url, apiKey)],
  ["openclaw", (url, _os, apiKey) => openclawSnippet(url, apiKey)],
  ["vibe", vibeSnippet],
  ["dsh", (url, _os, apiKey) => dshSnippet(url, apiKey)],
]);

const CONFIG_PATHS: Record<string, string> = {
  opencode: "~/.config/opencode/opencode.json",
  hermes: "~/.hermes/config.yaml",
  openclaw: "~/.openclaw/openclaw.json",
  vibe: "~/.vibe/config.toml",
  dsh: "~/.dsh/cordis.patch.yml",
};

// These read the key from UNSLOTH_API_KEY even when the caller passes one.
const ENV_ONLY_AGENT_IDS: ReadonlySet<string> = new Set(["codex", "vibe"]);

/**
 * Setup for one agent, or null for an unknown agent or a base that is not a URL.
 * A literal key appears only when `apiKey` is passed.
 */
export function buildMcpSnippet(
  agentId: string,
  base: string | null | undefined,
  os: AgentCommandOs,
  apiKey: string | null = null,
): McpSnippet | null {
  const url = mcpEndpointUrl(base);
  const build = BUILDERS.get(agentId);
  if (!(url && build)) {
    return null;
  }
  const key = apiKey || null;
  return {
    text: build(url, os, key),
    configPath: CONFIG_PATHS[agentId] ?? null,
    readsKeyEnv: key === null || ENV_ONLY_AGENT_IDS.has(agentId),
  };
}
