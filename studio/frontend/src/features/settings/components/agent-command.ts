// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The bare `unsloth start` only targets 127.0.0.1:8888 and auto-mints keys for loopback, so other
// servers need UNSLOTH_STUDIO_URL (and a key when non-loopback).

const DEFAULT_STUDIO_PORT = "8888";
const DEFAULT_AGENT = "claude";
const SAFE_SHELL_ARG_PATTERN = /^[A-Za-z0-9_./:@%+=,-]+$/;

export type AgentCommandOs = "unix" | "windows";

export const shSingle = (value: string): string =>
  value.replace(/'/g, "'\\''");
export const psSingle = (value: string): string => value.replace(/'/g, "''");

export function quoteShellArg(value: string, os: AgentCommandOs): string {
  if (SAFE_SHELL_ARG_PATTERN.test(value)) {
    return value;
  }
  return os === "windows" ? `'${psSingle(value)}'` : `'${shSingle(value)}'`;
}

// URL.hostname keeps IPv6 brackets ("[::1]").
export function normalizeHost(host: string): string {
  const lower = host.toLowerCase();
  return lower.startsWith("[") && lower.endsWith("]") ? lower.slice(1, -1) : lower;
}

// Only 127.0.0.1 gets the bare command: localhost can resolve to ::1, which the CLI never probes.
function isDefaultLocalHost(host: string): boolean {
  return host === "127.0.0.1";
}

// Mirrors the CLI is_loopback_url rule: localhost, ::1, and 127.0.0.0/8.
export function isLoopbackHost(host: string): boolean {
  if (host === "localhost" || host === "::1") return true;
  const octets = host.split(".");
  return (
    octets.length === 4 &&
    octets[0] === "127" &&
    octets.every((o) => /^\d{1,3}$/.test(o) && Number(o) <= 255)
  );
}

export function buildAgentCommand(
  base: string | null | undefined,
  key: string | null | undefined,
  os: AgentCommandOs,
  agent: string = DEFAULT_AGENT,
): string {
  const bare = `unsloth start ${agent}`;

  let url: URL | null = null;
  try {
    if (base) url = new URL(base);
  } catch {
    url = null;
  }
  if (!url) return bare;

  const host = normalizeHost(url.hostname);
  const loopback = isLoopbackHost(host);
  // The bare default probes plain HTTP, so HTTPS loopback keeps an explicit URL.
  if (url.protocol === "http:" && isDefaultLocalHost(host) && url.port === DEFAULT_STUDIO_PORT) {
    return bare;
  }

  let cmd = bare;
  if (!loopback && key) cmd += ` --api-key ${key}`;

  const studioUrl = url.origin;
  return os === "windows"
    ? `$env:UNSLOTH_STUDIO_URL="${studioUrl}"; ${cmd}`
    : `UNSLOTH_STUDIO_URL=${studioUrl} ${cmd}`;
}

export interface AgentShellCommands {
  primary: string;
  subagent: string;
  remoteSetup: string;
  passThrough: string[];
  dryRun: string;
}

const REMOTE_SETUP_COMMANDS: Record<AgentCommandOs, string> = {
  unix: `export UNSLOTH_STUDIO_URL=https://studio.example.com
export UNSLOTH_API_KEY=sk-unsloth-...
unsloth start claude`,
  windows: `$env:UNSLOTH_STUDIO_URL = "https://studio.example.com"
$env:UNSLOTH_API_KEY = "sk-unsloth-..."
unsloth start claude`,
};

function appendCommand(command: string, args: string): string {
  return args ? `${command} ${args}` : command;
}

export function buildAgentShellCommands(
  base: string | null | undefined,
  os: AgentCommandOs,
  agent: string,
  modelArgs: string,
): AgentShellCommands {
  const primaryBase = buildAgentCommand(base, null, os, agent);
  return {
    primary: appendCommand(primaryBase, modelArgs),
    subagent: appendCommand(
      `${primaryBase} --as-subagent`,
      modelArgs,
    ),
    remoteSetup: REMOTE_SETUP_COMMANDS[os],
    passThrough: [
      `${buildAgentCommand(base, null, os, "claude")} --continue`,
      `${buildAgentCommand(base, null, os, "codex")} --persist resume --last`,
    ],
    dryRun: `${buildAgentCommand(base, null, os, "claude")} --no-launch`,
  };
}

// Codex (/v1/responses) and Claude Code (/v1/messages) are llama-server only.
export const GGUF_ONLY_AGENTS: readonly string[] = ["codex", "claude"];

export function agentRunsOnActiveModel(
  agent: string,
  isGguf: boolean,
): boolean {
  return isGguf || !GGUF_ONLY_AGENTS.includes(agent);
}

// DEFAULT_AGENT is itself GGUF-only.
export const UNIVERSAL_AGENT = "opencode";

// Stays inside `offered`; null = nothing offered runs.
export function fallbackAgent(
  isGguf: boolean,
  offered: readonly string[] = [],
): string | null {
  const runs = (agent: string) => agentRunsOnActiveModel(agent, isGguf);
  if (offered.length === 0) {
    return runs(DEFAULT_AGENT) ? DEFAULT_AGENT : UNIVERSAL_AGENT;
  }
  for (const preference of [DEFAULT_AGENT, UNIVERSAL_AGENT]) {
    if (offered.includes(preference) && runs(preference)) {
      return preference;
    }
  }
  return offered.find(runs) ?? null;
}

export function pickCompatibleAgent(
  detectedAgents: readonly string[],
  currentAgent: string,
  isGguf: boolean,
  offered: readonly string[] = [],
): string | null {
  const preferred = detectedAgents.find((agent) =>
    agentRunsOnActiveModel(agent, isGguf),
  );
  if (preferred) {
    return preferred;
  }
  return agentRunsOnActiveModel(currentAgent, isGguf)
    ? null
    : fallbackAgent(isGguf, offered);
}

// The server wins since only it sees swaps from other tabs; null from both is unknown, not false.
export function resolveGgufCompatibility(
  fromStore: boolean | null,
  fromServer: boolean | null,
): boolean | null {
  return fromServer ?? fromStore;
}

// is_gguf defaults to False, so only a status naming a model is a verdict.
export function statusGgufVerdict(
  resident: string | null | undefined,
  isGguf: boolean | null | undefined,
): boolean | null {
  if (resident == null) return null;
  return isGguf ?? null;
}

export function sameBaseModelId(a: string, b: string): boolean {
  const base = (id: string) => id.trim().toLowerCase().split(":")[0];
  return (
    a.trim().toLowerCase() === b.trim().toLowerCase() || base(a) === base(b)
  );
}

// Status and catalog polls run on separate timers, so a swap can reach one first.
export function verdictDescribesModel(
  resident: string | null | undefined,
  named: string | null | undefined,
): boolean {
  if (resident == null || named == null) return true;
  return sameBaseModelId(resident, named);
}

export interface StatusAnswer {
  resident: string | null;
  isGguf: boolean | null;
}

// Disagreeing polls stay unknown; the store is blind to the swap that caused the disagreement.
export function compatibilityFromSources(
  fromStore: boolean | null,
  status: StatusAnswer | null,
  namedModel: string | null,
): boolean | null {
  if (status === null) {
    return fromStore;
  }
  if (!verdictDescribesModel(status.resident, namedModel)) {
    return null;
  }
  return resolveGgufCompatibility(fromStore, status.isGguf);
}
