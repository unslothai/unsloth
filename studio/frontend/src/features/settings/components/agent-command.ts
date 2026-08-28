// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Build the `unsloth start <agent>` command for the API-keys panel.
// `unsloth start` reads UNSLOTH_STUDIO_URL (default 127.0.0.1:8888) and only
// auto-mints a key for a loopback server, so the bare command is correct only for
// the default local server. For a non-default port or tunnel/remote base, emit the
// URL (plus a key for non-loopback) so the copy targets what the UI shows.

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

// URL.hostname brackets IPv6 literals (`new URL("http://[::1]:8888").hostname` is
// "[::1]"), so strip the brackets before matching the bare "::1" loopback rules below.
export function normalizeHost(host: string): string {
  const lower = host.toLowerCase();
  return lower.startsWith("[") && lower.endsWith("]") ? lower.slice(1, -1) : lower;
}

// The bare `unsloth start` probes exactly http://127.0.0.1:8888, so only that literal
// host earns the bare command. `localhost` can resolve to ::1 (and `::1` is never
// probed), so both keep an explicit UNSLOTH_STUDIO_URL -- harmless when they alias
// 127.0.0.1, correct when they don't.
function isDefaultLocalHost(host: string): boolean {
  return host === "127.0.0.1";
}

// Match the CLI auto-mint rule (is_loopback_url): localhost, ::1, and all of 127.0.0.0/8.
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
  // Unknown base: fall back to the bare default-local command.
  if (!url) return bare;

  const host = normalizeHost(url.hostname);
  const loopback = isLoopbackHost(host);
  // Default local server (http://127.0.0.1/localhost:8888): bare command
  // auto-discovers it. The CLI's bare default probes plain HTTP, so an HTTPS
  // loopback on the same port must keep its explicit UNSLOTH_STUDIO_URL.
  if (url.protocol === "http:" && isDefaultLocalHost(host) && url.port === DEFAULT_STUDIO_PORT) {
    return bare;
  }

  // Non-default server: set the URL; non-loopback also needs an explicit key.
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
