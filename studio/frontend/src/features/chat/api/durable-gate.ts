// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Tools the BROWSER has to execute itself, so a turn that uses one genuinely cannot proceed once the tab is gone
 * and must stay on the subscriber-owned (cancel-on-disconnect) stream.
 *
 * Intentionally empty: every tool Studio runs for a local model - web_search, web_fetch, python, terminal,
 * edit_file, render_html, image_generation, MCP - is executed by the SERVER and streams as chunk events, so it is
 * durable like plain text. The set exists to name the exception the durable gate keys on, not to re-list the
 * server's own tools. Add a name here only if a tool truly needs a live tab.
 */
export const BROWSER_EXECUTED_TOOLS: ReadonlySet<string> = new Set<string>([]);

/**
 * Whether this turn must fall back to the legacy stream that cancels when the browser disconnects.
 *
 * The discriminating field is `enabled_tools` - the resolved list of server-executed tools for the turn.
 * `requestPayload.tools` does NOT carry this information: on the local path the key is absent entirely, and on the
 * passthrough/external-provider path it is the caller's own OpenAI/Anthropic schema catalog (see
 * backend/routes/inference.py:_passthrough_client_tools) - a server-executed catalog in both cases. Keying the gate
 * off `tools` therefore read as "no tools" for local turns and as "browser tools!" for every passthrough turn that
 * carried a catalog, which silently forced those turns back onto the cancel-on-disconnect path.
 */
export function turnRequiresLegacyStream(requestPayload: unknown): boolean {
  const enabled = (requestPayload as { enabled_tools?: unknown } | undefined)?.enabled_tools;
  return Array.isArray(enabled) && enabled.some((name) => BROWSER_EXECUTED_TOOLS.has(String(name)));
}
