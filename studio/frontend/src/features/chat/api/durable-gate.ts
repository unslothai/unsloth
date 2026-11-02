// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Tools the BROWSER has to execute itself, so a turn that uses one genuinely cannot proceed once the tab is gone
 * and must stay on the subscriber-owned (cancel-on-disconnect) stream.
 *
 * Every other tool Studio runs for a local model - web_search, web_fetch, python, terminal, edit_file, render_html,
 * image_generation, MCP - is executed by the SERVER and streams as chunk events, so it is durable like plain text.
 * The browser tools are the exception: they drive this tab's browser panel. Add a name here only if a tool truly
 * needs a live tab. Spelled out rather than imported from features/browser/agent/tools.ts, which the barrel cycle
 * would leave uninitialized here; tests/browser-agent-gate.test.ts keeps the two lists equal.
 */
export const BROWSER_EXECUTED_TOOLS: ReadonlySet<string> = new Set<string>([
  "browser_navigate",
  "browser_snapshot",
  "browser_find",
  "browser_click",
  "browser_move_mouse",
  "browser_drag",
  "browser_type",
  "browser_select",
  "browser_scroll",
  "browser_press_key",
  "browser_evaluate",
  "browser_screenshot",
  "browser_history",
]);

/**
 * Keys on `enabled_tools`, the resolved server-executed tool list. `requestPayload.tools` is absent
 * locally and is a server-executed catalog on passthrough, so keying on it was wrong both ways.
 */
export function turnRequiresLegacyStream(requestPayload: unknown): boolean {
  const enabled = (requestPayload as { enabled_tools?: unknown } | undefined)?.enabled_tools;
  return Array.isArray(enabled) && enabled.some((name) => BROWSER_EXECUTED_TOOLS.has(String(name)));
}

/** Plain values the adapter already resolved, so the gate is a truth table. */
export type DurableRunCandidate = {
  /** A passthrough/external-provider turn: its own client owns the stream, so there is nothing to make durable. */
  externalProvider?: boolean;
  /** The resolved model is an audio model - it runs on its own legacy path regardless of the gate. */
  modelIsAudio?: boolean;
  /** A diffusion model is loaded: that path has no durable run to join. */
  loadedIsDiffusion?: boolean;
  /** THIS turn carries an attachment (image/audio/video), scanned out of the current turn's message alone.
   * A media turn stays on the subscriber-owned stream; a text follow-up to an earlier screenshot does not. */
  turnCarriesMedia?: boolean;
  /** The seeded partial is autosaved first and admission 409s a placeholder with content. */
  continuation?: unknown;
  /** The thread the run belongs to. No thread id means nowhere to reattach to, so nothing to make durable. */
  threadId?: string | null;
  /** That thread is incognito: its runs are not persisted, so a run has no stored history to resume from. */
  incognito?: boolean;
  /** The assistant message the run writes into. Without one there is no row for a follower to reattach to. */
  assistantMessageId?: string | null;
  hasUserMessage?: boolean;
};

/** The adapter's durability conjunction, extracted unchanged so it can be tested as a truth table. */
export function isDurableRunCandidate(input: DurableRunCandidate): boolean {
  return Boolean(
    !input.externalProvider &&
      input.modelIsAudio !== true &&
      input.loadedIsDiffusion !== true &&
      // Turn-scoped, not thread-scoped: a stale blob from an earlier turn must not refuse this one.
      input.turnCarriesMedia !== true &&
      !input.continuation &&
      input.threadId &&
      !input.incognito &&
      input.assistantMessageId &&
      input.hasUserMessage,
  );
}
