// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Tools the BROWSER must execute, forcing the cancel-on-disconnect stream. Intentionally empty:
 * every current tool runs on the server. Add one only if it truly needs a live tab.
 */
export const BROWSER_EXECUTED_TOOLS: ReadonlySet<string> = new Set<string>([]);

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
