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

/**
 * Everything the durability gate reads, as plain values.
 *
 * Each field is exactly what the adapter already had in hand at the gate: a resolved model flag, a resolved thread
 * id, THIS turn's attachment scan. Nothing here resolves anything itself - that is what lets the whole gate be a
 * truth table instead of a grep over the adapter's source.
 */
export type DurableRunCandidate = {
  /** A passthrough/external-provider turn: its own client owns the stream, so there is nothing to make durable. */
  externalProvider?: boolean;
  /** The resolved model is an audio model - it runs on its own legacy path regardless of the gate. */
  modelIsAudio?: boolean;
  /** A diffusion model is loaded: that path has no durable run to join. */
  loadedIsDiffusion?: boolean;
  /** No longer a gate term: media turns are durable by default, and UNSLOTH_STUDIO_DURABLE_MEDIA_TURNS=0 refuses
   * them in the backend (routes/chat_generation_runs.py), where the 400 degrades silently through
   * isLegacyFallbackChatGenerationAdmissionError. Kept so a caller passing it still type-checks. */
  turnCarriesMedia?: boolean;
  /** Continue: the seeded partial is autosaved before the request starts, and admission 409s a placeholder that
   * already has content - which is not one of the errors that falls back, so the turn would just fail. The adapter
   * holds a request object here, not a flag; what the gate reads is only whether one is present. */
  continuation?: unknown;
  /** The thread the run belongs to. No thread id means nowhere to reattach to, so nothing to make durable. */
  threadId?: string | null;
  /** That thread is incognito: its runs are not persisted, so a run has no stored history to resume from. */
  incognito?: boolean;
  /** The assistant message the run writes into. Without one there is no row for a follower to reattach to. */
  assistantMessageId?: string | null;
  /** The turn's user message exists - the run has something to answer. */
  hasUserMessage?: boolean;
};

/**
 * Whether THIS turn is a candidate for a durable (server-owned, reconnectable) generation run.
 *
 * Extracted from the adapter unchanged: every term below is one `&&` of the conjunction it was written as. It lives
 * here so the gate can be tested as what it is - a truth table over plain values - instead of by regexing the
 * adapter's source and hoping a token keeps its spelling.
 */
export function isDurableRunCandidate(input: DurableRunCandidate): boolean {
  return Boolean(
    !input.externalProvider &&
      input.modelIsAudio !== true &&
      input.loadedIsDiffusion !== true &&
      !input.continuation &&
      input.threadId &&
      !input.incognito &&
      input.assistantMessageId &&
      input.hasUserMessage,
  );
}
