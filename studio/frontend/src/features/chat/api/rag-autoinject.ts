// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a turn pre-retrieves from the attached documents before the model answers.
 *
 * Two independent settings decide this and they are easy to confuse, which is what
 * #9947 was really about:
 *
 * - The **Search** pill gates `web_search`. It has never gated `search_knowledge_base`,
 *   so it must not decide whether documents are consulted.
 * - **Auto-retrieve** (Auto / On / Off) decides pre-retrieval. Its own copy promises
 *   "On and Off force it either way", so an explicit Off is a user decision, not a
 *   default to be improved upon.
 *
 * Project sources sit between the two: with Search off the model is the only thing that
 * would call the tool, and small models under-call it, so a project chat pre-retrieves
 * under Auto rather than leaving grounding to chance. That is a better default, not a
 * licence to override Off -- hence the mode check comes first.
 *
 * Lifted out of chat-adapter.ts so it can be executed by a test. The adapter's
 * dependency graph (stores, toast, assistant-ui) cannot be loaded under `node --test`,
 * which is why every other chat-adapter test asserts against source text instead.
 */

import { parseParamCountB } from "@/lib/model-size";

/** Mirrors `RagAutoInject` in the chat runtime store, without importing the store. */
export type RagAutoInjectMode = "auto" | "on" | "off";

// Small models (<=9B) answer from memory instead of calling search, so "auto"
// forces retrieval for them and leaves it to larger ones.
export const AUTOINJECT_AUTO_MAX_SIZE_B = 9;

/** The size heuristic behind Auto. Exported for the test matrix. */
export function resolveAutoInject(
  mode: RagAutoInjectMode,
  checkpoint: string,
): boolean {
  if (mode === "on") {
    return true;
  }
  if (mode === "off") {
    return false;
  }
  const size = parseParamCountB(checkpoint);
  // Unknown size -> enable.
  return size === null || size <= AUTOINJECT_AUTO_MAX_SIZE_B;
}

/**
 * The `autoinject` flag to put on the wire.
 *
 * Project-only scopes are pre-retrieved so the Search pill cannot block grounding. KB and
 * mixed thread/project scopes keep the user's Auto-retrieve setting, matching
 * `build_rag_autoinject`, which forces only for a scope carrying `project_id` alone: a
 * thread attachment has to keep the caller's flag so the whole-document fallback runs
 * instead of one combined top-K search.
 *
 * An explicit Off outranks all of it. Always returns a boolean -- the deep-research
 * request builder used to put the raw mode string on the wire, where `"off"` read as ON
 * server-side.
 */
export function resolveRagAutoinject(
  mode: RagAutoInjectMode,
  checkpoint: string,
  projectOnlyScope: boolean,
): boolean {
  if (mode === "off") {
    return false;
  }
  if (projectOnlyScope) {
    return true;
  }
  return resolveAutoInject(mode, checkpoint);
}
