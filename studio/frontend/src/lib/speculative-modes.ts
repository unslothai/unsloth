// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The speculative-decoding vocabulary shared by the model picker and chat.
 *
 * Lives in lib/, not in either feature: both read it, and a low-level module
 * like the chat runtime store cannot import the model-picker barrel (the
 * feature-boundary lint rule bans deep imports, and the barrel pulls the
 * picker's components back into an eval-time cycle with chat).
 *
 * Mirrors the backend's _CANONICAL_SPEC_MODES (core/inference/llama_cpp.py).
 */
export const SPECULATIVE_TYPES = [
  "auto",
  "mtp",
  "dspark",
  "ngram",
  "mtp+ngram",
  "off",
] as const;

/**
 * The modes that consume spec_draft_n_max, i.e. the ones that launch a drafter
 * with a configurable depth. Named for the setting rather than for MTP: DSpark
 * is in here too. Mirrors DRAFT_N_MAX_SPEC_TYPES in
 * studio/backend/utils/openai_auto_switch_settings.py.
 */
export const DRAFT_N_MAX_SPEC_TYPES: ReadonlySet<string> = new Set([
  "mtp",
  "mtp+ngram",
  "dspark",
]);
