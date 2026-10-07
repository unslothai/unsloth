// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** In lib/ to avoid a feature-boundary import cycle. Mirrors the backend's _CANONICAL_SPEC_MODES. */
export const SPECULATIVE_TYPES = [
  "auto",
  "mtp",
  "dspark",
  "dflash",
  "ngram",
  "mtp+ngram",
  "off",
] as const;

/** Modes using spec_draft_n_max. Mirrors DRAFT_N_MAX_SPEC_TYPES (openai_auto_switch_settings.py). */
export const DRAFT_N_MAX_SPEC_TYPES: ReadonlySet<string> = new Set([
  "mtp",
  "mtp+ngram",
  "dspark",
  "dflash",
]);

/** MTP is excluded: whether it attaches a drafter file depends on the model. */
export const SEPARATE_DRAFT_MODEL_SPEC_TYPES: ReadonlySet<string> = new Set([
  "dspark",
  "dflash",
]);
