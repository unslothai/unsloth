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
  "dflash",
  "ngram",
  "mtp+ngram",
  "off",
] as const;

/** Values only an MLX load reads (studio/backend/core/inference/mlx_speculative.py). */
export const MLX_ONLY_SPEC_TYPES = ["eagle3"] as const;

/** What the MLX control offers, in its order. */
export const MLX_SPECULATIVE_TYPES = [
  "auto",
  "mtp",
  "dflash",
  "dspark",
  "eagle3",
  "ngram",
  "off",
] as const;

/** The mode an MLX load runs: every drafter kind also copies repeated text, so llama.cpp's `mtp+ngram` is `mtp`. */
export function mlxSpeculativeMode(mode: string): string {
  return mode === "mtp+ngram" ? "mtp" : mode;
}

/**
 * The modes that consume spec_draft_n_max, i.e. the ones that launch a drafter
 * with a configurable depth. Named for the setting rather than for MTP: DSpark
 * and DFlash are in here too. Mirrors DRAFT_N_MAX_SPEC_TYPES in
 * studio/backend/utils/openai_auto_switch_settings.py.
 */
export const DRAFT_N_MAX_SPEC_TYPES: ReadonlySet<string> = new Set([
  "mtp",
  "mtp+ngram",
  "dspark",
  "dflash",
  ...MLX_ONLY_SPEC_TYPES,
]);

/** The modes under which an MLX load reads a named companion drafter (spec_draft_model). */
export const DRAFTER_MODEL_SPEC_TYPES: ReadonlySet<string> = new Set([
  ...DRAFT_N_MAX_SPEC_TYPES,
  "auto",
]);

/** The mode a load sends: the model's own choice, else the standing preference, which GGUF loads write.
 *  On MLX its ngram reads as auto: n-gram copying alone would move a text model onto the vision runtime. */
export function resolveSpeculativeType(
  chosen: string | null,
  standing: string,
  isMlx: boolean,
): string {
  return chosen ?? (isMlx && standing === "ngram" ? "auto" : standing);
}

/**
 * The modes that always launch a SEPARATE draft model, and so a second context
 * with its own KV cache for the draft cache dtype to apply to.
 *
 * MTP is left out: whether it loads a drafter file (Gemma) or reads baked-in
 * heads out of the target GGUF (Qwen) is a property of the model, known only once
 * the loader has read its metadata. The backend emits the draft cache flags
 * wherever it emits --model-draft, so a stored setting still reaches an MTP load
 * that does attach one.
 */
export const SEPARATE_DRAFT_MODEL_SPEC_TYPES: ReadonlySet<string> = new Set([
  "dspark",
  "dflash",
]);
