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
]);

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

/**
 * The MLX half of that vocabulary. The methods mirror MLX_SPECULATIVE_METHODS
 * (core/inference/mlx_speculative.py); the mode list adds "off", which the backend accepts
 * through normalization rather than through that constant.
 */
export const MLX_SPECULATIVE_METHODS = ["mtp", "dflash", "eagle3"] as const;
export type MlxSpeculativeMethod = (typeof MLX_SPECULATIVE_METHODS)[number];

/** Menu order, matching the GGUF control: Auto first, Off last. */
export const MLX_SPECULATIVE_MODES = [
  "auto",
  ...MLX_SPECULATIVE_METHODS,
  "off",
] as const;
export type MlxSpeculativeMode = (typeof MLX_SPECULATIVE_MODES)[number];

/** The modes a resolution can report: `auto` is a request, never an answer. */
export const MLX_SPECULATIVE_RESOLVED_MODES = [
  "off",
  ...MLX_SPECULATIVE_METHODS,
] as const;
export type MlxSpeculativeResolvedMode =
  (typeof MLX_SPECULATIVE_RESOLVED_MODES)[number];

const VALID_MODES = new Set<string>(MLX_SPECULATIVE_MODES);

/** Draft block sizes the backend accepts, as `[min, max]`. */
export const MLX_DRAFT_BLOCK_SIZE_RANGE = [2, 16] as const;

/** `missing` covers an absent value, which carries no intent; a string naming no mode is refused. */
export function normalizeMlxSpeculativeMode(
  value: unknown,
  missing: MlxSpeculativeMode = "off",
): MlxSpeculativeMode {
  if (typeof value !== "string") {
    return missing;
  }
  const normalized = value.trim().toLowerCase();
  return VALID_MODES.has(normalized)
    ? (normalized as MlxSpeculativeMode)
    : "off";
}

export function normalizeMlxSpeculativeMethod(
  value: unknown,
): MlxSpeculativeMethod | null {
  const mode = normalizeMlxSpeculativeMode(value);
  return mode === "off" || mode === "auto" ? null : mode;
}

/** Auto re-resolves its drafter each load, so a pin alongside it would pin Auto's choice. */
export function normalizeMlxDraftModel(
  value: unknown,
  mode: MlxSpeculativeMode,
): string | null {
  return normalizeMlxSpeculativeMethod(mode) !== null &&
    typeof value === "string" &&
    value.trim()
    ? value.trim()
    : null;
}

export function normalizeMlxDraftBlockSize(
  value: unknown,
  mode: MlxSpeculativeMode,
): number | null {
  const [min, max] = MLX_DRAFT_BLOCK_SIZE_RANGE;
  return mode !== "off" && typeof value === "number" && Number.isFinite(value)
    ? Math.max(min, Math.min(max, Math.round(value)))
    : null;
}

/** Off-backend the setting collapses to Off rather than travelling as a value to compare. */
export function mlxSpeculativeLoadFields(
  intent: {
    mlxSpeculativeMode?: unknown;
    mlxDraftModel?: unknown;
    mlxDraftBlockSize?: unknown;
  },
  enabled: boolean,
): {
  // biome-ignore lint/style/useNamingConvention: backend API schema
  mlx_speculative_mode: MlxSpeculativeMode;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  mlx_draft_model: string | null;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  mlx_draft_block_size: number | null;
} {
  const mode = enabled
    ? normalizeMlxSpeculativeMode(intent.mlxSpeculativeMode, "auto")
    : "off";
  return {
    // biome-ignore lint/style/useNamingConvention: backend API schema
    mlx_speculative_mode: mode,
    // biome-ignore lint/style/useNamingConvention: backend API schema
    mlx_draft_model: normalizeMlxDraftModel(intent.mlxDraftModel, mode),
    // biome-ignore lint/style/useNamingConvention: backend API schema
    mlx_draft_block_size: normalizeMlxDraftBlockSize(
      intent.mlxDraftBlockSize,
      mode,
    ),
  };
}

