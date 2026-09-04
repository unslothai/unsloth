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
export const MLX_SPECULATIVE_METHODS = [
  "mtp",
  "dspark",
  "dflash2",
  "dflash",
  "eagle3",
] as const;
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

/** The same range as the user states it: the block counts the verified token too. */
export const MLX_DRAFT_TOKENS_RANGE = [
  MLX_DRAFT_BLOCK_SIZE_RANGE[0] - 1,
  MLX_DRAFT_BLOCK_SIZE_RANGE[1] - 1,
] as const;

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
  // Auto chooses the depth its method pays off at, and the control is hidden while it is
  // selected. Carrying across a depth set for an explicit method would keep overriding that
  // choice from a field the user can no longer see or clear.
  return normalizeMlxSpeculativeMethod(mode) !== null &&
    typeof value === "number" &&
    Number.isFinite(value)
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

export interface MlxSpeculativeCandidate {
  method: MlxSpeculativeMethod;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  repo_id: string;
  label: string;
  source: "builtin" | "cached" | "recommended";
  /** Whether the curated index proposes this checkpoint, which the cache does not retract. */
  recommended: boolean;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  approximate_size_bytes: number;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  estimated_memory_bytes: number;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  materialization_bytes: number;
  downloaded: boolean;
  compatible: boolean;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  runtime_supported: boolean;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  integration_ready: boolean;
  loadable: boolean;
  reason: string | null;
}

export interface MlxSpeculativeOptions {
  // biome-ignore lint/style/useNamingConvention: backend API schema
  target_model: string;
  experimental: boolean;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  runtime_supported: boolean;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  runtime_reason: string | null;
  candidates: MlxSpeculativeCandidate[];
  /** What Auto would run for this target, decided by the backend rather than re-derived. */
  // biome-ignore lint/style/useNamingConvention: backend API schema
  auto_method: MlxSpeculativeResolvedMode;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  auto_draft_model: string | null;
  // biome-ignore lint/style/useNamingConvention: backend API schema
  auto_reason: string | null;
}

/** Listing order between the methods, cheapest drafting first. Not a depth or a count. */
const METHOD_ORDER: Record<MlxSpeculativeMethod, number> = {
  mtp: 1,
  dspark: 2,
  dflash2: 3,
  dflash: 4,
  eagle3: 5,
};

/**
 * The order companions are listed in. Not Auto's own order, which the backend decides and
 * additionally ranks by precision, from cache state no candidate row carries. Companions
 * only: the caller has already dropped the target's own head.
 */
function compareListedCandidates(
  a: MlxSpeculativeCandidate,
  b: MlxSpeculativeCandidate,
): number {
  if (METHOD_ORDER[a.method] !== METHOD_ORDER[b.method]) {
    return METHOD_ORDER[a.method] - METHOD_ORDER[b.method];
  }
  const [left, right] = [a.repo_id.toLowerCase(), b.repo_id.toLowerCase()];
  return left < right ? -1 : left > right ? 1 : 0;
}

/**
 * The row a mode would draft with, so the panel can name it before a load. Auto's own pick
 * comes from the backend: two of its rules turn on the target rather than on any one
 * drafter, and neither is visible in the rows ranked here.
 */
export function selectMlxSpeculativeCandidate(
  candidates: readonly MlxSpeculativeCandidate[],
  mode: MlxSpeculativeMode,
  preferredRepo: string | null | undefined,
  autoDraftModel?: string | null,
): MlxSpeculativeCandidate | null {
  if (mode === "off") {
    return null;
  }
  if (mode === "auto") {
    const picked = autoDraftModel?.trim().toLowerCase();
    return picked
      ? (candidates.find((c) => c.repo_id.toLowerCase() === picked) ?? null)
      : null;
  }
  const matching = candidates.filter((c) => c.method === mode);
  const preferred = preferredRepo?.trim().toLowerCase();
  return preferred
    ? (matching.find((c) => c.repo_id.toLowerCase() === preferred) ?? null)
    : // One that can run now before one a download away, which is the order the list is
      // offered in. Taking the first selectable instead pins the recommended download over
      // a checkpoint already on disk, since recommendations are listed first.
      (matching.find((c) => c.loadable) ??
        matching.find(isSelectableMlxDraftCandidate) ??
        null);
}

/**
 * Auto and Off are never ruled out: Off runs nothing, and Auto is what still works when
 * the listing of drafters does not.
 */
export function isUnavailableMlxSpeculativeMode(
  candidates: readonly MlxSpeculativeCandidate[],
  option: MlxSpeculativeMode,
): boolean {
  return (
    normalizeMlxSpeculativeMethod(option) !== null &&
    selectMlxSpeculativeCandidate(candidates, option, null) === null
  );
}

/**
 * A pin is named even when it resolves to nothing, since it is what the user must see to
 * replace. Fetching is offered only for the checkpoint this request would draft with.
 */
export function mlxDraftRowCheckpoint(
  selected: MlxSpeculativeCandidate | null,
  resolved: MlxSpeculativeCandidate | null,
): { shown: MlxSpeculativeCandidate | null; fetchable: boolean } {
  const shown = selected ?? resolved;
  return {
    shown,
    fetchable:
      shown !== null && !shown.downloaded && shown.repo_id === resolved?.repo_id,
  };
}

/**
 * Not verdicts. The first is the drafter the picker itself can fetch; the rest are the
 * backend's _UNPROVEN_REASONS (core/inference/mlx_speculative.py), which it refuses to
 * answer from a target that is not on disk yet and settles once the load supplies it.
 */
const UNSETTLED_REASONS = new Set([
  "checkpoint_not_downloaded",
  "tokenizer_contract_unavailable",
  "target_config_unavailable",
  "target_weights_unmeasured",
]);

/** Loadable now, or still to be settled by the load: what neither can fix stays out. */
export function isSelectableMlxDraftCandidate(
  candidate: MlxSpeculativeCandidate,
): boolean {
  return (
    candidate.loadable ||
    (candidate.source !== "builtin" &&
      candidate.compatible &&
      candidate.runtime_supported &&
      candidate.integration_ready &&
      UNSETTLED_REASONS.has(candidate.reason ?? ""))
  );
}

/** Companions only: a target's own head is nothing to choose or download. Ready sorts first. */
export function selectableExternalMlxDraftCandidates(
  candidates: readonly MlxSpeculativeCandidate[],
): MlxSpeculativeCandidate[] {
  return candidates
    .filter((c) => c.source !== "builtin" && isSelectableMlxDraftCandidate(c))
    .sort(
      (a, b) =>
        Number(b.loadable) - Number(a.loadable) || compareListedCandidates(a, b),
    );
}

/** The companion a pin names, or null when it names none or names the target's head. */
export function selectExternalMlxDraftCandidate(
  candidates: readonly MlxSpeculativeCandidate[],
  preferredRepo: string | null | undefined,
): MlxSpeculativeCandidate | null {
  const preferred = preferredRepo?.trim().toLowerCase();
  return preferred
    ? (candidates.find(
        (c) => c.source !== "builtin" && c.repo_id.toLowerCase() === preferred,
      ) ?? null)
    : null;
}

/** Picking a checkpoint picks the method it implements; the two are one setting. */
export function mlxDraftSelection(candidate: MlxSpeculativeCandidate): {
  mlxSpeculativeMode: MlxSpeculativeMethod;
  mlxDraftModel: string;
} {
  return {
    mlxSpeculativeMode: candidate.method,
    mlxDraftModel: candidate.repo_id,
  };
}

