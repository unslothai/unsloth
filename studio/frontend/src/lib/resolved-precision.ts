// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Badges decide from both requested and engaged values, so a declined explicit request is visible. */

// Shared by image and video status payloads; declared here to avoid importing either feature.
export interface ResolvedControl {
  value: string | boolean | null;
  // Absent on older backends.
  requested?: string | boolean | null;
  source: "auto" | "explicit";
  // "applied" | "fell_back" | "unsupported". Absent on older backends, which only ever applied.
  status?: string;
  reason: string;
}

export type ResolvedBadgeTone = "auto" | "warn";

export interface ResolvedBadgeInfo {
  label: string;
  tone: ResolvedBadgeTone;
  tooltip: string;
}

// The backend refuses an explicit precision it cannot honor (409 or a load-progress error); the
// detail is shown as a toast description under this title.
export const PRECISION_REFUSAL_TITLE = "Requested precision is not available";

/** Mirrors the backend constant. */
export const DENSE_QUANT_KINDS = ["gguf", "pipeline"] as const;

export function isDenseQuantKind(kind: string | null | undefined): boolean {
  return (DENSE_QUANT_KINDS as readonly string[]).includes(
    (kind ?? "").trim().toLowerCase(),
  );
}

export function isPrecisionRefusal(message: string): boolean {
  return /_quant='[^']*' could not be used/.test(message);
}

function isOff(value: string | boolean | null | undefined): boolean {
  if (value === null || value === undefined || value === false) return true;
  if (value === true) return false;
  const text = String(value).trim().toLowerCase().replace(/-/g, "_");
  return text === "" || text === "none" || text === "off" || text === "0";
}

export function formatResolvedValue(key: string, value: string | boolean | null | undefined): string {
  if (key === "cpu_offload") return value ? "On" : "Off";
  if (value === null || value === undefined || value === "") return "Off";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === "_native_cudnn" || value.toLowerCase() === "cudnn") return "cuDNN";
  if (value === "sage_hub") return "SAGE";
  // Deferred speed auto: the dense pipe compiles on the 3rd image.
  if (value === "deferred") return "On from 3rd image";
  return value.toUpperCase();
}

/**
 * `status` is authoritative, since some controls report in a different vocabulary than requested.
 * Only explicit decline statuses count, so an unknown newer status is not shown as a failure.
 * Older backends without `status` fall back to comparing requested and engaged values.
 */
export function isResolvedHonored(resolved: ResolvedControl | undefined | null): boolean {
  if (!resolved) return true;
  if (resolved.source === "auto") return true;
  if (resolved.status) return resolved.status !== "fell_back" && resolved.status !== "unsupported";
  const requested = resolved.requested;
  if (requested === undefined || requested === null) return true;
  if (isOff(requested) && isOff(resolved.value)) return true;
  return (
    String(requested).trim().toLowerCase().replace(/-/g, "_") ===
    String(resolved.value ?? "").trim().toLowerCase().replace(/-/g, "_")
  );
}

/** Null when the caller set it and got it. A declined request shows both, e.g. "FP8 -> OFF". */
export function resolvedBadge(
  key: string,
  resolved: ResolvedControl | undefined | null,
): ResolvedBadgeInfo | null {
  if (!resolved) return null;
  const engaged = formatResolvedValue(key, resolved.value);
  if (!isResolvedHonored(resolved)) {
    const asked = formatResolvedValue(key, resolved.requested ?? null);
    const verb = resolved.status === "unsupported" ? "not supported" : "not applied";
    return {
      label: `${asked} → ${engaged}`,
      tone: "warn",
      tooltip: resolved.reason
        ? `You requested ${asked}; ${engaged} was used because ${resolved.reason}.`
        : `You requested ${asked}, but it was ${verb} here; ${engaged} was used instead.`,
    };
  }
  if (resolved.source !== "auto") return null;
  return { label: `Auto: ${engaged}`, tone: "auto", tooltip: resolved.reason };
}

/** A declined explicit request maps to the engaged value, so the select stops showing a stale ask. */
export function resolvedSelectValue<T extends string>(
  resolved: ResolvedControl | undefined | null,
  toOption: (value: string) => T | null,
): T | null {
  if (!resolved) return null;
  if (resolved.source === "auto") return toOption("auto");
  const source = isResolvedHonored(resolved) ? resolved.requested : resolved.value;
  if (source === undefined) return null;
  if (source === null) return toOption("none");
  if (typeof source === "boolean") return toOption(source ? "on" : "none");
  return toOption(String(source));
}

/**
 * Keys only the load-time half of the record: the backend rewrites some entries at generation
 * time, which would reseed mid-session and overwrite a Precision picked but not yet loaded.
 * Attention keys only its request, since generation rewrites its `value`.
 */
export function resolvedSeedKey(
  resolved: Record<string, ResolvedControl> | null | undefined,
): string | null {
  if (!resolved) return null;
  const part = (control: ResolvedControl | undefined, withValue: boolean): string => {
    if (!control) return "";
    const engaged = withValue ? String(control.value ?? "") : "";
    return `${control.source}:${String(control.requested ?? "")}:${engaged}`;
  };
  return [
    part(resolved.transformer_quant, true),
    part(resolved.text_encoder_quant, true),
    part(resolved.memory_mode, true),
    part(resolved.attention_backend, false),
    part(resolved.family_override, true),
  ].join("|");
}

/** sd.cpp reports `dtype: "gguf"` with no `engine`/`model_kind`. */
export function isNativeEngineStatus(status: {
  engine?: string | null;
  dtype?: string | null;
}): boolean {
  const engine = String(status.engine ?? "").trim().toLowerCase();
  if (engine) return engine.includes("sd_cpp") || engine.includes("sd.cpp") || engine === "native";
  return status.dtype === "gguf";
}

/**
 * BF16 is the last resort: GGUF and native sd.cpp loads run the checkpoint's own quantisation.
 * single_file loads are upcast to `torch_dtype`, so they read the dtype like any dense load.
 */
export function denseTransformerBuildLabel(status: {
  model_kind?: string | null;
  dtype?: string | null;
}): string {
  if (status.model_kind === "gguf" || status.dtype === "gguf") return "GGUF (as-is)";
  return denseDtypeLabel(status.dtype);
}

/** CPU, older accelerators and fp16-incompatible families load in other dtypes; unknown is BF16. */
function denseDtypeLabel(dtype: string | null | undefined): string {
  const text = String(dtype ?? "").trim().toLowerCase();
  if (text.includes("bfloat16") || text === "bf16") return "BF16";
  if (text.includes("float16") || text === "fp16") return "FP16";
  if (text.includes("float32") || text === "fp32") return "FP32";
  if (text.includes("float64")) return "FP64";
  return "BF16";
}

/** sd.cpp has no runtime TE quant, so a null there means "as stored", not BF16. */
export function denseTextEncoderBuildLabel(status: { dtype?: string | null }): string {
  return status.dtype === "gguf" ? "As in checkpoint" : denseDtypeLabel(status.dtype);
}

/** sd.cpp never runs the memory planner, so an absent `memory_mode` reports the offload alone. */
export function memoryRecipeValue(
  memoryMode: string | null | undefined,
  offloadPolicy: string | null | undefined,
): string {
  const offloading = offloadPolicy != null && offloadPolicy !== "" && offloadPolicy !== "none";
  if (!offloading) return memoryMode ?? "";
  if (!memoryMode) return `${offloadPolicy} offload`;
  return `${memoryMode} (${offloadPolicy} offload)`;
}
