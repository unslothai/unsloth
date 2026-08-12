// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Advanced panel's resolved-control decisions, as pure functions the pages render.
 *
 * The old badge rule was `source === "auto"`, so an EXPLICIT precision the backend declined
 * rendered no badge at all while the dropdown kept advertising the request: a Q4_K_M GGUF could
 * show FP8 with transformer FP8 disabled, and a successfully generated image proved nothing about
 * the precision that ran. These helpers decide from BOTH sides of the record -- what was asked for
 * and what engaged -- so a fallback is impossible to miss, and they seed the Advanced selects from
 * the loaded model instead of leaving them as pure local request state.
 */

// One Advanced control's engaged value + provenance. Structurally shared by the image and video
// status payloads; declared here so this module needs neither feature's api.ts.
export interface ResolvedControl {
  value: string | boolean | null;
  // What the caller asked for, or null when they left it to the backend. Absent on older backends.
  requested?: string | boolean | null;
  source: "auto" | "explicit";
  // "applied" | "fell_back" | "unsupported". Absent on older backends, which only ever applied.
  status?: string;
  reason: string;
}

export type ResolvedBadgeTone = "auto" | "warn";

export interface ResolvedBadgeInfo {
  // Full badge text, e.g. "Auto: FP8" or "FP8 -> OFF".
  label: string;
  // "auto" is the neutral informational pill; "warn" flags a request that did not survive.
  tone: ResolvedBadgeTone;
  // Hover tooltip: the backend's reason, prefixed for a declined request so the pill is readable
  // even when the reason alone reads like a statement of fact.
  tooltip: string;
}

// The backend refuses an EXPLICIT precision it cannot honor rather than loading at some other one
// (409 from /images/load and /video/load, or an error phase on load-progress once the decline can
// only be found mid-load). The detail is one long actionable sentence, so the pages show it as a
// toast DESCRIPTION under this title instead of as an unreadable single line.
export const PRECISION_REFUSAL_TITLE = "Requested precision is not available";

/** Whether a load failure is that refusal, so it can be presented as an actionable choice. */
export function isPrecisionRefusal(message: string): boolean {
  return /_quant='[^']*' could not be used/.test(message);
}

/** Whether an engaged/requested value means "this control is off". */
function isOff(value: string | boolean | null | undefined): boolean {
  if (value === null || value === undefined || value === false) return true;
  if (value === true) return false;
  const text = String(value).trim().toLowerCase().replace(/-/g, "_");
  return text === "" || text === "none" || text === "off" || text === "0";
}

/** The engaged value of a resolved Advanced control, formatted for its badge. */
export function formatResolvedValue(key: string, value: string | boolean | null | undefined): string {
  if (key === "cpu_offload") return value ? "On" : "Off";
  if (value === null || value === undefined || value === "") return "Off";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === "_native_cudnn" || value.toLowerCase() === "cudnn") return "cuDNN";
  // Deferred speed auto: the dense pipe stays exact/eager and compiles on the 3rd image (the tooltip carries the full reason).
  if (value === "deferred") return "On from 3rd image";
  return value.toUpperCase();
}

/**
 * Whether the caller's request survived. True when they left the control to the backend (there was
 * no request to betray) or when the backend says it applied it.
 *
 * The `status` field is authoritative rather than a `requested !== value` string comparison,
 * because several controls answer in a DIFFERENT vocabulary than they are asked in: memory_mode
 * takes "low_vram" and reports the offload policy "sequential", attention_backend takes "cudnn" and
 * reports "_native_cudnn". Comparing those directly would flag every honored request as a
 * fallback. The backend, which owns both vocabularies, classifies it once. Older backends send no
 * `status` at all, so they fall back to the mismatch test, which is still right for the precision
 * controls (the ones P1-2 is about) since those echo their own vocabulary.
 *
 * Only the two statuses that MEAN a decline are read as one. `status` is deliberately typed wider
 * than the backend's union so a NEWER backend's fourth value still parses, but treating anything
 * that is not "applied" as a failure defeats that: an unknown status would paint a red
 * "FP8 -> FP8" over a request that was honored, and on memory_mode (asked "low_vram", answered
 * "sequential") a "LOW_VRAM -> SEQUENTIAL" that never happened. An unrecognised status is not a
 * decline -- staying quiet is the safe direction, since the build that adds a status ships the
 * frontend that understands it.
 */
export function isResolvedHonored(resolved: ResolvedControl | undefined | null): boolean {
  if (!resolved) return true;
  if (resolved.source === "auto") return true;
  if (resolved.status) return resolved.status !== "fell_back" && resolved.status !== "unsupported";
  // Older backend: no status field. Compare directly, treating every "off" spelling as equal.
  const requested = resolved.requested;
  if (requested === undefined || requested === null) return true;
  if (isOff(requested) && isOff(resolved.value)) return true;
  return (
    String(requested).trim().toLowerCase().replace(/-/g, "_") ===
    String(resolved.value ?? "").trim().toLowerCase().replace(/-/g, "_")
  );
}

/**
 * The badge for one Advanced control, or null when there is nothing to say (the caller set it and
 * got it). A backend decision reads "Auto: X"; a declined request reads "FP8 -> OFF" so the ask and
 * the outcome are both on screen.
 */
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

/**
 * The value an Advanced select should show for the LOADED model, mapped into that select's option
 * vocabulary by `toOption`. Returns null when the record says nothing useful, so the caller keeps
 * whatever the user has typed.
 *
 *   - the backend chose it        -> "auto" (the badge already names what that resolved to)
 *   - an honored explicit request -> that request
 *   - a DECLINED explicit request -> the value that actually engaged
 *
 * The last line is the P1-2 fix on the input side: the selects were pure local request state, so a
 * declined FP8 stayed selected indefinitely and the interface went on displaying a precision the
 * loaded model was not running.
 */
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
 * The key the pages run their Advanced-reseed effect on: the LOAD-TIME half of the resolved record.
 *
 * NOT the whole record serialized. The backend rewrites entries of it at GENERATION time -- the
 * speed_mode and attention_backend entries when the deferred compile profile engages on the 3rd
 * image, the transformer_cache entry whenever the step-cache threshold flips -- so keying the
 * effect on the whole blob re-ran the reseed mid-session and overwrote a Precision the user had
 * picked but not yet loaded. That is the opposite of the intent: an edit made after the load is
 * meant to survive until the next LOAD replaces it.
 *
 * Only the controls the reseed writes are in the key, and for attention only the REQUEST side:
 * `value` is the field the generation-time rewrite touches, and the reseed's answer for an auto or
 * an honored request does not read it anyway. A reload still re-fires, including a Reapply with
 * new options, because that always lands a different request or engaged value on one of these.
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
    part(resolved.memory_mode, true),
    part(resolved.attention_backend, false),
  ].join("|");
}

/**
 * Whether a status describes the native sd.cpp engine rather than a diffusers pipeline.
 *
 * It reports `dtype: "gguf"` and no `engine`/`model_kind` of its own, and the two behave
 * differently in ways the Loaded-build panel has to say out loud -- its attention is chosen by
 * native flags, not by the PyTorch dispatcher, and its components are whatever the checkpoint
 * and its companion bundle hold.
 */
export function isNativeEngineStatus(status: {
  engine?: string | null;
  dtype?: string | null;
}): boolean {
  const engine = String(status.engine ?? "").trim().toLowerCase();
  if (engine) return engine.includes("sd_cpp") || engine.includes("sd.cpp") || engine === "native";
  return status.dtype === "gguf";
}

/**
 * The Loaded-build panel's Transformer row when no dense quant engaged.
 *
 * Shared by the image and video pages because the mistake it prevents is the same on both: the
 * default arm is BF16, and only a full diffusers repo actually is. A GGUF pick runs the
 * checkpoint's own quantisation, so the BF16 arm has to be the LAST resort.
 *
 * `dtype === "gguf"` is the native sd.cpp engine, which reports no `model_kind` at all -- without
 * that arm every native GGUF load, the default CPU path, was labelled BF16 in the one panel whose
 * whole job is to say what actually loaded.
 *
 * A single_file load is NOT "as in checkpoint": `from_single_file` is handed the resolved
 * `torch_dtype`, so an fp8 safetensors is upcast on load (the memory planner budgets the ~2x for
 * exactly that). Reporting the storage precision there hid the runtime dtype, which is the one
 * thing this row exists to state, so it reads the dtype like any other dense load.
 */
export function denseTransformerBuildLabel(status: {
  model_kind?: string | null;
  dtype?: string | null;
}): string {
  if (status.model_kind === "gguf" || status.dtype === "gguf") return "GGUF (as-is)";
  return denseDtypeLabel(status.dtype);
}

/**
 * The dtype the pipeline actually loaded in, not the one the happy path uses.
 *
 * A CPU diffusers load reports float32, an older accelerator resolves to float16, and an
 * fp16-incompatible family is promoted to float32 by the video loader -- and all three were
 * labelled BF16 by the panel whose whole job is to say what loaded. Unknown falls back to BF16,
 * which is what a diffusers load that reports nothing is.
 */
function denseDtypeLabel(dtype: string | null | undefined): string {
  const text = String(dtype ?? "").trim().toLowerCase();
  if (text.includes("bfloat16") || text === "bf16") return "BF16";
  if (text.includes("float16") || text === "fp16") return "FP16";
  if (text.includes("float32") || text === "fp32") return "FP32";
  if (text.includes("float64")) return "FP64";
  return "BF16";
}

/**
 * The Loaded-build panel's Text encoder row when no runtime text-encoder quant engaged.
 *
 * The native sd.cpp engine has no runtime TE quant at all, so its status always reports
 * `text_encoder_quant: null` -- which is not evidence of a bf16 encoder. Its companion bundle is
 * whatever the family's asset mapping names, and several are not bf16 (FLUX.1 loads
 * `t5xxl_fp16.safetensors`). A null on that engine means "as stored", not BF16.
 */
export function denseTextEncoderBuildLabel(status: { dtype?: string | null }): string {
  return status.dtype === "gguf" ? "As in checkpoint" : denseDtypeLabel(status.dtype);
}

/**
 * The Recipe popover's Memory row: the placement that actually ran.
 *
 * The two fields answer different questions and either can be absent. `memory_mode` is the
 * torchao-side memory planner's pick, and the native sd.cpp engine never runs it, so its records
 * carry `memory_mode: null` alongside a real `offload_policy`. Substituting "auto" there claimed
 * the planner had chosen a mode on a path that has no planner -- so an absent mode reports the
 * offload alone, which is all that engine knows.
 */
export function memoryRecipeValue(
  memoryMode: string | null | undefined,
  offloadPolicy: string | null | undefined,
): string {
  const offloading = offloadPolicy != null && offloadPolicy !== "" && offloadPolicy !== "none";
  if (!offloading) return memoryMode ?? "";
  if (!memoryMode) return `${offloadPolicy} offload`;
  return `${memoryMode} (${offloadPolicy} offload)`;
}
