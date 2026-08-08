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
 */
export function isResolvedHonored(resolved: ResolvedControl | undefined | null): boolean {
  if (!resolved) return true;
  if (resolved.source === "auto") return true;
  if (resolved.status) return resolved.status === "applied";
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
