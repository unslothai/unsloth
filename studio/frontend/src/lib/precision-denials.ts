// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SelectOption } from "./nvfp4-options.ts";

/**
 * The backend's `precision_options` (media status and download plan): each explicit transformer scheme mapped to the
 * reason this family or card refuses it (a family deny such as Wan NVFP4's failed accuracy gate, or a failed kernel
 * probe such as rowwise FP8 on Windows), or null when nothing known rules it out. Absent on an older backend.
 */
export type PrecisionOptions = Record<string, string | null | undefined> | null | undefined;

function deniedReason(value: string, denials: PrecisionOptions): string | null {
  const reason = denials?.[value.trim().toLowerCase()];
  return typeof reason === "string" && reason.trim() ? reason.trim() : null;
}

/**
 * Disable each option the backend refuses, with its reason. The backend's reason replaces the NVFP4 switch hint: with
 * the switch off it only reports a measured failure (Wan NVFP4), which turning the switch on would not fix.
 */
export function withPrecisionDenials(options: readonly SelectOption[], denials: PrecisionOptions): SelectOption[] {
  return options.map((option) => {
    const reason = deniedReason(option[0], denials);
    return reason ? [option[0], option[1], reason] : option;
  });
}

/** A held pick the backend now refuses falls back to Auto instead of being sent and refused at load. */
export function deniedSelectionFallback<T extends string>(value: T, denials: PrecisionOptions): T | "auto" {
  return deniedReason(value, denials) ? "auto" : value;
}

/** A download plan's denials, with the GPU selection ("auto" or a card index) the plan was made for. */
export type PlannedPrecision = {
  gpu: string;
  options: PrecisionOptions;
  /** The picked model (and its display id) the plan describes. */
  repos?: readonly (string | null | undefined)[];
} | null;

/**
 * Identity of the resident build, "none" when nothing is loaded. A plan's denials belong to the pick that planned
 * it: once the resident changes (that pick loaded, or the model was ejected or evicted) the page drops them, so an
 * ejected model's family limits never apply to the next pick.
 */
export function precisionResidentKey(
  status:
    | { loaded?: boolean | null; repo_id?: string | null; transformer_quant?: string | null; family?: string | null }
    | null
    | undefined,
): string {
  if (!status?.loaded) return "none";
  return JSON.stringify([status.repo_id ?? null, status.family ?? null, status.transformer_quant ?? null]);
}

/** The GPU selection a request carries: its first explicit card, else "auto". */
export function gpuSelectionKey(gpuIds: readonly number[] | null | undefined): string {
  return gpuIds && gpuIds.length > 0 ? String(gpuIds[0]) : "auto";
}

/**
 * The denials that describe the next load on `targetGpu`. A plan for that selection describes the pick in flight, so
 * it wins unless it is for the resident model itself (then the loaded build's status is the authority). Otherwise the
 * loaded build's apply when that load targets the resident's card (Auto, or the physical card the status judged them
 * on). A kernel-probe failure is per card, and a family refusal is per model: neither may leak to another.
 */
export function activePrecisionDenials(
  status:
    | {
        loaded?: boolean | null;
        repo_id?: string | null;
        display_repo_id?: string | null;
        precision_options?: PrecisionOptions;
        precision_options_card?: number | null;
      }
    | null
    | undefined,
  planned: PlannedPrecision,
  targetGpu = "auto",
): PrecisionOptions {
  const resident = status?.loaded ? [status.repo_id, status.display_repo_id].filter(Boolean) : [];
  const plannedForResident = (planned?.repos ?? []).some((repo) => repo && resident.includes(repo));
  if (planned && planned.gpu === targetGpu && !plannedForResident) return planned.options ?? null;
  const card = status?.precision_options_card;
  const residentTarget = targetGpu === "auto" || (card != null && String(card) === targetGpu);
  if (status?.loaded && status.precision_options && residentTarget) return status.precision_options;
  if (planned && planned.gpu === targetGpu) return planned.options ?? null;
  return null;
}
