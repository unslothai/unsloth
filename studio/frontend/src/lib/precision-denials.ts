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

/** The denials that describe the next load: the loaded build's, else the last planned pick's. */
export function activePrecisionDenials(
  status: { loaded?: boolean | null; precision_options?: PrecisionOptions } | null | undefined,
  planned: PrecisionOptions,
): PrecisionOptions {
  if (status?.loaded && status.precision_options) return status.precision_options;
  return planned ?? null;
}
