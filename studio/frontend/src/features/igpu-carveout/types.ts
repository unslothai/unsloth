// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Advice attached to a model load when the integrated GPU has less memory dedicated
 *  to it than the weights need, so they run from shared system memory. Sent at most
 *  once per allocation, and never on hardware where the setting does not exist. */
export interface IgpuCarveoutAdvice {
  /** GB currently dedicated to the integrated GPU. */
  current_gb: number;
  /** GB of weights this model wants resident. */
  needed_gb: number;
  /** GB to suggest: the smallest plausible setting that holds the weights. */
  suggested_gb: number;
  /** GB the machine has in total (visible RAM plus the current allocation). */
  machine_gb: number;
  /** GB left for the rest of the system if the suggestion is taken. */
  host_left_gb: number;
  /** Prose written by the backend, which owns the wording. */
  message: string;
}

/** Narrow an unknown load-response field. The notice quotes numbers, so a partial
 *  payload from an older or proxied backend is treated as no advice at all rather
 *  than rendered as "undefined GB". */
export function parseCarveoutAdvice(value: unknown): IgpuCarveoutAdvice | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const num = (key: string): number | null => {
    const n = raw[key];
    return typeof n === "number" && Number.isFinite(n) ? n : null;
  };
  const current_gb = num("current_gb");
  const needed_gb = num("needed_gb");
  const suggested_gb = num("suggested_gb");
  const machine_gb = num("machine_gb");
  const host_left_gb = num("host_left_gb");
  const message = raw.message;
  if (
    current_gb === null ||
    needed_gb === null ||
    suggested_gb === null ||
    machine_gb === null ||
    host_left_gb === null ||
    typeof message !== "string" ||
    !message.trim()
  ) {
    return null;
  }
  return { current_gb, needed_gb, suggested_gb, machine_gb, host_left_gb, message };
}
