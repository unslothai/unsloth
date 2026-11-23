// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** MiniMax-H3 4-step fast mode (pi-H3): a load-time switch, so turning it on or off is a reload. */

export const H3_FAST_STEPS = 4;
export const H3_FAST_GUIDANCE = 1;
// A draft mode: faster, but prompt adherence can drop on some prompts.
export const H3_FAST_LABEL = "4-step fast mode (draft)";
export const H3_FAST_DRAFT_NOTE =
  "A faster draft mode: about 3.6x faster than the 20-step default at similar sharpness, but prompt adherence can drop on some prompts.";

export interface H3FastAvailability {
  available: boolean;
  reason?: string | null;
  repo?: string;
  cached?: boolean;
}

export interface H3FastStatus {
  enabled: boolean;
  steps: number;
  guidance: number;
  sampler?: string;
  sigma_grid?: string;
  adapter_repo?: string;
  lora_layers?: number;
  overwrites?: number;
  levers?: string[];
  restore?: string;
  quality?: string;
  note?: string;
}

type Kind = "gguf" | "single_file" | "pipeline";

/** Sent only when on, for the fl2va partition of a pipeline or single-file pick, so a plain load is unchanged. */
export function h3FastLoadFields(
  enabled: boolean,
  kind: Kind,
  h3Task: string | null | undefined,
): { h3_fast_4step?: true } {
  if (!enabled || kind === "gguf" || h3Task === "ref2va") return {};
  return { h3_fast_4step: true };
}

/** Whether the switch shows (an H3 pick or load), and why it is off when it cannot be used. */
export function h3FastToggleState(args: {
  isH3: boolean;
  kind: Kind;
  h3Task: string | null | undefined;
  availability: H3FastAvailability | null;
}): { visible: boolean; disabled: boolean; reason: string | null } {
  const { isH3, kind, h3Task, availability } = args;
  if (!isH3) return { visible: false, disabled: true, reason: null };
  if (kind === "gguf") {
    return { visible: true, disabled: true, reason: "Not available for GGUF MiniMax-H3 files." };
  }
  if (h3Task === "ref2va") {
    return {
      visible: true,
      disabled: true,
      reason: "Reference-to-video has no pi-H3 adapter; it keeps its own sampler.",
    };
  }
  if (availability == null) return { visible: true, disabled: true, reason: "Checking the adapter..." };
  if (!availability.available) {
    return { visible: true, disabled: true, reason: availability.reason ?? "The pi-H3 adapter is unavailable." };
  }
  return { visible: true, disabled: false, reason: null };
}

/** The loaded build runs the fast mode: steps and guidance are fixed by the backend. */
export function h3FastEngaged(status: { h3_fast?: H3FastStatus | null } | null | undefined): boolean {
  return Boolean(status?.h3_fast?.enabled);
}

/** The switch differs from what is loaded, so the change needs a reload to take effect. */
export function h3FastNeedsReload(
  enabled: boolean,
  status: { loaded?: boolean; family?: string | null; h3_fast?: H3FastStatus | null } | null | undefined,
): boolean {
  if (!status?.loaded || status.family !== "minimax-h3") return false;
  return enabled !== h3FastEngaged(status);
}
