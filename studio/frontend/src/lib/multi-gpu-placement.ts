// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The backend's `multi_gpu` status (diffusion_multigpu.placement_status), the parts the Loaded build row reads. */
export interface MultiGpuStatus {
  decision?: {
    chosen?: string | null;
    summary?: string | null;
    primary?: string | null;
    rejected?: { kind?: string | null; reason?: string | null }[] | null;
  } | null;
  cfg_parallel?: {
    armed?: boolean | null;
    declined?: string | null;
    placement?: { chosen?: string | null; reason?: string | null } | null;
  } | null;
  [key: string]: unknown;
}

const CHOSEN_LABELS: Record<string, string> = {
  companion_split: "Text encoder / VAE on a second card",
  layer_split: "Denoiser split across cards",
  cfg_replica: "Guidance pass on a second card",
  one_card: "One card",
  offload: "One card with offload",
};

/** Component placement and the guidance split match one card exactly; a batch split does not (diffusion_multigpu). */
export const MULTI_GPU_BATCH_NOTE =
  "Placing components on other cards and splitting the guidance pass keep the output identical to one card. Splitting a batch across cards runs different matrix shapes on each card, so a batch split can differ slightly from one card.";

function stripPrefix(text: string): string {
  return text.replace(/^placement:\s*/, "").trim();
}

/**
 * Loaded build row for a multi-card load: what the placement chose and why, with the batch-split note. null when the
 * load had one card to use (no decision recorded) or an older backend sent nothing.
 */
export function multiGpuPlacementRow(
  status: MultiGpuStatus | null | undefined,
): { label: string; detail: string } | null {
  const decision = status?.decision;
  const chosen = (decision?.chosen ?? "").trim();
  if (!decision || !chosen) return null;
  const parts: string[] = [];
  const summary = stripPrefix(decision.summary ?? "");
  if (summary) parts.push(`${summary.charAt(0).toUpperCase()}${summary.slice(1)}.`);
  // The CFG replica verdict (chosen + reason) when the planner weighed one and the summary does not already say it.
  const replica = status?.cfg_parallel?.placement;
  const replicaReason = (replica?.reason ?? "").trim();
  if (replicaReason && !summary.includes(stripPrefix(replicaReason))) {
    parts.push(
      replica?.chosen === "cfg_replica"
        ? `Guidance pass on a second card: ${stripPrefix(replicaReason)}.`
        : `No guidance pass on a second card: ${stripPrefix(replicaReason)}.`,
    );
  }
  const rejected = (decision.rejected ?? [])
    .map((r) => (r?.reason ?? "").trim())
    .filter((r) => r && !parts.some((p) => p.includes(r)));
  if (rejected.length) parts.push(`Not chosen: ${rejected.join("; ")}.`);
  parts.push(MULTI_GPU_BATCH_NOTE);
  return { label: CHOSEN_LABELS[chosen] ?? chosen.replace(/_/g, " "), detail: parts.join(" ") };
}
