// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GgufFitClass } from "../../../../lib/gguf-fit.ts";
import type { VramFitStatus } from "../../../../lib/vram.ts";
import {
  artifactForRepoId,
  curatedArtifactFit,
  type CatalogGroup,
  type DeviceBudget,
} from "./model-catalog.ts";

export type PickerFitStatus = GgufFitClass | VramFitStatus | "offload";

/** Presentation and filters may offer offloadable media without changing Auto's preferred
 * resident artifact. A host pool is already GPU-addressable RAM, never a second allowance. */
export function curatedPickerFit(
  repoId: string,
  catalog: CatalogGroup[],
  budget: DeviceBudget & { hostPooledMemory?: boolean },
) {
  const fit = curatedArtifactFit(repoId, catalog, budget);
  if (!fit) return undefined;
  const group = artifactForRepoId(repoId, catalog)?.group;
  const media = group?.scope === "image" || group?.scope === "video";
  if (!media || fit.sizeGb == null) {
    return { ...fit, status: fit.fits ? null : ("exceeds" as const) };
  }
  const ramGb = budget.hostPooledMemory ? 0 : Math.max(0, budget.systemRamGb);
  const deviceGb = Math.max(0, budget.gpuGb) + ramGb;
  // Reserve working memory using the catalog's existing 70% allowance.
  const allowanceGb = deviceGb * 0.7;
  const fits = (!budget.hostPooledMemory && fit.fits) || fit.sizeGb <= allowanceGb;
  const offloads = fits && ramGb > 0 && fit.sizeGb > budget.gpuGb * 0.7;
  return {
    ...fit,
    fits,
    allowanceGb,
    deviceGb,
    device: ramGb > 0 ? ("GPU + RAM" as const) : ("GPU" as const),
    status: !fits ? ("exceeds" as const) : offloads ? ("offload" as const) : null,
  };
}

export function mediaGgufPickerStatus(fit: GgufFitClass): PickerFitStatus {
  return fit === "partial" ? "offload" : fit === "oom" ? "exceeds" : fit;
}
