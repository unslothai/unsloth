// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/chat";

type FamilyOverrideArtifact = {
  artifact_kind?: LocalModelInfo["artifact_kind"];
  task?: string | null;
};

export type FamilyOverrideMediaKind = "image" | "video";

/** An on-device row that may bypass media-task classification after an explicit family choice.
 *
 * This is intentionally structural: model_format cannot distinguish a complete pipeline from
 * one safetensors component. The backend reports whether the root contains a conventional or
 * Modular Diffusers manifest; each page admits only the contracts its loader accepts.
 */
export function isFamilyOverrideLocalCandidate(
  model: FamilyOverrideArtifact,
  familyOverride: string | null | undefined,
  mediaKind: FamilyOverrideMediaKind | null | undefined,
  modularFamilyOverrides?: readonly string[] | null,
): boolean {
  const family = familyOverride?.trim().toLowerCase();
  const familySupportsModular = Boolean(
    family &&
      modularFamilyOverrides?.some(
        (candidate) => candidate.trim().toLowerCase() === family,
      ),
  );
  const manifestIsLoadable =
    model.artifact_kind === "diffusers_pipeline" ||
    (mediaKind === "video" &&
      model.artifact_kind === "diffusers_modular_pipeline" &&
      familySupportsModular);
  return (
    Boolean(family && family !== "auto") &&
    model.task == null &&
    manifestIsLoadable
  );
}

/** Opaque pipeline roots stay out of chat and task pickers until the explicit contract applies. */
export function localArtifactPassesOverrideGate(
  model: FamilyOverrideArtifact,
  familyOverride: string | null | undefined,
  mediaKind: FamilyOverrideMediaKind | null | undefined,
  modularFamilyOverrides?: readonly string[] | null,
): boolean {
  const isPipelineRoot =
    model.artifact_kind === "diffusers_pipeline" ||
    model.artifact_kind === "diffusers_modular_pipeline";
  return (
    model.task != null ||
    !isPipelineRoot ||
    isFamilyOverrideLocalCandidate(
      model,
      familyOverride,
      mediaKind,
      modularFamilyOverrides,
    )
  );
}
