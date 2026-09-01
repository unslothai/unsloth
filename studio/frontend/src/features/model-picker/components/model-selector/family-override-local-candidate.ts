// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/chat";

type FamilyOverrideArtifact = {
  artifact_kind?: LocalModelInfo["artifact_kind"];
  task?: string | null;
};

/** An on-device row that may bypass media-task classification after an explicit family choice.
 *
 * This is intentionally structural: model_format cannot distinguish a complete pipeline from
 * one safetensors component. The backend only assigns diffusers_pipeline after finding the
 * pipeline index at the directory root.
 */
export function isFamilyOverrideLocalCandidate(
  model: FamilyOverrideArtifact,
  familyOverride: string | null | undefined,
): boolean {
  return (
    Boolean(familyOverride && familyOverride !== "auto") &&
    model.task == null &&
    model.artifact_kind === "diffusers_pipeline"
  );
}

/** Opaque pipeline roots stay out of chat and task pickers until the explicit contract applies. */
export function localArtifactPassesOverrideGate(
  model: FamilyOverrideArtifact,
  familyOverride: string | null | undefined,
): boolean {
  return (
    model.task != null ||
    model.artifact_kind !== "diffusers_pipeline" ||
    isFamilyOverrideLocalCandidate(model, familyOverride)
  );
}
