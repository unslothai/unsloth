// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type FamilyOverrideMediaKind = "image" | "video";
export type FamilyOverrideArtifactKind =
  | "diffusers_pipeline"
  | "diffusers_modular_pipeline";

export function familyOverrideArtifactKind(
  familyOverride: string | null | undefined,
  mediaKind: FamilyOverrideMediaKind | null | undefined,
  modularFamilyOverrides?: readonly string[] | null,
): FamilyOverrideArtifactKind | undefined {
  const family = familyOverride?.trim().toLowerCase();
  if (!family || family === "auto") return undefined;
  const familySupportsModular = modularFamilyOverrides?.some(
    (candidate) => candidate.trim().toLowerCase() === family,
  );
  return mediaKind === "video" && familySupportsModular
    ? "diffusers_modular_pipeline"
    : "diffusers_pipeline";
}
