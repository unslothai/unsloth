// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type FamilyOverrideMediaKind = "image" | "video";
export type FamilyOverrideArtifactKind =
  | "diffusers_pipeline"
  | "diffusers_modular_pipeline";

/** Whether one scanned root satisfies the exact manifest contract a family loader requires. */
export function artifactKindSupportsFamilyOverride(
  artifactKind: string | null | undefined,
  requiredKind: FamilyOverrideArtifactKind | null | undefined,
): boolean {
  if (!requiredKind) return false;
  return (
    artifactKind === requiredKind || artifactKind === "diffusers_dual_pipeline"
  );
}

/** A family choice may classify only a structurally valid row whose task is unknown. */
export function taskOpaqueArtifactSupportsFamilyOverride(
  task: string | null | undefined,
  artifactKind: string | null | undefined,
  requiredKind: FamilyOverrideArtifactKind | null | undefined,
): boolean {
  return (
    (task == null || task.trim() === "") &&
    artifactKindSupportsFamilyOverride(artifactKind, requiredKind)
  );
}

/** Restore the selector from the canonical family that actually engaged, not a request alias. */
export function resolvedFamilyOverrideSelection(
  control:
    | {
        source?: "auto" | "explicit";
        value?: unknown;
        requested?: unknown;
      }
    | null
    | undefined,
): string | undefined {
  if (control?.source === "auto") return "auto";
  if (typeof control?.value === "string" && control.value.trim()) {
    return control.value;
  }
  // Compatibility with a response that predates the engaged-value field.
  return typeof control?.requested === "string" && control.requested.trim()
    ? control.requested
    : undefined;
}

/** Only an opaque artifact consumes the explicit family that admitted its row. */
export function familyOverrideForPick(
  familyOverride: string | null | undefined,
  required: boolean,
): string | undefined {
  if (!required) {
    return undefined;
  }
  const family = familyOverride?.trim();
  return family && family.toLowerCase() !== "auto" ? family : undefined;
}

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
