// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";

export type FamilyOverrideArtifactKind = "diffusers_pipeline" | "diffusers_modular_pipeline";

const LABELS: Readonly<Record<string, string>> = {
  "flux.1": "FLUX.1",
  "flux.2-klein": "FLUX.2 Klein",
  "flux.2-dev": "FLUX.2 Dev",
  "flux.1-kontext": "FLUX.1 Kontext",
  "qwen-image": "Qwen Image",
  "qwen-image-2.1": "Qwen Image 2.1",
  "qwen-image-edit": "Qwen Image Edit",
  "z-image": "Z-Image",
  "krea-2": "Krea 2",
  "lumina-2": "Lumina 2",
  "hunyuanimage-2.1": "HunyuanImage 2.1",
  "hidream-i1": "HiDream-I1",
  "ideogram-4": "Ideogram 4",
  sdxl: "SDXL",
  "minimax-h3": "MiniMax-H3",
  "ltx-2": "LTX-2",
  "wan2.2-ti2v-5b": "Wan2.2 TI2V 5B",
  "wan2.2-t2v-a14b": "Wan2.2 T2V A14B",
  "hunyuanvideo-1.5": "HunyuanVideo 1.5 (480p)",
  "hunyuanvideo-1.5-720p": "HunyuanVideo 1.5 (720p)",
};

export const FAMILY_OVERRIDE_HINT =
  "Architecture family. Auto detects it from the repository or pipeline metadata. Choose one only for a custom Diffusers pipeline whose metadata does not identify a supported family.";

/** Options come from the backend registry, so a new family cannot leave the UI stale. */
export function familyOverrideOptions(supported: readonly string[] | null | undefined): [string, string][] {
  const names = [...new Set(supported ?? [])].filter(Boolean);
  return [["auto", "Auto (detect)"], ...names.map((n): [string, string] => [n, LABELS[n] ?? n])];
}

/** The trimmed family when one is explicitly chosen; undefined for blank or Auto. */
export function explicitFamily(value: unknown): string | undefined {
  const family = typeof value === "string" ? value.trim() : "";
  return family && family.toLowerCase() !== "auto" ? family : undefined;
}

/** A family choice admits only a structurally valid row whose task is unknown; dual roots satisfy either loader. */
export function taskOpaqueArtifactSupportsFamilyOverride(
  task: string | null | undefined,
  artifactKind: string | null | undefined,
  requiredKind: FamilyOverrideArtifactKind | null | undefined,
): boolean {
  return (
    !task?.trim() &&
    Boolean(requiredKind) &&
    (artifactKind === requiredKind || artifactKind === "diffusers_dual_pipeline")
  );
}

/** Restore the selector from the canonical family that engaged, not a request alias. */
export function resolvedFamilyOverrideSelection(
  control: { source?: "auto" | "explicit"; value?: unknown; requested?: unknown } | null | undefined,
): string | undefined {
  if (control?.source === "auto") return "auto";
  for (const v of [control?.value, control?.requested]) {
    if (typeof v === "string" && v.trim()) return v;
  }
  return undefined;
}

export function familyOverrideArtifactKind(
  familyOverride: string | null | undefined,
  modularFamilies?: readonly string[] | null,
): FamilyOverrideArtifactKind | undefined {
  const family = explicitFamily(familyOverride)?.toLowerCase();
  if (!family) return undefined;
  return modularFamilies?.some((f) => f.trim().toLowerCase() === family)
    ? "diffusers_modular_pipeline"
    : "diffusers_pipeline";
}

/** Family selection plus what the selector shows: a pinned snapshot is labelled by its logical id. */
export function useFamilyOverride(
  status: { loaded: boolean; repo_id: string | null; display_repo_id?: string | null; modular_families?: string[] } | null,
) {
  const [familyOverride, setFamilyOverride] = useState("auto");
  return {
    familyOverride,
    setFamilyOverride,
    opaqueKind: familyOverrideArtifactKind(familyOverride, status?.modular_families),
    selectorModelId: status?.loaded && status.repo_id ? (status.display_repo_id ?? status.repo_id) : undefined,
  };
}
