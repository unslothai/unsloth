// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CSSProperties } from "react";

import type {
  DiffusionConditioning,
  DiffusionGenerateRequest,
  LocalizedEditMode,
} from "./api";
import { type SizeLimits, fitSize, matchSourceSize } from "./image-size.ts";

export const ANNOTATION_COLORS: ReadonlyArray<{ name: string; value: string }> =
  [
    { name: "red", value: "#ff2020" },
    { name: "blue", value: "#1f6bff" },
    { name: "green", value: "#1fc23a" },
    { name: "yellow", value: "#ffd400" },
  ];

export const TRANSPARENCY_CHECKER: CSSProperties = {
  backgroundImage:
    "repeating-conic-gradient(rgb(128 128 128 / 0.28) 0% 25%, transparent 0% 50%)",
  backgroundSize: "16px 16px",
};

/** Total inputs when the backend does not say: the FLUX.2 reference limit it has always had. */
export const DEFAULT_MAX_CONDITION_IMAGES = 4;

export function maxAdditionalImages(
  conditioning: DiffusionConditioning | null | undefined,
  localizedMode: LocalizedEditMode | null,
): number {
  const total =
    conditioning?.max_condition_images ?? DEFAULT_MAX_CONDITION_IMAGES;
  return Math.max(0, total - 1 - (localizedMode === "mask" ? 1 : 0));
}

export function additionalImageNumber(
  index: number,
  localizedMode: LocalizedEditMode | null,
): number {
  return index + (localizedMode === "mask" ? 3 : 2);
}

export const REFERENCE_DETAIL_LABELS: Record<number, string> = {
  512: "Low (512)",
  1024: "Standard (1024)",
  2048: "High (2048)",
};

export function seedReferenceResolution(
  allowed: readonly number[],
  tier: number,
): number | null {
  if (!allowed.length) return null;
  if (allowed.includes(tier)) return tier;
  if (allowed.includes(1024)) return 1024;
  return allowed[0];
}

export const OFFICIAL_2K_PRESETS: ReadonlyArray<{
  label: string;
  width: number;
  height: number;
}> = [
  { label: "1:1", width: 2048, height: 2048 },
  { label: "4:3", width: 2400, height: 1792 },
  { label: "3:4", width: 1792, height: 2400 },
  { label: "3:2", width: 2528, height: 1696 },
  { label: "2:3", width: 1696, height: 2528 },
  { label: "16:9", width: 2752, height: 1536 },
  { label: "9:16", width: 1536, height: 2752 },
];

export function presetsWithin(limits: SizeLimits) {
  return OFFICIAL_2K_PRESETS.filter(
    (p) =>
      Math.max(p.width, p.height) <= limits.maxSide &&
      p.width * p.height <= limits.maxPixels &&
      p.width % limits.multiple === 0 &&
      p.height % limits.multiple === 0,
  );
}

export const MINIMAX_H3_IMAGE_FAMILY = "minimax-h3-image";

/** MiniMax-H3-Image's own sizes (its model card): the 4 MP canvas per aspect, then the 16 MP table. It renders
 *  on at least 4 MP and resizes back, so a smaller request costs the 4 MP time. Mirrors the backend's
 *  H3_IMAGE_PRESETS. */
export const MINIMAX_H3_IMAGE_PRESETS: ReadonlyArray<{
  label: string;
  width: number;
  height: number;
}> = [
  { label: "1:1 4 MP", width: 2048, height: 2048 },
  { label: "4:3 4 MP", width: 2368, height: 1760 },
  { label: "3:4 4 MP", width: 1760, height: 2368 },
  { label: "3:2 4 MP", width: 2496, height: 1664 },
  { label: "2:3 4 MP", width: 1664, height: 2496 },
  { label: "16:9 4 MP", width: 2720, height: 1536 },
  { label: "9:16 4 MP", width: 1536, height: 2720 },
  { label: "1:1 16 MP", width: 4096, height: 4096 },
  { label: "4:3 16 MP", width: 4736, height: 3552 },
  { label: "3:4 16 MP", width: 3552, height: 4736 },
  { label: "3:2 16 MP", width: 5024, height: 3360 },
  { label: "2:3 16 MP", width: 3360, height: 5024 },
  { label: "16:9 16 MP", width: 5472, height: 3072 },
  { label: "9:16 16 MP", width: 3072, height: 5472 },
  { label: "7:3 16 MP", width: 6272, height: 2688 },
];

function isMinimaxH3Image(family: string | null | undefined): boolean {
  return (family ?? "").trim().toLowerCase() === MINIMAX_H3_IMAGE_FAMILY;
}

export function familyPresetsWithin(
  family: string | null | undefined,
  limits: SizeLimits,
) {
  if (!isMinimaxH3Image(family)) return presetsWithin(limits);
  return MINIMAX_H3_IMAGE_PRESETS.filter(
    (p) =>
      Math.max(p.width, p.height) <= limits.maxSide &&
      p.width * p.height <= limits.maxPixels &&
      p.width % limits.multiple === 0 &&
      p.height % limits.multiple === 0,
  );
}

export function familyPresetsCopy(family: string | null | undefined): {
  label: string;
  hint: string;
} {
  if (isMinimaxH3Image(family)) {
    return {
      label: "Model sizes",
      hint: "MiniMax-H3-Image's native sizes. It always renders at least 4 MP and resizes back to the size you ask for; the 16 MP sizes take about four times the memory and time.",
    };
  }
  return {
    label: "2K presets",
    hint: "The model's native 2K sizes. They take several times the memory and time of a 1 megapixel image.",
  };
}

/** What the inputs are called: MiniMax-H3-Image prompts refer to them as <Picture N>. */
export function inputImageWord(
  family: string | null | undefined,
): "Picture" | "Image" {
  return isMinimaxH3Image(family) ? "Picture" : "Image";
}

export function inputImageReference(
  family: string | null | undefined,
  n: number,
): string {
  return isMinimaxH3Image(family) ? `<Picture ${n}>` : `Image ${n}`;
}

export const TRANSPARENCY_PREFIX = "This is an RGBA image with transparency.";
export const TRANSPARENCY_SUFFIX =
  "The image has alpha channel and the background is transparent.";

export function withTransparencyPrompt(prompt: string): string {
  let body = prompt.trim();
  if (!body.startsWith(TRANSPARENCY_PREFIX)) {
    body = `${TRANSPARENCY_PREFIX} ${body}`.trim();
  }
  if (!body.endsWith(TRANSPARENCY_SUFFIX)) {
    body = `${body} ${TRANSPARENCY_SUFFIX}`.trim();
  }
  return body;
}

/** Region wording from the official demo cases: a leading phrase naming the marked area, or for
 *  annotations a closing sentence asking for the marks to be left out. */
export function withLocalizedHint(
  prompt: string,
  mode: LocalizedEditMode,
  colors: readonly string[] = [],
): string {
  const body = prompt.trim();
  if (mode === "annotate") {
    const named = colors.length ? joinWords(colors) : "annotation";
    const suffix = `Do not render the ${named} annotation lines in the image.`;
    return body.endsWith(suffix) ? body : `${body} ${suffix}`.trim();
  }
  const prefix =
    mode === "paint"
      ? "In the area marked with white paint,"
      : "In the area marked by the mask,";
  return body.startsWith(prefix) ? body : `${prefix} ${body}`.trim();
}

function joinWords(words: readonly string[]): string {
  if (words.length < 2) return words[0] ?? "";
  return `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

export type EditSizing = "source" | "custom";

export function resolveEditSize(
  sizing: EditSizing,
  source: { width: number; height: number } | null,
  matchResolution: number,
  custom: { width: number; height: number },
  limits: SizeLimits,
): { width: number; height: number } {
  if (sizing === "source" && source && source.width > 0 && source.height > 0) {
    return matchSourceSize(
      source.width,
      source.height,
      matchResolution,
      limits,
    );
  }
  return fitSize(custom.width, custom.height, limits);
}

/** The conditioned half of a Reference / unified or multi-image Edit request. Empty slots are dropped in order. */
export function conditionedRequestFields(opts: {
  workflow: "edit" | "reference";
  initImage: string;
  extras: readonly string[];
  referenceResolution: number | null;
  conditioning: DiffusionConditioning | null | undefined;
  localized: { mode: LocalizedEditMode; image: string } | null;
}): Pick<
  DiffusionGenerateRequest,
  | "workflow"
  | "init_image"
  | "reference_images"
  | "reference_resolution"
  | "localized_edit"
> {
  const allowed = opts.conditioning?.reference_resolutions ?? [];
  const extras = opts.extras.filter(Boolean);
  const unified = Boolean(opts.conditioning?.unified_edit);
  const multiEdit = Boolean(opts.conditioning?.multi_image_edit);
  const localized =
    opts.workflow === "edit" && unified && opts.localized
      ? opts.localized
      : undefined;
  return {
    workflow: opts.workflow,
    init_image: opts.initImage,
    reference_images:
      extras.length && (opts.workflow === "reference" || unified || multiEdit)
        ? extras
        : undefined,
    reference_resolution:
      opts.referenceResolution != null &&
      allowed.includes(opts.referenceResolution)
        ? opts.referenceResolution
        : undefined,
    localized_edit: localized,
  };
}

export function restoreInputsNote(image: {
  workflow?: string | null;
  reference_image_count?: number | null;
  localized_edit?: LocalizedEditMode | null;
}): string | null {
  const extras = image.reference_image_count ?? 0;
  const layer =
    image.localized_edit === "mask"
      ? "the mask"
      : image.localized_edit === "annotate"
        ? "the annotations"
        : image.localized_edit === "paint"
          ? "the painted region"
          : null;
  if (image.workflow === "edit" || image.workflow === "reference") {
    const parts = ["the source image"];
    if (layer) parts.push(layer);
    if (extras > 0)
      parts.push(
        `${extras} additional image${extras === 1 ? "" : "s"} in the same order`,
      );
    return parts.length > 1
      ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`
      : parts[0];
  }
  return null;
}
