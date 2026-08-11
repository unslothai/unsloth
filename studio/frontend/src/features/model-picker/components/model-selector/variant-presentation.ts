// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type PresentableGgufVariant = {
  filename: string;
  quant: string;
  display_label?: string | null;
  size_bytes: number;
};

export type GgufVariantPresentationGroup<T extends PresentableGgufVariant> = {
  key: string;
  title: string | null;
  description: string | null;
  variants: T[];
};

const H3_FILENAME = /^minimax_h3_(fl2va|ref2va)(?:_pruned)?-(.+)\.gguf$/i;
const GGUF_SHARD_SUFFIX = /-\d{5}-of-\d{5}$/i;
const PATH_SEPARATOR = /[\\/]/;

type H3Presentation = {
  group: "text-frames" | "reference-media";
  quantLabel: string;
  build: "Full" | "Pruned";
};

function h3Presentation(
  variant: PresentableGgufVariant,
): H3Presentation | null {
  const filename =
    variant.filename.split(PATH_SEPARATOR).at(-1) ?? variant.filename;
  const match = H3_FILENAME.exec(filename);
  if (!match) {
    return null;
  }
  return {
    group:
      match[1].toLowerCase() === "ref2va" ? "reference-media" : "text-frames",
    quantLabel: match[2].replace(GGUF_SHARD_SUFFIX, ""),
    build: filename.toLowerCase().includes("_pruned-") ? "Pruned" : "Full",
  };
}

export function ggufVariantPickerLabel(
  variant: PresentableGgufVariant,
  options?: { h3Grouped?: boolean; hideH3PrunedBuild?: boolean },
): string {
  const h3 = options?.h3Grouped ? h3Presentation(variant) : null;
  if (h3) {
    return options?.hideH3PrunedBuild && h3.build === "Pruned"
      ? h3.quantLabel
      : `${h3.quantLabel} · ${h3.build}`;
  }
  return variant.display_label?.trim() || variant.quant;
}

export function h3PickerHasOnlyPrunedBuilds(
  variants: readonly PresentableGgufVariant[],
): boolean {
  return (
    variants.length > 0 &&
    variants.every((variant) => h3Presentation(variant)?.build === "Pruned")
  );
}

export function groupGgufVariantsForPicker<T extends PresentableGgufVariant>(
  variants: readonly T[],
): GgufVariantPresentationGroup<T>[] {
  const classified = variants.map((variant) => ({
    variant,
    presentation: h3Presentation(variant),
  }));
  if (
    classified.length === 0 ||
    classified.some(({ presentation }) => presentation === null)
  ) {
    return [
      {
        key: "quantizations",
        title: null,
        description: null,
        variants: [...variants],
      },
    ];
  }

  const textFrames = classified
    .filter(({ presentation }) => presentation?.group === "text-frames")
    .map(({ variant }) => variant);
  const referenceMedia = classified
    .filter(({ presentation }) => presentation?.group === "reference-media")
    .map(({ variant }) => variant);

  return [
    {
      key: "text-frames",
      title: "Text / first and last frames",
      description:
        "Generate from a prompt, optionally using first and last frame images.",
      variants: textFrames,
    },
    {
      key: "reference-media",
      title: "Reference media",
      description: "Generate using reference images, video, or audio.",
      variants: referenceMedia,
    },
  ].filter((group) => group.variants.length > 0);
}

export function preferredGgufVariantByGroup<T extends PresentableGgufVariant>(
  groups: readonly GgufVariantPresentationGroup<T>[],
  defaultVariant: string | null,
): Map<string, T | null> {
  const allVariants = groups.flatMap((group) => group.variants);
  const defaultDetail = allVariants.find(
    (variant) => variant.quant === defaultVariant,
  );
  const defaultPresentation = defaultDetail
    ? h3Presentation(defaultDetail)
    : null;

  return new Map(
    groups.map((group) => {
      const exactDefault = group.variants.find(
        (variant) => variant.quant === defaultVariant,
      );
      if (exactDefault || !defaultDetail) {
        return [group.key, exactDefault ?? null];
      }
      const semanticCounterpart = defaultPresentation
        ? group.variants.find((variant) => {
            const presentation = h3Presentation(variant);
            return (
              presentation?.quantLabel.toLowerCase() ===
                defaultPresentation.quantLabel.toLowerCase() &&
              presentation.build === defaultPresentation.build
            );
          })
        : undefined;
      if (semanticCounterpart) {
        return [group.key, semanticCounterpart];
      }
      const nearestSize = [...group.variants].sort(
        (left, right) =>
          Math.abs(left.size_bytes - defaultDetail.size_bytes) -
          Math.abs(right.size_bytes - defaultDetail.size_bytes),
      )[0];
      return [group.key, nearestSize ?? null];
    }),
  );
}
