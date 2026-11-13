// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelVisionCapability = {
  isVision?: boolean;
  isGguf?: boolean;
};

/** GGUF catalog rows can report `isVision: false` before variant metadata; prefer the variant hint. */
export function isKnownTextOnlySelection(
  selection: ModelVisionCapability,
  catalogModel?: ModelVisionCapability,
): boolean {
  if (selection.isVision !== undefined) {
    return selection.isVision === false;
  }
  if (selection.isGguf === true || catalogModel?.isGguf === true) {
    return false;
  }
  return catalogModel?.isVision === false;
}


export function normalizeGgufVisionCapability(
  value: unknown,
): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}
