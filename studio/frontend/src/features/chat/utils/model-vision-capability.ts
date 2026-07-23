// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelVisionCapability = {
  isVision?: boolean;
  isGguf?: boolean;
};

/**
 * Whether a selected model is authoritatively known to be text-only.
 *
 * A GGUF repo's catalog row can predate the variant metadata request and report
 * `isVision: false` even though the selected quant has an mmproj companion.
 * Prefer the variant-level hint when present; otherwise treat GGUF capability
 * as unknown rather than warning from stale catalog metadata.
 */
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
