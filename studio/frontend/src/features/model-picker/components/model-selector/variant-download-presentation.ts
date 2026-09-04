// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GgufVariantDetail } from "@/features/chat";
import type { ModelSelectorChangeMeta } from "./types";

/** Describe a sole missing drafter as the transfer, not as the cached model. */
export function pendingDrafterPresentation(
  variant: GgufVariantDetail,
): ModelSelectorChangeMeta["downloadPresentation"] {
  const filename = variant.pending_drafter_filename?.trim();
  const expectedBytes = variant.pending_drafter_size_bytes ?? 0;
  if (!filename || !Number.isFinite(expectedBytes) || expectedBytes <= 0) {
    return undefined;
  }
  const basename = filename.replaceAll("\\", "/").split("/").at(-1) ?? filename;
  const lower = basename.toLowerCase();
  const label = lower.startsWith("mtp-")
    ? "MTP companion"
    : lower.startsWith("dspark-") || lower.startsWith("dflash-")
      ? "Draft companion"
      : "Model companion";
  return { label, filename: basename, expectedBytes };
}
