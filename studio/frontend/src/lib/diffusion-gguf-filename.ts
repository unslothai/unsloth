// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listGgufVariants } from "@/features/hub";
import { isGgufName, pickGgufFilename } from "./gguf-filename-pick";

/** Shares the picker's cached listing. Null when ambiguous or unreadable. */
export async function resolveDiffusionGgufFilename(
  repoId: string,
  options?: {
    quant?: string | null;
    localPath?: string | null;
    hfToken?: string;
  },
): Promise<string | null> {
  const quant = options?.quant?.trim() || null;
  if (quant && isGgufName(quant)) return quant;
  try {
    const res = await listGgufVariants(repoId, options?.hfToken, {
      preferLocalCache: true,
      localPath: options?.localPath ?? null,
    });
    return pickGgufFilename(
      Array.isArray(res?.variants) ? res.variants : [],
      quant,
    );
  } catch {
    // The caller prompts instead.
    return null;
  }
}
