// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listGgufVariants } from "@/features/hub";
import { isGgufName, pickGgufFilename } from "./gguf-filename-pick";

/** The .gguf for a pick that arrived with only a repo id (and maybe a quant label). Shares the picker rows' cached listing,
 *  so the row just clicked usually costs no request. Null when the repo is ambiguous or unreadable. */
export async function resolveDiffusionGgufFilename(
  repoId: string,
  options?: {
    quant?: string | null;
    localPath?: string | null;
    hfToken?: string;
  },
): Promise<string | null> {
  const quant = options?.quant?.trim() || null;
  // Already a filename: no listing needed.
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
    // Unreachable Hub or unreadable directory: the caller prompts instead.
    return null;
  }
}
