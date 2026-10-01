// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GgufVariantsResponse } from "@/features/chat";

/** Publish cached quants before optional remote discovery, keeping their load paths and readiness. */
export async function loadPickerGgufVariants<T extends GgufVariantsResponse>(
  list: (localOnly: boolean) => Promise<T>,
  options: {
    onDevice: boolean;
    showAllQuantizations: boolean;
    canDiscoverRemote: () => boolean;
    signal?: AbortSignal;
  },
  onCached: (response: T) => void,
): Promise<T> {
  if (!options.onDevice) return list(false);

  const cached = await list(true);
  onCached(cached);
  if (
    !options.showAllQuantizations ||
    options.signal?.aborted ||
    !options.canDiscoverRemote()
  ) {
    return cached;
  }

  try {
    const remote = await list(false);
    const variants = new Map(
      remote.variants.map((v) => [v.quant.toLowerCase(), v]),
    );
    for (const local of cached.variants) {
      const published = variants.get(local.quant.toLowerCase());
      variants.set(local.quant.toLowerCase(), {
        ...published,
        ...local,
        update_available: published?.update_available ?? local.update_available,
      });
    }
    return {
      ...cached,
      variants: [...variants.values()],
      has_vision: cached.has_vision || remote.has_vision,
      default_variant: cached.default_variant ?? remote.default_variant,
    };
  } catch {
    // A remote timeout or connection error must not replace the usable disk answer.
    return cached;
  }
}
