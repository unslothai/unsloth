// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GgufVariantDetail, GgufVariantsResponse } from "@/features/chat";

/** Hub-only state the disk answer cannot know: a newer revision or a companion the Hub added. */
function withHubState<V extends GgufVariantDetail>(
  local: V,
  published: V | undefined,
): V {
  const drafter = local.pending_drafter_filename ? local : published;
  return {
    ...published,
    ...local,
    update_available: published?.update_available ?? local.update_available,
    pending_drafter_filename: drafter?.pending_drafter_filename,
    pending_drafter_size_bytes: drafter?.pending_drafter_size_bytes,
  };
}

export async function loadPickerGgufVariants<T extends GgufVariantsResponse>(
  list: (localOnly: boolean) => Promise<T>,
  options: {
    onDevice: boolean;
    canDiscoverRemote: () => boolean;
    signal?: AbortSignal;
  },
  onCached: (response: T) => void,
): Promise<T> {
  if (!options.onDevice) return list(false);

  const cached = await list(true);
  onCached(cached);
  // Remote discovery still runs with All quantizations off: update badges come only from the Hub.
  if (options.signal?.aborted || !options.canDiscoverRemote()) {
    return cached;
  }

  try {
    const remote = await list(false);
    const variants = new Map(
      remote.variants.map((v) => [v.quant.toLowerCase(), v]),
    );
    for (const local of cached.variants) {
      const key = local.quant.toLowerCase();
      variants.set(key, withHubState(local, variants.get(key)));
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

/** The sole cached quant from the disk answer, withheld when a reachable Hub reports an update or
 *  a missing companion for it (only the expander carries those actions). */
export async function readSoleQuantLocalFirst<
  T extends GgufVariantsResponse,
  S extends { variant: GgufVariantDetail },
>(
  list: (localOnly: boolean) => Promise<T>,
  pick: (response: T) => S | null,
  canDiscoverRemote: () => boolean,
): Promise<S | null> {
  const sole = pick(await list(true));
  if (!sole || !canDiscoverRemote()) return sole;
  let remote: T;
  try {
    remote = await list(false);
  } catch {
    return sole;
  }
  const quant = sole.variant.quant.toLowerCase();
  const published = remote.variants.find(
    (v) => v.quant.toLowerCase() === quant,
  );
  return published?.update_available || published?.pending_drafter_filename
    ? null
    : sole;
}
