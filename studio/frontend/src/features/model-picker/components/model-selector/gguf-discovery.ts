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
    ...(published
      ? {
          size_bytes: published.size_bytes,
          download_size_bytes: published.download_size_bytes,
        }
      : {}),
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

  let cached: T;
  try {
    cached = await list(true);
  } catch (error) {
    if (options.signal?.aborted || !options.canDiscoverRemote()) throw error;
    return list(false);
  }
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
      default_variant: remote.default_variant ?? cached.default_variant,
    };
  } catch {
    // A remote error must not replace the usable disk answer.
    return cached;
  }
}

/** Whether the Hub reports an update or missing companion for a cached quant; only the
  *  expander carries those actions. */
export async function hubWithdrawsSoleQuant<T extends GgufVariantsResponse>(
  listRemote: () => Promise<T>,
  quant: string,
): Promise<boolean> {
  let remote: T;
  try {
    remote = await listRemote();
  } catch {
    return false;
  }
  const published = remote.variants.find(
    (v) => v.quant.toLowerCase() === quant.toLowerCase(),
  );
  return Boolean(
    published?.update_available || published?.pending_drafter_filename,
  );
}

export function createTaskLimiter(limit: number) {
  let active = 0;
  const queue: (() => void)[] = [];
  const next = () => {
    if (active >= limit) return;
    const start = queue.shift();
    if (start) start();
  };
  return function run<R>(task: () => Promise<R>): Promise<R> {
    return new Promise<R>((resolve, reject) => {
      queue.push(() => {
        active += 1;
        task()
          .then(resolve, reject)
          .finally(() => {
            active -= 1;
            next();
          });
      });
      next();
    });
  };
}
