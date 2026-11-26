// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useHfTokenStore } from "@/features/hub";
import { getHfEndpoint } from "@/lib/hf-endpoint";
import { hubFetch } from "@/lib/hub-fetch";
import { listFiles } from "@huggingface/hub";
import { useEffect, useMemo, useState } from "react";

// Summed from files: the Hub's parameter counts read packed MLX weights at full width.
const sizes = new Map<string, number | null>();
const inflight = new Map<string, Promise<number | null>>();
// A failed listing (often a 429) waits this long before it is asked again.
const FAILURE_COOLDOWN_MS = 5 * 60_000;
const failedAt = new Map<string, number>();
const MAX_CONCURRENT = 3;
let active = 0;
const waiting: Array<() => void> = [];

async function withSlot<T>(run: () => Promise<T>): Promise<T> {
  if (active >= MAX_CONCURRENT) {
    await new Promise<void>((resolve) => waiting.push(resolve));
  }
  active++;
  try {
    return await run();
  } finally {
    active--;
    waiting.shift()?.();
  }
}

function keyOf(id: string): string {
  return `${getHfEndpoint()}::${id.toLowerCase()}`;
}

function coolingDown(key: string): boolean {
  const at = failedAt.get(key);
  return at !== undefined && Date.now() - at < FAILURE_COOLDOWN_MS;
}

function fetchSize(id: string): Promise<number | null> {
  const key = keyOf(id);
  const flying = inflight.get(key);
  if (flying) {
    return flying;
  }
  const promise = withSlot(async () => {
    let total = 0;
    for await (const file of listFiles({
      repo: { type: "model", name: id },
      hubUrl: getHfEndpoint(),
      accessToken: useHfTokenStore.getState().token || undefined,
      fetch: hubFetch,
    })) {
      if (file.type === "file" && file.path.endsWith(".safetensors")) {
        total += file.size;
      }
    }
    return total > 0 ? total : null;
  })
    .then((bytes) => {
      sizes.set(key, bytes);
      failedAt.delete(key);
      return bytes;
    })
    .catch(() => {
      failedAt.set(key, Date.now());
      return null;
    })
    .finally(() => inflight.delete(key));
  inflight.set(key, promise);
  return promise;
}

export function useHubRepoSizes(
  ids: readonly string[],
  enabled: boolean,
): ReadonlyMap<string, number> {
  const [version, bump] = useState(0);
  const joined = enabled ? ids.join("\n") : "";
  useEffect(() => {
    if (!joined) {
      return;
    }
    let live = true;
    for (const id of joined.split("\n")) {
      if (!sizes.has(keyOf(id)) && !coolingDown(keyOf(id))) {
        void fetchSize(id).then(() => {
          if (live) {
            bump((n) => n + 1);
          }
        });
      }
    }
    return () => {
      live = false;
    };
  }, [joined]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` marks a listing that arrived.
  return useMemo(() => {
    const out = new Map<string, number>();
    for (const id of joined ? joined.split("\n") : []) {
      const bytes = sizes.get(keyOf(id));
      if (bytes) {
        out.set(id.toLowerCase(), bytes);
      }
    }
    return out;
  }, [joined, version]);
}
