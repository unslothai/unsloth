// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useHfTokenStore } from "@/features/hub";
import { getHfEndpoint } from "@/lib/hf-endpoint";
import { hubFetch } from "@/lib/hub-fetch";
import { listFiles } from "@huggingface/hub";
import { useEffect, useMemo, useState } from "react";

// Summed from the repo's files: the Hub's safetensors parameter counts read packed MLX weights
// as full-width (a 26 GB 4-bit repo comes out at 97 GB).
const sizes = new Map<string, number | null>();
const inflight = new Map<string, Promise<number | null>>();
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
      return bytes;
    })
    // Not cached: a dropped connection or missing token retries on the next list change.
    .catch(() => null)
    .finally(() => inflight.delete(key));
  inflight.set(key, promise);
  return promise;
}

/** Download size of each repo in `ids`, filled in as listings arrive. */
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
      if (!sizes.has(keyOf(id))) {
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
