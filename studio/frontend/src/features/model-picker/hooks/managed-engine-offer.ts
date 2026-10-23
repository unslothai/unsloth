// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type ManagedEngine = "vllm" | "sglang";

/** /validate's managed_engine_offer: the Default engine cannot run this quantization (#11728). */
export interface ManagedEngineOffer {
  quantization: string;
  engines: ManagedEngine[];
}

type Resolver = (engine: ManagedEngine | null) => void;

// One open offer; a newer request declines the older one.
let pendingResolver: Resolver | null = null;

interface ManagedEngineOfferStore {
  open: boolean;
  modelName: string | null;
  offer: ManagedEngineOffer | null;
  request: (
    modelName: string,
    offer: ManagedEngineOffer,
    signal?: AbortSignal,
  ) => Promise<ManagedEngine | null>;
  resolve: (engine: ManagedEngine | null) => void;
}

export const useManagedEngineOfferStore = create<ManagedEngineOfferStore>()(
  (set) => ({
    open: false,
    modelName: null,
    offer: null,
    request: (modelName, offer, signal) =>
      new Promise<ManagedEngine | null>((resolve) => {
        pendingResolver?.(null);
        if (signal?.aborted) {
          pendingResolver = null;
          set({ open: false });
          resolve(null);
          return;
        }
        const settle: Resolver = (engine) => {
          signal?.removeEventListener("abort", onAbort);
          resolve(engine);
        };
        const onAbort = () => {
          if (pendingResolver !== settle) return;
          pendingResolver = null;
          set({ open: false });
          settle(null);
        };
        signal?.addEventListener("abort", onAbort, { once: true });
        pendingResolver = settle;
        set({ open: true, modelName, offer });
      }),
    resolve: (engine) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({ open: false });
      resolver?.(engine);
    },
  }),
);

/** The engines of an offer this build can load with, in the backend's order. */
export function offeredEngines(
  offer: ManagedEngineOffer | null | undefined,
): ManagedEngine[] {
  return (offer?.engines ?? []).filter(
    (engine): engine is ManagedEngine =>
      engine === "vllm" || engine === "sglang",
  );
}

/** Pause a Default-engine load on the offer dialog; resolves the engine to load with, or null. */
export async function confirmManagedEngineIfNeeded(
  modelName: string,
  offer: ManagedEngineOffer | null | undefined,
  signal?: AbortSignal,
): Promise<ManagedEngine | null> {
  const engines = offeredEngines(offer);
  if (!offer || engines.length === 0) return null;
  return useManagedEngineOfferStore
    .getState()
    .request(modelName, { ...offer, engines }, signal);
}
