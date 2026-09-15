// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The frontend's HuggingFace endpoints, held in one place. `config/env.ts` pushes
 * the values in when `/api/health` answers.
 *
 * A store, not plain module state, so React sees it change. Imports only zustand,
 * deliberately: `network.ts` imports this and the unit tests import that under bare
 * node, which cannot evaluate what `config/env.ts` would bring.
 */

import { useStore } from "zustand";
import { createStore } from "zustand/vanilla";

export const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
export const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

const store = createStore<{ endpoint: string; datasetsServer: string }>(() => ({
  endpoint: DEFAULT_HF_ENDPOINT,
  datasetsServer: DEFAULT_DATASETS_SERVER,
}));

/**
 * Accept a value from `/api/health` only if it parses as an http(s) URL. Not a
 * second copy of the backend's policy: `utils/hf_endpoint.py` is the only
 * producer and has already sanitised and canonicalised it. Deciding the rules
 * twice is what made the two disagree about an IDN host and an IPv6 literal.
 */
function usableEndpoint(raw: string | null | undefined): string | null {
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  try {
    const { protocol } = new URL(trimmed);
    if (protocol !== "https:" && protocol !== "http:") return null;
  } catch {
    return null;
  }
  return trimmed.replace(/\/+$/, "");
}

/**
 * Apply the endpoints reported by `/api/health`. A blank, absent or unparseable
 * value leaves the current one alone: older backends report neither field, and
 * resetting would strand a mirror-only deployment on huggingface.co.
 */
export function setHfEndpoints(
  endpoint?: string | null,
  datasetsServer?: string | null,
): void {
  const next = {
    endpoint: usableEndpoint(endpoint),
    datasetsServer: usableEndpoint(datasetsServer),
  };
  store.setState((prev) => ({
    endpoint: next.endpoint ?? prev.endpoint,
    datasetsServer: next.datasetsServer ?? prev.datasetsServer,
  }));
}

export function resetHfEndpoints(): void {
  store.setState({
    endpoint: DEFAULT_HF_ENDPOINT,
    datasetsServer: DEFAULT_DATASETS_SERVER,
  });
}

export function getHfEndpoint(): string {
  return store.getState().endpoint;
}

export function getHfDatasetsServerBase(): string {
  return store.getState().datasetsServer;
}

/** Subscribed, so a mirror arriving after the first render re-runs the caller. */
export function useHfEndpoint(): string {
  return useStore(store, (s) => s.endpoint);
}

export function useHfDatasetsServer(): string {
  return useStore(store, (s) => s.datasetsServer);
}
