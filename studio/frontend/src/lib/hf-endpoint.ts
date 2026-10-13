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

export type HubSource = "huggingface" | "modelscope";

const store = createStore<{
  endpoint: string;
  datasetsServer: string;
  source: HubSource;
  /** Kept after a switch back: requests already queued for the adapter still go there. */
  modelScopeBase: string | null;
  proxyBases: readonly string[];
}>(() => ({
  endpoint: DEFAULT_HF_ENDPOINT,
  datasetsServer: DEFAULT_DATASETS_SERVER,
  source: "huggingface",
  modelScopeBase: null,
  proxyBases: [],
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
 * resetting would strand a mirror-only deployment on huggingface.co. `proxied`
 * marks a value that is the backend's relay rather than the endpoint itself.
 */
export function setHfEndpoints(
  endpoint?: string | null,
  datasetsServer?: string | null,
  source?: string | null,
  proxied: { endpoint?: boolean; datasetsServer?: boolean } = {},
): void {
  const next = {
    endpoint: usableEndpoint(endpoint),
    datasetsServer: usableEndpoint(datasetsServer),
  };
  store.setState((prev) => {
    const nextSource =
      source === "huggingface" || source === "modelscope" ? source : prev.source;
    const applied = next.endpoint ? nextSource : prev.source;
    const relays = [
      proxied.endpoint ? next.endpoint : null,
      proxied.datasetsServer ? next.datasetsServer : null,
    ].filter((base): base is string => base !== null && !prev.proxyBases.includes(base));
    return {
      endpoint: next.endpoint ?? prev.endpoint,
      datasetsServer: next.datasetsServer ?? prev.datasetsServer,
      source: applied,
      modelScopeBase:
        next.endpoint && applied === "modelscope" ? next.endpoint : prev.modelScopeBase,
      proxyBases: relays.length ? [...prev.proxyBases, ...relays] : prev.proxyBases,
    };
  });
}

export function resetHfEndpoints(): void {
  store.setState({
    endpoint: DEFAULT_HF_ENDPOINT,
    datasetsServer: DEFAULT_DATASETS_SERVER,
    source: "huggingface",
    modelScopeBase: null,
    proxyBases: [],
  });
}

export function getHfEndpoint(): string {
  return store.getState().endpoint;
}

export function getHubSource(): HubSource {
  return store.getState().source;
}

export function isModelScopeHubUrl(url: string): boolean {
  const base = store.getState().modelScopeBase;
  return base !== null && url.startsWith(`${base}/`);
}

export function isProxiedHubUrl(url: string): boolean {
  return store.getState().proxyBases.some((base) => url.startsWith(`${base}/`));
}

let sessionRefresh: (() => Promise<boolean>) | null = null;

export function setHubSessionRefresh(refresh: () => Promise<boolean>): void {
  sessionRefresh = refresh;
}

export function refreshHubSession(): Promise<boolean> {
  return sessionRefresh ? sessionRefresh() : Promise.resolve(false);
}

export function useHubSource(): HubSource {
  return useStore(store, (s) => s.source);
}

export function useHubName(): string {
  return useHubSource() === "modelscope" ? "ModelScope" : "Hugging Face";
}

/** The datasets-server knows Hugging Face repos only: a ModelScope id would get another dataset's data. */
export function hasDatasetsServer(): boolean {
  return store.getState().source !== "modelscope";
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
