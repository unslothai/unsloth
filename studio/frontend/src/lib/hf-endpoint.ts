// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `config/env.ts` pushes values in from `/api/health`. Imports only zustand so `network.ts`
 * stays testable under bare node. */

import { useStore } from "zustand";
import { createStore } from "zustand/vanilla";

export const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
export const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

export type HubSource = "huggingface" | "modelscope";

type RepoKind = "model" | "dataset";

type ModelScopePages = {
  base: string;
  /** Lowercased Hugging Face id -> ModelScope id; wins over `orgs`. */
  aliases: Record<RepoKind, Record<string, string>>;
  /** Lowercased Hugging Face model org -> ModelScope org. */
  orgs: Record<string, string>;
};

const store = createStore<{
  endpoint: string;
  datasetsServer: string;
  source: HubSource;
  /** Kept after a switch back: requests already queued for the adapter still go there. */
  modelScopeBase: string | null;
  proxyBases: readonly string[];
  /** The web origin behind a relayed `endpoint`; ignored once `endpoint` changes. */
  pageBase: { endpoint: string; base: string } | null;
  modelScopePages: ModelScopePages | null;
}>(() => ({
  endpoint: DEFAULT_HF_ENDPOINT,
  datasetsServer: DEFAULT_DATASETS_SERVER,
  source: "huggingface",
  modelScopeBase: null,
  proxyBases: [],
  pageBase: null,
  modelScopePages: null,
}));

/** Only checks it parses as http(s): `utils/hf_endpoint.py` already sanitised it, and a second
 * policy disagreed about IDN and IPv6 hosts. */
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

/** Blank or unparseable keeps the current value, so a mirror-only deployment is not reset.
 * `proxied` marks the backend's relay. */
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
    pageBase: null,
    modelScopePages: null,
  });
}

function aliasTable(raw: unknown): Record<string, string> {
  // No prototype: an org or id named like `constructor` must not read an inherited property.
  const table: Record<string, string> = Object.create(null);
  if (raw && typeof raw === "object") {
    for (const [from, to] of Object.entries(raw)) {
      if (typeof to === "string") table[from.toLowerCase()] = to;
    }
  }
  return table;
}

/** `/api/health`'s web pages for the current `endpoint`: a relayed mirror's own origin, and
 * `modelscope_pages` (`{base, aliases, orgs}`) while ModelScope is the source. */
export function setHubPageLinks(pageBase?: string | null, modelScope?: unknown): void {
  const ms = modelScope as {
    base?: unknown;
    aliases?: Record<string, unknown>;
    orgs?: unknown;
  } | null;
  const msBase = usableEndpoint(typeof ms?.base === "string" ? ms.base : null);
  store.setState((prev) => ({
    modelScopePages: msBase
      ? {
          base: msBase,
          aliases: {
            model: aliasTable(ms?.aliases?.model),
            dataset: aliasTable(ms?.aliases?.dataset),
          },
          orgs: aliasTable(ms?.orgs),
        }
      : prev.modelScopePages,
  }));
  const base = usableEndpoint(pageBase);
  if (base) setHubPageBase(store.getState().endpoint, base);
}

export function setHubPageBase(endpoint: string, pageBase: string): void {
  const base = usableEndpoint(pageBase);
  if (base && store.getState().endpoint === endpoint) {
    store.setState({ pageBase: { endpoint, base } });
  }
}

/** The relayed endpoint whose web origin is still unknown, or null. */
export function missingHubPageBase(): string | null {
  const { endpoint, source, proxyBases, pageBase } = store.getState();
  return source === "huggingface" &&
    proxyBases.includes(endpoint) &&
    pageBase?.endpoint !== endpoint
    ? endpoint
    : null;
}

const MODELSCOPE_PAGE_PATH =
  /^\/(datasets\/)?([^/?#]+\/[^/?#]+)(?:\/?|\/resolve\/(?:main|master)\/([^?#]+))([?#].*)?$/;

/** Where a browser should open a URL built on `getHfEndpoint()`: the hub's own web page rather
 * than the backend's mount or relay, which would only redirect there. Other URLs pass through. */
export function hubWebUrl(url: string): string {
  const { endpoint, source, pageBase, modelScopePages } = store.getState();
  if (url !== endpoint && !url.startsWith(`${endpoint}/`)) return url;
  const rest = url.slice(endpoint.length);
  if (source === "modelscope") {
    const match = modelScopePages ? MODELSCOPE_PAGE_PATH.exec(rest) : null;
    if (!modelScopePages || !match) return url;
    const kind: RepoKind = match[1] ? "dataset" : "model";
    const [org, name] = match[2].split("/");
    const renamed = kind === "model" ? modelScopePages.orgs[org.toLowerCase()] : undefined;
    const repo =
      modelScopePages.aliases[kind][match[2].toLowerCase()] ??
      (renamed ? `${renamed}/${name}` : match[2]);
    const file = match[3] ? `/resolve/master/${match[3]}` : "";
    return `${modelScopePages.base}/${kind}s/${repo}${file}${match[4] ?? ""}`;
  }
  return pageBase?.endpoint === endpoint ? `${pageBase.base}${rest}` : url;
}

export function useHubWebUrl(url: string): string {
  return useStore(store, () => hubWebUrl(url));
}

export function useMissingHubPageBase(): string | null {
  return useStore(store, missingHubPageBase);
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

/** Hugging Face repos only: a ModelScope id would get another dataset's data. */
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
