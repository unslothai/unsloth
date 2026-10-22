// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type {
  FlowDocument,
  FlowKind,
  MediaValue,
  RunStatus,
  StoryManifest,
  WorldSpec,
} from "../types";

export interface SessionRecord {
  slot: string;
  state: unknown;
  label?: string | null;
  updatedAt?: number;
}

export interface InvokeResult {
  outputs: Record<string, unknown>;
  session_id?: string;
  rev?: number;
}

/** Everything the players need from the backend; the smoke harness swaps in a mock. */
export interface PlayerApi {
  flowInfo(
    flowId: string,
  ): Promise<{ id: string; name: string; kind: FlowKind; doc?: FlowDocument }>;
  story(flowId: string): Promise<StoryManifest>;
  world(flowId: string): Promise<WorldSpec>;
  invoke(
    flowId: string,
    entry: string,
    inputs: Record<string, unknown>,
    opts?: { sessionId?: string; rev?: number; signal?: AbortSignal },
  ): Promise<InvokeResult>;
  listSessions(flowId: string): Promise<SessionRecord[]>;
  loadSession(flowId: string, slot: string): Promise<SessionRecord | null>;
  saveSession(
    flowId: string,
    slot: string,
    state: unknown,
    label?: string,
  ): Promise<void>;
  /** Re-run the flow with a new seed (Story/World Generator re-roll) and wait for it to finish. */
  reroll(
    flowId: string,
    seed: number,
    onStatus?: (status: RunStatus) => void,
    signal?: AbortSignal,
  ): Promise<void>;
  mediaBlob(media: MediaValue): Promise<Blob>;
}

export class PlayerApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = formatFastApiDetail(body?.detail) || detail;
    } catch {
      // keep statusText
    }
    throw new PlayerApiError(detail || `HTTP ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export function mediaPath(media: MediaValue): string {
  if (media.url) return media.url;
  const ref = media.ref;
  const colon = ref.indexOf(":");
  const kind = ref.slice(0, colon);
  const id = ref.slice(colon + 1);
  if (kind === "image") return `/api/inference/images/gallery/${enc(id)}/file`;
  if (kind === "video") return `/api/inference/video/gallery/${enc(id)}/file`;
  if (kind === "audio") return `/api/inference/audio/gallery/${enc(id)}/file`;
  return `/api/flows/assets/${enc(ref)}`;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("aborted", "AbortError"));
    });
  });

export const playerApi: PlayerApi = {
  async flowInfo(flowId) {
    const rec = await json<{ id: string; name: string; doc: FlowDocument }>(
      await authFetch(`/api/flows/${enc(flowId)}`),
    );
    return { id: rec.id, name: rec.name, kind: rec.doc.kind, doc: rec.doc };
  },
  async story(flowId) {
    return json<StoryManifest>(
      await authFetch(`/api/flows/${enc(flowId)}/story`),
    );
  },
  async world(flowId) {
    return json<WorldSpec>(await authFetch(`/api/flows/${enc(flowId)}/world`));
  },
  async invoke(flowId, entry, inputs, opts) {
    const res = await authFetch(
      `/api/flows/${enc(flowId)}/invoke`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entry,
          inputs,
          session_id: opts?.sessionId,
          rev: opts?.rev,
        }),
        signal: opts?.signal,
      },
      { retryNetworkErrors: false },
    );
    return json<InvokeResult>(res);
  },
  async listSessions(flowId) {
    const body = await json<{ sessions: SessionRecord[] }>(
      await authFetch(`/api/flows/${enc(flowId)}/sessions`),
    );
    return body.sessions ?? [];
  },
  async loadSession(flowId, slot) {
    const res = await authFetch(
      `/api/flows/${enc(flowId)}/sessions/${enc(slot)}`,
    );
    if (res.status === 404) return null;
    return json<SessionRecord>(res);
  },
  async saveSession(flowId, slot, state, label) {
    await json<unknown>(
      await authFetch(`/api/flows/${enc(flowId)}/sessions/${enc(slot)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, label }),
      }),
    );
  },
  async reroll(flowId, seed, onStatus, signal) {
    const rec = await json<{ doc: FlowDocument }>(
      await authFetch(`/api/flows/${enc(flowId)}`),
    );
    const doc = { ...rec.doc, seed };
    const { run_id } = await json<{ run_id: string }>(
      await authFetch(`/api/flows/${enc(flowId)}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc }),
        signal,
      }),
    );
    for (;;) {
      await sleep(1500, signal);
      const snap = await json<{
        run: { status: RunStatus; error?: string | null };
      }>(await authFetch(`/api/flows/runs/${enc(run_id)}`, { signal }));
      onStatus?.(snap.run.status);
      if (snap.run.status === "completed") return;
      if (["failed", "cancelled", "interrupted"].includes(snap.run.status)) {
        throw new PlayerApiError(
          snap.run.error || `The run ${snap.run.status}`,
          500,
        );
      }
    }
  },
  async mediaBlob(media) {
    const res = await authFetch(mediaPath(media));
    if (!res.ok)
      throw new PlayerApiError(
        `Could not load media (${res.status})`,
        res.status,
      );
    return res.blob();
  },
};

const MAX_URLS = 96;
const urlCache = new Map<string, Promise<string>>();

/** An object URL for a MediaValue, fetched with auth once and kept in a small LRU. */
export function mediaObjectUrl(
  api: PlayerApi,
  media: MediaValue,
): Promise<string> {
  const key = media.ref + (media.url ?? "");
  const hit = urlCache.get(key);
  if (hit) {
    urlCache.delete(key);
    urlCache.set(key, hit);
    return hit;
  }
  const p = api.mediaBlob(media).then((b) => URL.createObjectURL(b));
  p.catch(() => urlCache.delete(key));
  urlCache.set(key, p);
  while (urlCache.size > MAX_URLS) {
    const [oldKey, oldUrl] = urlCache.entries().next().value as [
      string,
      Promise<string>,
    ];
    urlCache.delete(oldKey);
    oldUrl.then((u) => URL.revokeObjectURL(u)).catch(() => undefined);
  }
  return p;
}

export function clearMediaUrls(): void {
  for (const p of urlCache.values())
    p.then((u) => URL.revokeObjectURL(u)).catch(() => undefined);
  urlCache.clear();
}
