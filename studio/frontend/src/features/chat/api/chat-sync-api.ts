// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

const MAX_RETRIES = 2;
const BACKOFF_MS = [1000, 2000];

async function syncFetch(
  url: string,
  options: RequestInit,
  retries = MAX_RETRIES,
): Promise<Response | null> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await authFetch(url, options);
      if (res.ok) return res;
      if (res.status >= 400 && res.status < 500) return res; // don't retry client errors
    } catch {
      // network error, retry
    }
    if (attempt < retries) {
      await new Promise((r) => setTimeout(r, BACKOFF_MS[attempt]));
    }
  }
  console.warn(`[chat-sync] Failed after ${retries + 1} attempts: ${options.method} ${url}`);
  return null;
}

function jsonBody(data: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
}

// Per-thread queue ensures create lands before any update/message/delete
const threadQueues = new Map<string, Promise<unknown>>();

function enqueueThreadSync(
  threadId: string,
  op: () => Promise<Response | null>,
): void {
  const prev = threadQueues.get(threadId) ?? Promise.resolve();
  const next = prev
    .catch(() => undefined)
    .then(op)
    .finally(() => {
      if (threadQueues.get(threadId) === next) threadQueues.delete(threadId);
    });
  threadQueues.set(threadId, next);
  void next;
}

export function syncCreateThread(data: {
  id: string;
  title: string;
  model_type: string;
  model_id: string;
  pair_id?: string;
  created_at: number;
}): void {
  enqueueThreadSync(data.id, () => syncFetch("/api/chat/threads", jsonBody(data)));
}

export function syncUpdateThread(
  threadId: string,
  data: { title?: string; archived?: boolean },
): void {
  enqueueThreadSync(threadId, () =>
    syncFetch(`/api/chat/threads/${threadId}`, { ...jsonBody(data), method: "PATCH" }),
  );
}

export function syncDeleteThread(threadId: string): void {
  enqueueThreadSync(threadId, () =>
    syncFetch(`/api/chat/threads/${threadId}`, { method: "DELETE" }),
  );
}

export function syncUpsertMessage(
  threadId: string,
  data: {
    id: string;
    role: string;
    content: string;
    attachments?: string;
    metadata?: string;
    created_at: number;
  },
): void {
  enqueueThreadSync(threadId, () =>
    syncFetch(`/api/chat/threads/${threadId}/messages`, jsonBody(data)),
  );
}

export interface HydrateThread {
  id: string;
  title: string;
  model_type: string;
  model_id: string;
  pair_id: string | null;
  archived: boolean;
  created_at: number;
  updated_at: number;
}

export interface HydrateMessage {
  id: string;
  thread_id: string;
  role: string;
  content: string;
  attachments: string | null;
  metadata: string | null;
  created_at: number;
}

export interface HydrateResponse {
  threads: HydrateThread[];
  messages: HydrateMessage[];
}

export async function fetchHydrate(): Promise<HydrateResponse | null> {
  try {
    const res = await authFetch("/api/chat/hydrate");
    if (!res.ok) return null;
    return (await res.json()) as HydrateResponse;
  } catch {
    return null;
  }
}
