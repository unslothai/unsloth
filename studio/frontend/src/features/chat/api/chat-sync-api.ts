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

export function syncCreateThread(data: {
  id: string;
  title: string;
  model_type: string;
  model_id: string;
  pair_id?: string;
  created_at: number;
}): void {
  void syncFetch("/api/chat/threads", jsonBody(data));
}

export function syncUpdateThread(
  threadId: string,
  data: { title?: string; archived?: boolean },
): void {
  void syncFetch(`/api/chat/threads/${threadId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function syncDeleteThread(threadId: string): void {
  void syncFetch(`/api/chat/threads/${threadId}`, { method: "DELETE" });
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
  void syncFetch(`/api/chat/threads/${threadId}/messages`, jsonBody(data));
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
