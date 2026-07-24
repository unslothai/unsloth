// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type MemoryScope = "global" | "project";

export interface ChatMemory {
  id: string;
  scope: MemoryScope;
  projectId: string | null;
  content: string;
  sourceType: "manual" | "explicit" | "heuristic" | "model";
  sourceThreadId: string | null;
  sourceMessageId: string | null;
  createdAt: number;
  updatedAt: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(path, init);
  const body = (await response.json().catch(() => null)) as {
    detail?: string;
  } | null;

  if (!response.ok) {
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }

  return body as T;
}

export function listChatMemories(scope?: MemoryScope, projectId?: string) {
  const params = new URLSearchParams();
  if (scope) params.set("scope", scope);
  if (projectId) params.set("project_id", projectId);

  return request<{ memories: ChatMemory[] }>(
    `/api/chat/memories${params.size ? `?${params}` : ""}`,
  );
}

export function createChatMemory(payload: {
  content: string;
  scope: MemoryScope;
  projectId?: string | null;
}) {
  return request<{ memory: ChatMemory | null; created: boolean }>(
    "/api/chat/memories",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export function updateChatMemory(
  id: string,
  payload: {
    content: string;
    scope: MemoryScope;
    projectId?: string | null;
  },
) {
  return request<{ memory: ChatMemory }>(
    `/api/chat/memories/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export function deleteChatMemory(id: string) {
  return request<{ deleted: boolean }>(
    `/api/chat/memories/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
}

export function clearChatMemories(scope: MemoryScope, projectId?: string) {
  const params = new URLSearchParams({ scope });
  if (projectId) params.set("project_id", projectId);

  return request<{ deleted: number }>(`/api/chat/memories?${params}`, {
    method: "DELETE",
  });
}

export function exportChatMemories() {
  return request<{ version: number; memories: ChatMemory[] }>(
    "/api/chat/memories/export",
  );
}

export function applyChatMemoryCapture(
  payload: {
    threadId: string;
    sourceMessageId: string;
    rawOutput: string;
  },
  signal?: AbortSignal,
) {
  return request<{ memories: ChatMemory[] }>(
    "/api/chat/memories/apply-capture",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),

      signal,
    },
  );
}
