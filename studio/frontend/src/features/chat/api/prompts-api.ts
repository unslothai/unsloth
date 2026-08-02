// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export interface PromptEntry {
  id: string;
  name: string;
  text: string;
  createdAt: number;
  updatedAt: number;
}

export interface PromptListEntry {
  id: string;
  name: string;
  items: string[];
  createdAt: number;
  updatedAt: number;
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${res.status})`);
  }
  return body as T;
}

export async function listPromptEntries(): Promise<PromptEntry[]> {
  const res = await authFetch("/api/prompts/entries");
  const data = await parseJsonOrThrow<{ entries: PromptEntry[] }>(res);
  return data.entries;
}

export async function savePromptEntry(entry: PromptEntry): Promise<PromptEntry> {
  const res = await authFetch(`/api/prompts/entries/${entry.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  return parseJsonOrThrow<PromptEntry>(res);
}

export async function deletePromptEntry(id: string): Promise<void> {
  const res = await authFetch(`/api/prompts/entries/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}

export async function bulkSavePromptEntries(entries: PromptEntry[]): Promise<number> {
  if (!entries.length) return 0;
  const res = await authFetch("/api/prompts/entries/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entries }),
  });
  const data = await parseJsonOrThrow<{ count: number }>(res);
  return data.count;
}

export async function listPromptLists(): Promise<PromptListEntry[]> {
  const res = await authFetch("/api/prompts/lists");
  const data = await parseJsonOrThrow<{ lists: PromptListEntry[] }>(res);
  return data.lists;
}

export async function savePromptList(list: PromptListEntry): Promise<PromptListEntry> {
  const res = await authFetch(`/api/prompts/lists/${list.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(list),
  });
  return parseJsonOrThrow<PromptListEntry>(res);
}

export async function deletePromptList(id: string): Promise<void> {
  const res = await authFetch(`/api/prompts/lists/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}

export async function bulkSavePromptLists(lists: PromptListEntry[]): Promise<number> {
  if (!lists.length) return 0;
  const res = await authFetch("/api/prompts/lists/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lists }),
  });
  const data = await parseJsonOrThrow<{ count: number }>(res);
  return data.count;
}
