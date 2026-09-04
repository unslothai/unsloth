// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type {
  AdapterRow,
  CompactReport,
  MemoryRecord,
  MemorySummary,
  OperatorItem,
  PackRow,
} from "../types";

const BASE = "/api/unforgettable";

async function request<T>(
  path: string,
  init?: { method?: string; body?: object },
): Promise<T> {
  const response = await authFetch(`${BASE}${path}`, {
    method: init?.method,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  const json = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      json && typeof json === "object"
        ? formatFastApiDetail((json as { detail?: unknown }).detail)
        : null;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return json as T;
}

export function fetchSummary(): Promise<MemorySummary> {
  return request("/summary");
}

export function fetchRecords(params: {
  status?: string;
  kind?: string;
  q?: string;
  provenance?: string;
  limit?: number;
  offset?: number;
}): Promise<{ records: MemoryRecord[] }> {
  const query = new URLSearchParams();
  if (params.status) query.set("status", params.status);
  if (params.kind) query.set("kind", params.kind);
  if (params.q) query.set("q", params.q);
  if (params.provenance) query.set("provenance", params.provenance);
  if (params.limit != null) query.set("limit", String(params.limit));
  if (params.offset != null) query.set("offset", String(params.offset));
  const suffix = query.toString() ? `?${query}` : "";
  return request(`/records${suffix}`);
}

export function fetchRecord(id: string): Promise<MemoryRecord> {
  return request(`/records/${encodeURIComponent(id)}`);
}

export function patchProposedRecord(
  id: string,
  payload: { title?: string; body?: string },
): Promise<MemoryRecord> {
  return request(`/records/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: payload,
  });
}

export function admitRecord(
  id: string,
  force = false,
): Promise<MemoryRecord> {
  return request(`/records/${encodeURIComponent(id)}/admit`, {
    method: "POST",
    body: { force },
  });
}

export function rejectRecord(
  id: string,
  reason?: string,
): Promise<MemoryRecord> {
  return request(`/records/${encodeURIComponent(id)}/reject`, {
    method: "POST",
    body: { reason },
  });
}

export function deprecateRecord(
  id: string,
  reason?: string,
): Promise<MemoryRecord> {
  return request(`/records/${encodeURIComponent(id)}/deprecate`, {
    method: "POST",
    body: { reason },
  });
}

export function fetchCompiled(): Promise<{ records: MemoryRecord[] }> {
  return request("/compiled");
}

export function compileRecord(id: string): Promise<MemoryRecord> {
  return request(`/compile/${encodeURIComponent(id)}`, { method: "POST" });
}

export function uncompileRecord(id: string): Promise<MemoryRecord> {
  return request(`/uncompile/${encodeURIComponent(id)}`, { method: "POST" });
}

export function fetchContradictions(): Promise<{
  contradictions: {
    title_key: string;
    record_ids: string[];
    reason: string;
  }[];
}> {
  return request("/contradictions");
}

export function fetchAdmissions(): Promise<{
  admissions: Record<string, unknown>[];
}> {
  return request("/admissions");
}

export function fetchRollouts(): Promise<{
  rollouts: Record<string, unknown>[];
}> {
  return request("/rollouts");
}

export function runCompact(apply = false): Promise<CompactReport> {
  return request("/compact", { method: "POST", body: { apply } });
}

export function runReview(apply = false): Promise<{ items: OperatorItem[] }> {
  return request("/review", { method: "POST", body: { apply } });
}

export function runMine(apply = false): Promise<{ items: OperatorItem[] }> {
  return request("/mine", { method: "POST", body: { apply } });
}

export function fetchAdapters(): Promise<{ adapters: AdapterRow[] }> {
  return request("/adapters");
}

export function fetchPacks(): Promise<{ packs: PackRow[] }> {
  return request("/packs");
}

export function promoteAdapter(
  id: string,
  force = false,
): Promise<AdapterRow> {
  return request(`/adapters/${encodeURIComponent(id)}/promote`, {
    method: "POST",
    body: { force },
  });
}

export function rollbackAdapter(): Promise<{ promoted: AdapterRow | null }> {
  return request("/adapters/rollback", { method: "POST" });
}
