// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The voice decision log's read side: stats for the settings cards, the JSONL
// export (saved through the app's file-save path) and clearing it.

import { authFetch } from "@/features/auth";
import { downloadFile } from "@/lib/native-files";
import { voiceTrace } from "./voice-trace";

export type LayaModel = "judge" | "filler" | "interject" | "addressed";

export interface LayaModelStats {
  verdicts: number;
  with_outcome: number;
  accuracy: number | null;
  errors: Record<string, number>;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
}

export interface LayaStats {
  models: Record<LayaModel, LayaModelStats>;
  rows: number;
  log_bytes: number;
  path: string;
}

export async function fetchLayaStats(): Promise<LayaStats | null> {
  try {
    const res = await authFetch("/api/inference/voice/laya/stats");
    if (!res.ok) return null;
    return (await res.json()) as LayaStats;
  } catch {
    return null;
  }
}

export interface LayaExportOptions {
  model?: LayaModel | "all";
  only?: "disagreements" | "all";
  threadId?: string | null;
  /** ISO time; rows logged before it are left out. */
  since?: string | null;
  mix?: boolean;
}

/** Download the export as JSONL; resolves to the row count. Throws on a failed fetch or a cancelled save. */
export async function exportLayaLog(opts: LayaExportOptions = {}): Promise<number> {
  const params = new URLSearchParams();
  params.set("model", opts.model ?? "all");
  params.set("only", opts.only ?? "disagreements");
  if (opts.threadId) params.set("thread_id", opts.threadId);
  if (opts.since) params.set("since", opts.since);
  if (opts.mix) params.set("mix", "1");
  const res = await authFetch(`/api/inference/voice/laya/export?${params.toString()}`);
  if (!res.ok) throw new Error(`Export failed with status ${res.status}.`);
  const text = await res.text();
  const rows = Number(res.headers.get("X-Row-Count") ?? text.split("\n").filter(Boolean).length);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const scope = opts.threadId ? `chat-${opts.threadId.slice(0, 8)}` : params.get("model");
  await downloadFile(text, `laya-${scope}-${stamp}.jsonl`, "application/jsonl");
  voiceTrace("laya.export", { ...opts, rows });
  return rows;
}

export async function clearLayaLog(): Promise<number> {
  const res = await authFetch("/api/inference/voice/laya/log", { method: "DELETE" });
  if (!res.ok) throw new Error(`Clear failed with status ${res.status}.`);
  const removed = ((await res.json()) as { removed?: number }).removed ?? 0;
  voiceTrace("laya.clear", { removed });
  return removed;
}
