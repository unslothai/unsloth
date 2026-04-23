// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { db } from "../../db";
import type { BenchmarkResultRecord } from "../types";

function extractText(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (p): p is { type: "text"; text: string } =>
        typeof p === "object" && p !== null && p.type === "text",
    )
    .map((p) => p.text)
    .join("");
}

export async function downloadBenchmarkJsonl(benchmarkId: string): Promise<void> {
  const threads = await db.threads
    .where("benchmarkId")
    .equals(benchmarkId)
    .sortBy("createdAt");

  const runName = threads[0]?.benchmarkName ?? benchmarkId;
  const records: BenchmarkResultRecord[] = [];

  for (const thread of threads) {
    const messages = await db.messages
      .where("threadId")
      .equals(thread.id)
      .sortBy("createdAt");

    // Pair user messages with their following assistant response.
    // Uses forward scanning instead of index offsets to handle tool-call
    // messages or retries that may appear between user and assistant turns.
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role !== "user") continue;
      const response = messages.slice(i + 1).find((m) => m.role === "assistant");
      if (!response) continue;

      const meta = response.metadata as Record<string, unknown> | undefined;
      records.push({
        run_id: benchmarkId,
        run_name: runName,
        model_id: thread.modelId ?? "",
        model_name: thread.title,
        prompt_id: msg.id,
        prompt_text: extractText(msg.content),
        response_text: extractText(response.content),
        latency_ms: meta?.latencyMs as number | undefined,
        input_tokens: meta?.inputTokens as number | undefined,
        output_tokens: meta?.outputTokens as number | undefined,
        tokens_per_sec: meta?.tokensPerSec as number | undefined,
        inference_params: (meta?.inferenceParams as Record<string, unknown>) ?? {},
        timestamp: new Date(response.createdAt).toISOString(),
      });
    }
  }

  const jsonl = records.map((r) => JSON.stringify(r)).join("\n");
  const blob = new Blob([jsonl], { type: "application/jsonl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `benchmark-${benchmarkId.slice(0, 8)}-${new Date().toISOString().slice(0, 10)}.jsonl`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
