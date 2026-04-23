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

async function getBenchmarkRecords(
  benchmarkId: string,
): Promise<BenchmarkResultRecord[]> {
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

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role !== "user") continue;
      const response = messages
        .slice(i + 1)
        .find((m) => m.role === "assistant");
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
        inference_params:
          (meta?.inferenceParams as Record<string, unknown>) ?? {},
        timestamp: new Date(response.createdAt).toISOString(),
      });
    }
  }

  return records;
}

/** Save a blob to disk. Uses showSaveFilePicker when available so the user
 *  chooses the destination every time; falls back to a <a> download. */
async function saveBlob(
  content: string,
  mimeType: string,
  suggestedName: string,
  extensions: string[],
): Promise<void> {
  if (typeof window !== "undefined" && "showSaveFilePicker" in window) {
    try {
      const fileHandle = await (
        window as unknown as {
          showSaveFilePicker: (opts: unknown) => Promise<FileSystemFileHandle>;
        }
      ).showSaveFilePicker({
        suggestedName,
        types: [
          {
            description: mimeType,
            accept: { [mimeType]: extensions },
          },
        ],
      });
      const writable = await fileHandle.createWritable();
      await writable.write(content);
      await writable.close();
      return;
    } catch (err) {
      // User cancelled or API not available — fall through to <a> download
      if ((err as { name?: string })?.name === "AbortError") return;
    }
  }
  // Fallback
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const dateSuffix = () => new Date().toISOString().slice(0, 10);

export async function downloadBenchmarkJsonl(
  benchmarkId: string,
): Promise<void> {
  const records = await getBenchmarkRecords(benchmarkId);
  const jsonl = records.map((r) => JSON.stringify(r)).join("\n");
  await saveBlob(
    jsonl,
    "application/jsonl",
    `benchmark-${benchmarkId.slice(0, 8)}-${dateSuffix()}.jsonl`,
    [".jsonl"],
  );
}

export async function downloadBenchmarkCsv(
  benchmarkId: string,
): Promise<void> {
  const records = await getBenchmarkRecords(benchmarkId);

  const HEADERS: (keyof BenchmarkResultRecord)[] = [
    "run_id",
    "run_name",
    "model_id",
    "model_name",
    "prompt_id",
    "prompt_text",
    "response_text",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "tokens_per_sec",
    "timestamp",
  ];

  function csvCell(value: unknown): string {
    const s = value == null ? "" : String(value);
    return `"${s.replace(/"/g, '""')}"`;
  }

  const rows = [
    HEADERS.join(","),
    ...records.map((r) =>
      HEADERS.map((h) => csvCell(r[h])).join(","),
    ),
  ].join("\r\n");

  await saveBlob(
    rows,
    "text/csv",
    `benchmark-${benchmarkId.slice(0, 8)}-${dateSuffix()}.csv`,
    [".csv"],
  );
}
