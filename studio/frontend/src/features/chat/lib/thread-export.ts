// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { db } from "../db";
import type { MessageRecord, ThreadRecord } from "../types";
import { downloadTextFile } from "@/lib/download";

function extractText(content: MessageRecord["content"]): string {
  if (!Array.isArray(content)) return "";
  return content
    .filter((p) => p.type === "text")
    .map((p) => (p as { type: "text"; text: string }).text)
    .join("\n");
}

async function getThreadData(threadId: string) {
  const thread = await db.threads.get(threadId);
  if (!thread) return null;
  const messages = await db.messages
    .where("threadId")
    .equals(threadId)
    .sortBy("createdAt");
  return { thread, messages };
}

export async function exportAsMarkdown(threadId: string): Promise<void> {
  const data = await getThreadData(threadId);
  if (!data) return;
  const { thread, messages } = data;

  const lines: string[] = [
    `# ${thread.title}`,
    "",
    `> Exported from Unsloth Studio on ${new Date().toISOString()}`,
    `> Model: ${thread.modelId || "unknown"}`,
    "",
  ];

  for (const msg of messages) {
    const role = msg.role === "user" ? "User" : "Assistant";
    const text = extractText(msg.content);
    const feedback =
      msg.feedback ? ` [${msg.feedback === "thumbs_up" ? "+" : "-"}]` : "";
    lines.push(`## ${role}${feedback}`, "", text, "");
  }

  downloadTextFile(
    buildExportFilename(thread, "md"),
    lines.join("\n"),
    "text/markdown",
  );
}

export async function exportAsJSON(threadId: string): Promise<void> {
  const data = await getThreadData(threadId);
  if (!data) return;
  const { thread, messages } = data;

  const payload = {
    thread: { ...thread },
    messages: messages.map((m) => ({ ...m })),
    exportedAt: new Date().toISOString(),
  };

  downloadTextFile(
    buildExportFilename(thread, "json"),
    JSON.stringify(payload, null, 2),
    "application/json",
  );
}

export async function exportAsJSONL(threadId: string): Promise<void> {
  const data = await getThreadData(threadId);
  if (!data) return;
  const { thread, messages } = data;

  // OpenAI chat format -- standard SFT structure, no extra fields
  const chatMessages = messages.map((m) => ({
    role: m.role === "user" ? "user" : "assistant",
    content: extractText(m.content),
  }));

  const line = JSON.stringify({ messages: chatMessages });
  downloadTextFile(
    buildExportFilename(thread, "jsonl"),
    line + "\n",
    "application/x-ndjson",
  );
}

function sanitizeFilename(name: string): string {
  return name
    .replace(/[^\p{L}\p{N}_\- ]/gu, "")
    .replace(/\s+/g, "_")
    .slice(0, 80)
    || "chat_export";
}

function buildExportFilename(
  thread: ThreadRecord,
  ext: "md" | "json" | "jsonl",
): string {
  const base = sanitizeFilename(thread.title);
  const suffix = thread.pairId
    ? `_${sanitizeFilename(thread.modelId || thread.modelType || "compare")}`
    : "";
  return `${base}${suffix}.${ext}`;
}

export async function getExportThreadIds(
  threadOrPairId: string,
  type: "single" | "compare",
): Promise<string[]> {
  if (type === "single") return [threadOrPairId];
  const paired = await db.threads
    .where("pairId")
    .equals(threadOrPairId)
    .toArray();
  return paired.map((t: ThreadRecord) => t.id);
}
