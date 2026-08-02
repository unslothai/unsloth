// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";

import { cn } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  BookmarkIcon,
  DownloadIcon,
  GripVerticalIcon,
  LayoutListIcon,
  PencilIcon,
  PlayIcon,
  PlusIcon,
  Trash2Icon,
  UploadIcon,
  XIcon,
} from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import {
  type PromptEntry,
  type PromptListEntry,
  bulkSavePromptEntries,
  bulkSavePromptLists,
  deletePromptEntry,
  deletePromptList,
  listPromptEntries,
  listPromptLists,
  savePromptEntry,
  savePromptList,
} from "../api/prompts-api";
import {
  listStoredChatMessages,
  listStoredChatThreads,
  saveStoredChatThread,
  syncStoredChatMessages,
} from "../utils/chat-history-storage";
import { notifyChatHistoryUpdated } from "../api/chat-api";
import { isMcpImageToolResult } from "../api/chat-adapter";
import { usePlusMenuPrefsStore } from "../stores/plus-menu-prefs-store";
import type { ThreadRecord, MessageRecord } from "../types";

function newId(): string {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
}

function now(): number {
  return Date.now();
}

function sanitizeFilename(name: string): string {
  return name.replace(/[\\/:*?"<>|]/g, "_").slice(0, 80) || "export";
}

async function downloadBlob(
  content: string | Blob | Uint8Array,
  filename: string,
  mimeType: string,
): Promise<void> {
  return downloadFile(content, filename, mimeType);
}

function csvEscape(val: string): string {
  return `"${val.replace(/"/g, '""')}"`;
}

function exportPromptJsonl(entry: PromptEntry): Promise<void> {
  return downloadBlob(
    JSON.stringify({ name: entry.name, text: entry.text }),
    `${sanitizeFilename(entry.name)}.jsonl`,
    "application/x-ndjson",
  );
}

function exportPromptCsv(entry: PromptEntry): Promise<void> {
  return downloadBlob(
    `name,text\n${csvEscape(entry.name)},${csvEscape(entry.text)}`,
    `${sanitizeFilename(entry.name)}.csv`,
    "text/csv",
  );
}

function exportAllPromptsJsonl(entries: PromptEntry[]): Promise<void> {
  const lines = entries.map((e) => JSON.stringify({ name: e.name, text: e.text })).join("\n");
  return downloadBlob(lines, "prompts.jsonl", "application/x-ndjson");
}

function exportAllPromptsCsv(entries: PromptEntry[]): Promise<void> {
  const rows = entries.map((e) => `${csvEscape(e.name)},${csvEscape(e.text)}`).join("\n");
  return downloadBlob(`name,text\n${rows}`, "prompts.csv", "text/csv");
}

function exportListJsonl(entry: PromptListEntry): Promise<void> {
  return downloadBlob(
    JSON.stringify({ name: entry.name, items: entry.items }),
    `${sanitizeFilename(entry.name)}.jsonl`,
    "application/x-ndjson",
  );
}

function exportAllListsJsonl(entries: PromptListEntry[]): Promise<void> {
  const lines = entries.map((e) => JSON.stringify({ name: e.name, items: e.items })).join("\n");
  return downloadBlob(lines, "prompt-lists.jsonl", "application/x-ndjson");
}

function exportListCsv(entry: PromptListEntry): Promise<void> {
  const rows = entry.items
    .map((text, i) => `${csvEscape(entry.name)},${i + 1},${csvEscape(text)}`)
    .join("\n");
  return downloadBlob(
    `list_name,order,prompt_text\n${rows}`,
    `${sanitizeFilename(entry.name)}.csv`,
    "text/csv",
  );
}

function exportAllListsCsv(entries: PromptListEntry[]): Promise<void> {
  const rows = entries
    .flatMap((e) => e.items.map((text, i) => `${csvEscape(e.name)},${i + 1},${csvEscape(text)}`))
    .join("\n");
  return downloadBlob(`list_name,order,prompt_text\n${rows}`, "prompt-lists.csv", "text/csv");
}

function contentBlocksToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return JSON.stringify(content);
  const parts: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
      const p = part as Record<string, unknown>;
      if (p.type === "text" && typeof p.text === "string") {
        parts.push(p.text);
      } else if (p.type === "reasoning" || p.type === "thinking") {
        const thinkText =
          typeof p.thinking === "string"
            ? p.thinking
            : typeof p.text === "string"
              ? p.text
              : "";
        if (thinkText) {
          parts.push("[thinking]\n" + thinkText + "\n[/thinking]");
        }
      } else if (p.type === "tool-call") {
        // Keep base64 image payloads out of every export format: use the
        // model-visible text for MCP image results (matches chat replay).
        const result = isMcpImageToolResult(p.result) ? p.result.text : p.result;
        parts.push(
          JSON.stringify({
            tool_call: p.toolName,
            args: p.args,
            result,
          }),
        );
      } else if (p.type === "image") {
        parts.push("[image attachment]");
      } else if (p.type === "audio") {
        parts.push("[audio attachment]");
      }
    }
    return parts.join("\n\n");
  }

// Order via parentId chain: createdAt misorders turns (GPT response slots
// predate the user's next message); the parent chain is timestamp-independent.
type _Msg = { id: string; parentId?: string | null; createdAt?: number };

function orderByParentChain<T extends _Msg>(
  messages: T[],
  options: {
    /** Append messages off the selected chain (abandoned branches) at the
     *  end. Full exports keep everything; fine-tune conversion must not,
     *  since alternate replies would merge into one conversation. */
    includeSiblings?: boolean;
  } = {},
): T[] {
  const { includeSiblings = true } = options;
  const byId = new Map<string, T>(messages.map((m) => [m.id, m]));
  const childrenOf = new Map<string | null, T[]>();
  for (const m of messages) {
    const pid = m.parentId ?? null;
    if (!childrenOf.has(pid)) childrenOf.set(pid, []);
    childrenOf.get(pid)!.push(m);
  }

  const result: T[] = [];
  let cur: string | null = null;
  while (childrenOf.has(cur)) {
    const children: T[] = childrenOf.get(cur)!;
    const next: T = children.reduce((a: T, b: T) =>
      (a.createdAt ?? 0) >= (b.createdAt ?? 0) ? a : b,
    );
    result.push(next);
    cur = next.id;
    byId.delete(next.id);
  }

  if (includeSiblings) {
    for (const [, m] of byId) result.push(m);
  }
  return result;
}

async function loadConversationMessages(threadId: string) {
  const raw = await listStoredChatMessages(threadId);
  if (raw.length === 0) {
    toast.info("No messages in this conversation to export.");
    return null;
  }
  // No parentId = legacy flat thread (already DB createdAt-sorted); walking the
  // chain would invert order, so keep raw order.
  const hasParentIds = raw.some((m) => (m as { parentId?: unknown }).parentId != null);
  if (!hasParentIds) return raw;
  return orderByParentChain(raw) as typeof raw;
}

function exportTs(): string {
  return new Date().toISOString().slice(0, 19).replace(/:/g, "-");
}

// Attachments live in msg.attachments[].content, not msg.content, so flatten
// both here or they'd be dropped on export.
function messageToText(msg: { content: unknown; attachments?: unknown }): string {
  const parts: string[] = [];
  const main = contentBlocksToText(msg.content);
  if (main) parts.push(main);
  if (Array.isArray(msg.attachments)) {
    for (const attachment of msg.attachments as Array<{ content?: unknown }>) {
      if (!attachment?.content) continue;
      const attText = contentBlocksToText(attachment.content);
      if (attText) parts.push(attText);
    }
  }
  return parts.join("\n\n");
}

// OpenAI messages array (tool-calling + multimodal fine-tuning): tool calls →
// "tool_calls" + separate "role":"tool" messages; images → "image_url" parts;
// audio dropped; thinking kept as a text part.

type OAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

type OAIToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

type OAIMessage =
  | { role: "user" | "system"; content: string | OAIContentPart[] }
  | { role: "assistant"; content: string | null; tool_calls?: OAIToolCall[] }
  | { role: "tool"; tool_call_id: string; name: string; content: string };

function messageToOpenAI(msg: { role: unknown; content: unknown; attachments?: unknown }): OAIMessage[] {
  const role = (msg.role as string) ?? "user";
  const blocks = Array.isArray(msg.content) ? msg.content : [];
  const attachments = Array.isArray(msg.attachments) ? msg.attachments : [];

  const allParts: Record<string, unknown>[] = [
    ...blocks.map((b) => b as Record<string, unknown>),
    ...attachments.flatMap((a) => {
      const att = a as { content?: unknown };
      return Array.isArray(att.content)
        ? (att.content as Record<string, unknown>[])
        : [];
    }),
  ];

  if (role === "assistant") {
    const textParts: string[] = [];
    const toolCalls: OAIToolCall[] = [];
    const toolResults: OAIMessage[] = [];

    for (const p of allParts) {
      if (p.type === "text" && typeof p.text === "string") {
        textParts.push(p.text);
      } else if (p.type === "reasoning" || p.type === "thinking") {
        const t = typeof p.thinking === "string" ? p.thinking : typeof p.text === "string" ? p.text : "";
        if (t) textParts.push(`<thinking>\n${t}\n</thinking>`);
      } else if (p.type === "tool-call") {
        const id = typeof p.toolCallId === "string" ? p.toolCallId : `call_${toolCalls.length}`;
        const name = typeof p.toolName === "string" ? p.toolName : "unknown";
        const argsStr = p.args != null ? JSON.stringify(p.args) : (typeof p.argsText === "string" ? p.argsText : "{}");
        toolCalls.push({ id, type: "function", function: { name, arguments: argsStr } });
        if (p.result !== undefined && p.result !== null) {
          // Keep base64 image payloads out of exports: MCP image results carry
          // their model-visible text alongside the data, so serialize the text
          // (matching chat replay) instead of the full object.
          const resultStr =
            typeof p.result === "string"
              ? p.result
              : isMcpImageToolResult(p.result)
                ? p.result.text
                : JSON.stringify(p.result);
          toolResults.push({ role: "tool", tool_call_id: id, name, content: resultStr });
        }
      }
    }

    const content = textParts.join("\n\n") || null;
    const assistantMsg: OAIMessage = toolCalls.length > 0
      ? { role: "assistant", content, tool_calls: toolCalls }
      : { role: "assistant", content: content ?? "" };

    return toolResults.length > 0
      ? [assistantMsg, ...toolResults]
      : [assistantMsg];
  }

  const contentParts: OAIContentPart[] = [];
  let hasNonText = false;

  for (const p of allParts) {
    if (p.type === "text" && typeof p.text === "string") {
      contentParts.push({ type: "text", text: p.text });
    } else if (p.type === "image" && typeof p.image === "string") {
      contentParts.push({ type: "image_url", image_url: { url: p.image } });
      hasNonText = true;
    }
  }

  if (!hasNonText) {
    const text = contentParts.map((p) => (p.type === "text" ? p.text : "")).join("\n\n");
    return text ? [{ role: role as "user" | "system", content: text }] : [];
  }
  return contentParts.length > 0 ? [{ role: role as "user" | "system", content: contentParts }] : [];
}

// ShareGPT training JSONL (human/system/gpt turns).
export async function exportConversationShareGPT(threadId: string): Promise<void> {
  const messages = await loadConversationMessages(threadId);
  if (!messages) return;

  const conversations: Array<{ from: string; value: string }> = [];
  for (const msg of messages) {
    const role = msg.role as string;
    const from = role === "user" ? "human" : role === "system" ? "system" : "gpt";
    const value = messageToText(msg);
    if (value.trim()) conversations.push({ from, value });
  }

  if (conversations.length === 0) { toast.info("No exportable content."); return; }
  await downloadBlob(
    JSON.stringify({ conversations }),
    "conversation-" + exportTs() + ".jsonl",
    "application/x-ndjson",
  );
}

// OpenAI/ChatML JSONL: {"messages": [{"role","content"}, ...]} per conversation;
// Unsloth reads this as a ChatML dataset.
export async function exportConversationRawJsonl(threadId: string): Promise<void> {
  const messages = await loadConversationMessages(threadId);
  if (!messages) return;

  const oaiMsgs: OAIMessage[] = messages.flatMap((msg) => messageToOpenAI(msg));
  if (oaiMsgs.length === 0) { toast.info("No exportable content."); return; }
  await downloadBlob(
    JSON.stringify({ messages: oaiMsgs }),
    "conversation-" + exportTs() + ".jsonl",
    "application/x-ndjson",
  );
}

export async function exportConversationCsv(threadId: string): Promise<void> {
  const messages = await loadConversationMessages(threadId);
  if (!messages) return;

  const rows = ["role,content"];
  for (const msg of messages) {
    const content = messageToText(msg);
    if (!content.trim()) continue;
    rows.push(`${csvEscape(msg.role as string)},${csvEscape(content)}`);
  }

  if (rows.length <= 1) { toast.info("No exportable content."); return; }
  await downloadBlob(
    rows.join("\n"),
    "conversation-" + exportTs() + ".csv",
    "text/csv",
  );
}

export type ConvExportFormat = "jsonl-raw" | "csv" | "sharegpt";

const EXPORT_FORMAT_LABELS: Record<ConvExportFormat, string> = {
  "jsonl-raw": "Raw JSONL",
  csv: "CSV",
  sharegpt: "ShareGPT JSONL",
};

export const EXPORT_FORMATS_LIST = (
  Object.keys(EXPORT_FORMAT_LABELS) as ConvExportFormat[]
).map((fmt) => ({ fmt, label: EXPORT_FORMAT_LABELS[fmt] }));

async function buildThreadContent(
  threadId: string,
  format: ConvExportFormat,
): Promise<string | null> {
  const messages = await loadConversationMessages(threadId);
  if (!messages) return null;

  if (format === "jsonl-raw") {
    // OpenAI/ChatML: Unsloth reads the "messages" key as ChatML.
    const oaiMsgs: OAIMessage[] = messages.flatMap((msg) => messageToOpenAI(msg));
    if (oaiMsgs.length === 0) return null;
    return JSON.stringify({ messages: oaiMsgs });
  }

  if (format === "sharegpt") {
    const conversations: Array<{ from: string; value: string }> = [];
    for (const msg of messages) {
      const role = msg.role as string;
      const value = messageToText(msg);
      if (value.trim()) conversations.push({ from: role === "user" ? "human" : role === "system" ? "system" : "gpt", value });
    }
    if (conversations.length === 0) return null;
    return JSON.stringify({ conversations });
  }

  const rows: string[] = [];
  for (const msg of messages) {
    const content = messageToText(msg);
    if (!content.trim()) continue;
    rows.push(`${csvEscape(msg.role as string)},${csvEscape(content)}`);
  }
  return rows.length > 0 ? rows.join("\n") : null;
}

function csvHeader(format: ConvExportFormat): string {
  return format === "csv" ? "role,content" : "";
}

function exportExt(format: ConvExportFormat): string {
  return format === "csv" ? "csv" : "jsonl";
}

function exportMime(format: ConvExportFormat): string {
  return format === "csv" ? "text/csv" : "application/x-ndjson";
}

export async function exportBulkConversationsMerged(
  threadIds: string[],
  format: ConvExportFormat,
  basename: string,
): Promise<void> {
  if (threadIds.length === 0) { toast.info("No conversations to export."); return; }

  const parts: string[] = [];
  const header = csvHeader(format);

  for (const id of threadIds) {
    const content = await buildThreadContent(id, format);
    if (content) parts.push(content);
  }

  if (parts.length === 0) { toast.info("No exportable content."); return; }

  const body = header
    ? header + "\n" + parts.join("\n")
    : parts.join("\n");

  await downloadBlob(
    body,
    `${basename}.${exportExt(format)}`,
    exportMime(format),
  );
}

export async function exportBulkConversationsSeparate(
  threadIds: string[],
  format: ConvExportFormat,
  basename: string,
): Promise<void> {
  if (threadIds.length === 0) { toast.info("No conversations to export."); return; }

  const { zipSync, strToU8 } = await import("fflate");
  const ext = exportExt(format);
  const header = csvHeader(format);
  const files: Record<string, Uint8Array> = {};

  for (const id of threadIds) {
    const content = await buildThreadContent(id, format);
    if (!content) continue;
    const body = header ? header + "\n" + content : content;
    files[`${id}.${ext}`] = strToU8(body);
  }

  if (Object.keys(files).length === 0) { toast.info("No exportable content."); return; }

  const zipped = zipSync(files);
  await downloadBlob(zipped, `${basename}.zip`, "application/zip");
}

// Scope-level bulk export shared by the sidebar Recents menu and
// Settings -> Chat -> Data. "recents" = chats outside projects; "all" adds
// project chats.
export async function bulkExportConversationsByScope(
  scope: "recents" | "all",
  format: ConvExportFormat,
  merged: boolean,
): Promise<void> {
  try {
    const threads = await listStoredChatThreads({
      includeArchived: false,
      ...(scope === "recents" ? { projectId: null } : {}),
    });
    const ids = [...new Set(threads.map((t) => t.id))];
    if (ids.length === 0) {
      toast.info("No conversations to export.");
      return;
    }
    const ts = new Date().toISOString().slice(0, 10);
    const basename = scope === "all" ? `all-chats-${ts}` : `recents-${ts}`;
    if (merged) {
      await exportBulkConversationsMerged(ids, format, basename);
    } else {
      await exportBulkConversationsSeparate(ids, format, basename);
    }
  } catch (error) {
    if (!isDownloadCancelled(error)) {
      toast.error("Export failed.");
    }
  }
}

export async function exportProjectConversations(
  threadIds: string[],
  format: ConvExportFormat,
  projectName: string,
): Promise<void> {
  const safe = projectName.replace(/[^a-z0-9_-]/gi, "_").slice(0, 40);
  await exportBulkConversationsMerged(
    threadIds,
    format,
    `project-${safe}-${exportTs()}`,
  );
}

// ── Fine-tuning export ─────────────────────────────────────────────────────
// One JSONL line per conversation: {"messages": [{"role", "content"}]} with
// string-only content in system/user/assistant turns. Unsloth's training tab
// detects this as ChatML natively (no column mapping, no standardization) and
// it works with train-on-completions masking, which only trains on assistant
// turns. Reasoning, tool calls, and images are dropped: clean SFT targets.

export type FineTuneMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

const FINE_TUNE_ROLES = new Set(["system", "user", "assistant"]);

/** Plain text of a message: text blocks plus text-type attachment parts. */
function messageToPlainText(msg: {
  content: unknown;
  attachments?: unknown;
}): string {
  const parts: string[] = [];
  const collect = (blocks: unknown) => {
    // Legacy and imported histories can store content as a plain string.
    if (typeof blocks === "string") {
      if (blocks.trim()) parts.push(blocks);
      return;
    }
    if (!Array.isArray(blocks)) return;
    for (const b of blocks) {
      if (!b || typeof b !== "object") {
        continue;
      }
      const block = b as Record<string, unknown>;
      if (block.type === "text" && typeof block.text === "string" && block.text) {
        parts.push(block.text);
      }
    }
  };
  collect(msg.content);
  if (Array.isArray(msg.attachments)) {
    for (const attachment of msg.attachments as Array<{ content?: unknown }>) {
      collect(attachment?.content);
    }
  }
  return parts.join("\n\n").trim();
}

/** Merge consecutive same-role turns so chat templates format cleanly. */
function mergeSameRoleTurns(turns: FineTuneMessage[]): FineTuneMessage[] {
  const merged: FineTuneMessage[] = [];
  for (const turn of turns) {
    const last = merged[merged.length - 1];
    if (last && last.role === turn.role) {
      last.content += `\n\n${turn.content}`;
    } else {
      merged.push({ ...turn });
    }
  }
  return merged;
}

/** Conversation turns for fine-tuning, or null when the thread has no
 *  usable user + assistant exchange. Consecutive same-role turns merge,
 *  assistant turns before the first user turn drop (an assistant target
 *  with no prompt teaches nothing), and trailing non-assistant turns drop
 *  so chat templates format cleanly. */
function messagesToFineTuneTurns(
  messages: Array<{ role: unknown; content: unknown; attachments?: unknown }>,
): FineTuneMessage[] | null {
  const raw: FineTuneMessage[] = [];
  for (const msg of messages) {
    const role = msg.role as FineTuneMessage["role"];
    if (!FINE_TUNE_ROLES.has(role)) continue;
    const content = messageToPlainText(msg);
    if (!content) continue;
    raw.push({ role, content });
  }
  const firstUser = raw.findIndex((t) => t.role === "user");
  if (firstUser === -1) return null;
  const turns = mergeSameRoleTurns(
    raw.filter((t, i) => i >= firstUser || t.role === "system"),
  );
  while (turns.length > 0 && turns[turns.length - 1].role !== "assistant") {
    turns.pop();
  }
  const hasUser = turns.some((t) => t.role === "user");
  const hasAssistant = turns.some((t) => t.role === "assistant");
  return hasUser && hasAssistant ? turns : null;
}

export type FineTuneExportResult = {
  lines: string[];
  conversations: number;
  skipped: number;
};

/** Dataset shapes the Train tab detects without column mapping. */
export type FineTuneFormat = "openai" | "sharegpt" | "alpaca";

const SHAREGPT_FROM: Record<FineTuneMessage["role"], string> = {
  system: "system",
  user: "human",
  assistant: "gpt",
};

/** JSONL lines for one conversation in the chosen format. Alpaca is
 *  single-turn, so each user to assistant pair becomes its own record with
 *  the system prompt and earlier exchange carried in the input field. */
function turnsToFineTuneLines(
  turns: FineTuneMessage[],
  format: FineTuneFormat,
): string[] {
  if (format === "sharegpt") {
    return [
      JSON.stringify({
        conversations: turns.map((t) => ({
          from: SHAREGPT_FROM[t.role],
          value: t.content,
        })),
      }),
    ];
  }
  if (format === "alpaca") {
    const lines: string[] = [];
    const context: string[] = [];
    let system = "";
    let pendingUser: string | null = null;
    for (const t of turns) {
      if (t.role === "system") {
        system = system ? `${system}\n\n${t.content}` : t.content;
        continue;
      }
      if (t.role === "user") {
        pendingUser = t.content;
        continue;
      }
      if (pendingUser === null) continue;
      const inputParts = [];
      if (system) inputParts.push(system);
      if (context.length > 0) inputParts.push(context.join("\n"));
      lines.push(
        JSON.stringify({
          instruction: pendingUser,
          input: inputParts.join("\n\n"),
          output: t.content,
        }),
      );
      context.push(`User: ${pendingUser}`, `Assistant: ${t.content}`);
      pendingUser = null;
    }
    return lines;
  }
  return [JSON.stringify({ messages: turns })];
}

/** Every non-archived chat (Recents and Projects) as training-ready JSONL. */
export async function buildFineTuneJsonl(
  format: FineTuneFormat = "openai",
): Promise<FineTuneExportResult> {
  const threads = await listStoredChatThreads({ includeArchived: false });
  const ids = [...new Set(threads.map((t) => t.id))];
  const lines: string[] = [];
  let conversations = 0;
  let skipped = 0;
  for (const id of ids) {
    const raw = await listStoredChatMessages(id);
    const hasParentIds = raw.some(
      (m) => (m as { parentId?: unknown }).parentId != null,
    );
    // Chain only: retries/regenerations leave sibling branches, and mixing
    // alternate replies into one conversation corrupts the training targets.
    const ordered = hasParentIds
      ? (orderByParentChain(raw, { includeSiblings: false }) as typeof raw)
      : raw;
    const turns = messagesToFineTuneTurns(ordered);
    const converted = turns ? turnsToFineTuneLines(turns, format) : [];
    if (converted.length === 0) {
      skipped += 1;
      continue;
    }
    conversations += 1;
    lines.push(...converted);
  }
  return { lines, conversations, skipped };
}

/** Download the fine-tuning JSONL; returns the conversation count. */
export async function exportFineTuneJsonl(
  format: FineTuneFormat = "openai",
): Promise<number> {
  const { lines, conversations, skipped } = await buildFineTuneJsonl(format);
  if (conversations === 0) {
    toast.info("No chats with a user and assistant exchange to export.");
    return 0;
  }
  const suffix = format === "openai" ? "" : `-${format}`;
  await downloadBlob(
    lines.join("\n"),
    `chat-finetune${suffix}-${exportTs()}.jsonl`,
    "application/x-ndjson",
  );
  if (skipped > 0) {
    toast.success(
      `Exported ${conversations} conversation${conversations === 1 ? "" : "s"} (${skipped} without a full exchange skipped).`,
    );
  }
  return conversations;
}

// role:"tool" results are absorbed into the preceding assistant tool-call
// part's `result` field rather than becoming separate records.
function oaiMessagesToRecords(
  oaiMsgs: unknown[],
  threadId: string,
  baseTs: number,
): MessageRecord[] {
  const toolResults = new Map<string, string>();
  for (const m of oaiMsgs) {
    const msg = m as Record<string, unknown>;
    if (msg.role === "tool" && typeof msg.tool_call_id === "string") {
      toolResults.set(msg.tool_call_id, typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? ""));
    }
  }

  const records: MessageRecord[] = [];
  let prevId: string | null = null;
  let idx = 0;

  for (const m of oaiMsgs) {
    const msg = m as Record<string, unknown>;
    const role = msg.role as string;
    if (role === "tool") continue;

    const id = crypto.randomUUID();

    let content: unknown[];

    if (role === "assistant") {
      const parts: unknown[] = [];
      if (typeof msg.content === "string" && msg.content.trim()) {
        parts.push({ type: "text", text: msg.content });
      }
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          const tcObj = tc as Record<string, unknown>;
          const fn = (tcObj.function as Record<string, unknown>) ?? {};
          const tcId = typeof tcObj.id === "string" ? tcObj.id : crypto.randomUUID();
          const name = typeof fn.name === "string" ? fn.name : "unknown";
          const argsStr = typeof fn.arguments === "string" ? fn.arguments : "{}";
          let args: unknown = {};
          try { args = JSON.parse(argsStr); } catch { /* keep empty */ }
          const result = toolResults.get(tcId);
          parts.push({
            type: "tool-call",
            toolCallId: tcId,
            toolName: name,
            args,
            argsText: argsStr,
            ...(result !== undefined ? { result } : {}),
          });
        }
      }
      content = parts;
    } else {
      const raw = msg.content;
      if (Array.isArray(raw)) {
        content = raw.flatMap((p): unknown[] => {
          const part = p as Record<string, unknown>;
          if (part.type === "text" && typeof part.text === "string") {
            return [{ type: "text", text: part.text }];
          }
          if (part.type === "image_url") {
            const iu = (part.image_url as Record<string, unknown>) ?? {};
            return [{ type: "image", image: typeof iu.url === "string" ? iu.url : "" }];
          }
          return [];
        });
      } else {
        content = typeof raw === "string" && raw.trim() ? [{ type: "text", text: raw }] : [];
      }
    }

    if (content.length === 0) continue;

    records.push({
      id,
      threadId,
      parentId: prevId,
      role: role as MessageRecord["role"],
      content: content as MessageRecord["content"],
      createdAt: baseTs + idx,
    });
    prevId = id;
    idx++;
  }

  return records;
}

function sharegptToRecords(
  conversations: unknown[],
  threadId: string,
  baseTs: number,
): MessageRecord[] {
  const records: MessageRecord[] = [];
  let prevId: string | null = null;
  let idx = 0;
  for (const c of conversations) {
    const conv = c as Record<string, unknown>;
    const from = typeof conv.from === "string" ? conv.from : "";
    const value = typeof conv.value === "string" ? conv.value : "";
    if (!value.trim()) continue;
    const role: MessageRecord["role"] = from === "human" ? "user" : from === "system" ? "system" : "assistant";
    const id = crypto.randomUUID();
    records.push({
      id,
      threadId,
      parentId: prevId,
      role,
      content: [{ type: "text", text: value }] as MessageRecord["content"],
      createdAt: baseTs + idx,
    });
    prevId = id;
    idx++;
  }
  return records;
}

function csvToRecords(csvText: string, threadId: string, baseTs: number): MessageRecord[] {
  // parseCsv handles quoted newlines, so multi-line message content
  // round-trips; a naive per-line split would break those records.
  const rows = parseCsv(csvText).slice(1);
  const records: MessageRecord[] = [];
  let prevId: string | null = null;
  let idx = 0;
  for (const row of rows) {
    if (row.length < 2) continue;
    const role = row[0].trim();
    const content = row.slice(1).join(",");
    if (!content.trim()) continue;
    const validRole = role === "user" || role === "assistant" || role === "system" ? role : "user";
    const id = crypto.randomUUID();
    records.push({
      id,
      threadId,
      parentId: prevId,
      role: validRole as MessageRecord["role"],
      content: [{ type: "text", text: content }] as MessageRecord["content"],
      createdAt: baseTs + idx,
    });
    prevId = id;
    idx++;
  }
  return records;
}

interface ParsedConversation {
  title: string;
  threadId: string;
  messages: MessageRecord[];
}

function parseImportText(text: string, filename: string): ParsedConversation[] {
  const results: ParsedConversation[] = [];
  const basename = filename.replace(/\.[^.]+$/, "");

  const isJsonl = /\.(jsonl|ndjson)$/i.test(filename);
  const isCsv = /\.csv$/i.test(filename);

  if (isCsv) {
    const threadId = crypto.randomUUID();
    const messages = csvToRecords(text, threadId, Date.now());
    if (messages.length > 0) {
      results.push({ title: basename, threadId, messages });
    }
    return results;
  }

  if (isJsonl) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    lines.forEach((line, lineIdx) => {
      let obj: Record<string, unknown>;
      try { obj = JSON.parse(line); } catch { return; }

      // Fresh ID: reusing the exported thread_id would clobber an existing
      // thread on import.
      const threadId = crypto.randomUUID();
      const title = typeof obj.title === "string" ? obj.title : `${basename} ${lineIdx + 1}`;
      const baseTs = typeof obj.created_at === "number" ? obj.created_at : Date.now() + lineIdx;

      let messages: MessageRecord[] = [];

      if (Array.isArray(obj.messages)) {
        messages = oaiMessagesToRecords(obj.messages, threadId, baseTs);
      } else if (Array.isArray(obj.conversations)) {
        messages = sharegptToRecords(obj.conversations, threadId, baseTs);
      }

      if (messages.length > 0) {
        results.push({ title, threadId, messages });
      }
    });
    return results;
  }

  // Unknown extension: retry as JSONL.
  return parseImportText(text, filename + ".jsonl");
}

export async function importConversationsFromFile(
  file: File,
  projectId: string | null = null,
): Promise<number> {
  const text = await file.text();
  const parsed = parseImportText(text, file.name);
  if (parsed.length === 0) return 0;

  const now = Date.now();
  await Promise.all(
    parsed.map(async ({ title, threadId, messages }) => {
      const thread: ThreadRecord = {
        id: threadId,
        title,
        modelType: "base",
        projectId: projectId ?? null,
        archived: false,
        createdAt: messages[0]?.createdAt ?? now,
      };
      await saveStoredChatThread(thread);
      await syncStoredChatMessages(threadId, messages, { pruneMissing: false });
    }),
  );

  notifyChatHistoryUpdated();
  return parsed.length;
}

// ShareGPT training exports: prompt → one record (human turn + empty gpt slot);
// list → one multi-turn record, each item a human turn.
function exportPromptTrainingJsonl(entry: PromptEntry): Promise<void> {
  const record = {
    conversations: [
      { from: "human", value: entry.text },
      { from: "gpt", value: "" },
    ],
  };
  return downloadBlob(
    JSON.stringify(record),
    `${sanitizeFilename(entry.name)}-training.jsonl`,
    "application/x-ndjson",
  );
}

function exportPromptsTrainingJsonl(entries: PromptEntry[]): Promise<void> {
  const lines = entries
    .map((e) =>
      JSON.stringify({
        conversations: [
          { from: "human", value: e.text },
          { from: "gpt", value: "" },
        ],
      }),
    )
    .join("\n");
  return downloadBlob(lines, "prompts-training.jsonl", "application/x-ndjson");
}

function exportListTrainingJsonl(entry: PromptListEntry): Promise<void> {
  const conversations = entry.items.flatMap((text) => [
    { from: "human", value: text },
    { from: "gpt", value: "" },
  ]);
  return downloadBlob(
    JSON.stringify({ conversations }),
    `${sanitizeFilename(entry.name)}-training.jsonl`,
    "application/x-ndjson",
  );
}

function exportListsTrainingJsonl(entries: PromptListEntry[]): Promise<void> {
  const lines = entries
    .map((e) => {
      const conversations = e.items.flatMap((text) => [
        { from: "human", value: text },
        { from: "gpt", value: "" },
      ]);
      return JSON.stringify({ conversations });
    })
    .join("\n");
  return downloadBlob(lines, "prompt-lists-training.jsonl", "application/x-ndjson");
}

// RFC 4180 CSV parser: handles quoted fields with embedded newlines/commas.
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let i = 0;

  function finishRow() {
    rows.push(row);
    row = [];
  }

  while (i < text.length) {
    if (text[i] === '"') {
      i++;
      let cell = "";
      while (i < text.length) {
        if (text[i] === '"') {
          if (text[i + 1] === '"') {
            cell += '"';
            i += 2;
          } else {
            i++;
            break;
          }
        } else {
          cell += text[i++];
        }
      }
      row.push(cell);
      if (text[i] === ",") { i++; }
      else if (text[i] === "\r") { i++; if (text[i] === "\n") i++; finishRow(); }
      else if (text[i] === "\n") { i++; finishRow(); }
    } else if (text[i] === "\r") {
      i++;
      if (text[i] === "\n") i++;
      row.push("");
      finishRow();
    } else if (text[i] === "\n") {
      i++;
      row.push("");
      finishRow();
    } else {
      let cell = "";
      while (i < text.length && text[i] !== "," && text[i] !== "\r" && text[i] !== "\n") {
        cell += text[i++];
      }
      row.push(cell);
      if (text[i] === ",") { i++; }
      else if (text[i] === "\r") { i++; if (text[i] === "\n") i++; finishRow(); }
      else if (text[i] === "\n") { i++; finishRow(); }
    }
  }
  if (row.length > 0) rows.push(row);
  return rows;
}

async function importPromptsFromText(text: string, isCsv: boolean): Promise<{ count: number; skipped: number }> {
  const entries: PromptEntry[] = [];
  let skipped = 0;
  if (isCsv) {
    const rows = parseCsv(text).slice(1);
    for (const cells of rows) {
      const name = cells[0]?.trim();
      const promptText = cells[1]?.trim();
      if (promptText) {
        entries.push({
          id: newId(),
          name: name || "Imported",
          text: promptText,
          createdAt: now(),
          updatedAt: now(),
        });
      }
    }
  } else {
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      try {
        const obj = JSON.parse(line) as Record<string, unknown>;
        if (typeof obj.text === "string" && obj.text.trim()) {
          entries.push({
            id: newId(),
            name: typeof obj.name === "string" ? obj.name || "Imported" : "Imported",
            text: obj.text.trim(),
            createdAt: now(),
            updatedAt: now(),
          });
        } else {
          skipped++;
        }
      } catch {
        skipped++;
      }
    }
  }
  if (entries.length > 0) await bulkSavePromptEntries(entries);
  return { count: entries.length, skipped };
}

async function importListsFromText(text: string, isCsv: boolean): Promise<{ count: number; skipped: number }> {
  const lists: PromptListEntry[] = [];
  let skipped = 0;
  if (isCsv) {
    const rows = parseCsv(text).slice(1);
    const listMap = new Map<string, Array<{ order: number; text: string }>>();
    for (const cells of rows) {
      const listName = cells[0]?.trim();
      const promptText = cells[2]?.trim();
      if (listName && promptText) {
        const order = parseInt(cells[1] ?? "0", 10) || 0;
        if (!listMap.has(listName)) listMap.set(listName, []);
        listMap.get(listName)!.push({ order, text: promptText });
      }
    }
    for (const [listName, items] of listMap.entries()) {
      const sorted = items.sort((a, b) => a.order - b.order).map((x) => x.text);
      if (sorted.length > 0) {
        lists.push({
          id: newId(),
          name: listName,
          items: sorted,
          createdAt: now(),
          updatedAt: now(),
        });
      }
    }
  } else {
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      try {
        const obj = JSON.parse(line) as Record<string, unknown>;
        if (Array.isArray(obj.items) && obj.items.length > 0) {
          const items = (obj.items as unknown[]).filter(
            (x): x is string => typeof x === "string" && x.trim().length > 0,
          );
          if (items.length > 0) {
            lists.push({
              id: newId(),
              name: typeof obj.name === "string" ? obj.name || "Imported" : "Imported",
              items,
              createdAt: now(),
              updatedAt: now(),
            });
          } else {
            skipped++;
          }
        } else {
          skipped++;
        }
      } catch {
        skipped++;
      }
    }
  }
  if (lists.length > 0) await bulkSavePromptLists(lists);
  return { count: lists.length, skipped };
}

async function importCollectionFromText(text: string): Promise<{ prompts: number; lists: number }> {
  const entries: PromptEntry[] = [];
  const listEntries: PromptListEntry[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    try {
      const obj = JSON.parse(line) as Record<string, unknown>;
      if (obj.type === "prompt" && typeof obj.text === "string" && obj.text.trim()) {
        entries.push({
          id: newId(),
          name: typeof obj.name === "string" ? obj.name || "Imported" : "Imported",
          text: obj.text.trim(),
          createdAt: now(),
          updatedAt: now(),
        });
      } else if (obj.type === "prompt_list" && Array.isArray(obj.items) && obj.items.length > 0) {
        const items = (obj.items as unknown[]).filter(
          (x): x is string => typeof x === "string" && x.trim().length > 0,
        );
        if (items.length > 0) {
          listEntries.push({
            id: newId(),
            name: typeof obj.name === "string" ? obj.name || "Imported" : "Imported",
            items,
            createdAt: now(),
            updatedAt: now(),
          });
        }
      }
    } catch {
      /* */
    }
  }
  if (entries.length > 0) await bulkSavePromptEntries(entries);
  if (listEntries.length > 0) await bulkSavePromptLists(listEntries);
  return { prompts: entries.length, lists: listEntries.length };
}

type ExportScope = "single" | "training";
type ExportFormat = "jsonl" | "csv";

type ExportModalCtx =
  | { kind: "prompt"; entry: PromptEntry }
  | { kind: "list"; entry: PromptListEntry }
  | { kind: "bulk"; tab: Tab; prompts: PromptEntry[]; lists: PromptListEntry[] };

function ExportModal({
  ctx,
  onClose,
}: {
  ctx: ExportModalCtx;
  onClose: () => void;
}): ReactElement {
  const [scope, setScope] = useState<ExportScope>("single");
  const [format, setFormat] = useState<ExportFormat>("jsonl");

  const csvAvailable = scope === "single";

  useEffect(() => {
    if (!csvAvailable) setFormat("jsonl");
  }, [csvAvailable]);

  const handleExport = useCallback(async () => {
    try {
      if (ctx.kind === "prompt") {
        if (scope === "training") await exportPromptTrainingJsonl(ctx.entry);
        else if (format === "csv") await exportPromptCsv(ctx.entry);
        else await exportPromptJsonl(ctx.entry);
      } else if (ctx.kind === "list") {
        if (scope === "training") await exportListTrainingJsonl(ctx.entry);
        else if (format === "csv") await exportListCsv(ctx.entry);
        else await exportListJsonl(ctx.entry);
      } else {
        const { tab, prompts, lists } = ctx;
        if (scope === "training") {
          if (tab === "prompts") {
            if (prompts.length === 0) { toast.info("No prompts to export"); return; }
            await exportPromptsTrainingJsonl(prompts);
          } else {
            if (lists.length === 0) { toast.info("No prompt lists to export"); return; }
            await exportListsTrainingJsonl(lists);
          }
        } else if (tab === "prompts") {
          if (prompts.length === 0) { toast.info("No prompts to export"); return; }
          if (format === "csv") await exportAllPromptsCsv(prompts);
          else await exportAllPromptsJsonl(prompts);
        } else {
          if (lists.length === 0) { toast.info("No prompt lists to export"); return; }
          if (format === "csv") await exportAllListsCsv(lists);
          else await exportAllListsJsonl(lists);
        }
      }
      onClose();
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error("Could not save export.", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }, [ctx, scope, format, onClose]);

  const singleLabel =
    ctx.kind === "prompt"
      ? "Single Prompt"
      : ctx.kind === "list"
        ? "Prompt List"
        : ctx.kind === "bulk" && ctx.tab === "prompts"
          ? "All Prompts"
          : "All Prompt Lists";

  const singleDesc =
    ctx.kind === "prompt"
      ? "Export raw prompt name and text"
      : ctx.kind === "list"
        ? "Export list name and all prompt items"
        : ctx.kind === "bulk" && ctx.tab === "prompts"
          ? "One JSONL or CSV record per saved prompt"
          : "One JSONL or CSV record per saved list";

  return (
    <Dialog open onOpenChange={onClose}>
      {/* */}
      <DialogContent className="sm:max-w-[520px] gap-0 p-0 overflow-hidden">
        <div className="flex flex-col gap-5 p-6">
          {/* */}
          <DialogTitle className="text-base font-semibold tracking-tight">Export</DialogTitle>
          <DialogDescription className="sr-only">Choose export type and format.</DialogDescription>

          {/* */}
          <div className="flex flex-col gap-2">
            <p className="text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground/60">
              Export as
            </p>
            <div className="flex flex-col gap-2">
              {/* */}
              <label
                className={cn(
                  "flex w-full cursor-pointer items-center gap-3 rounded-lg border px-4 py-3 transition-all",
                  scope === "single"
                    ? "border-ring-strong bg-primary/5"
                    : "border-border/60 hover:border-border hover:bg-muted/30",
                )}
              >
                <input
                  type="radio"
                  name="export-scope"
                  value="single"
                  checked={scope === "single"}
                  onChange={() => setScope("single")}
                  className="accent-primary shrink-0"
                />
                <div className="min-w-0">
                  <p className="text-sm font-semibold leading-none">{singleLabel}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{singleDesc}</p>
                </div>
              </label>

              {/* */}
              <label
                className={cn(
                  "flex w-full cursor-pointer items-start gap-3 rounded-lg border px-4 py-3 transition-all",
                  scope === "training"
                    ? "border-ring-strong bg-primary/5"
                    : "border-border/60 hover:border-border hover:bg-muted/30",
                )}
              >
                <input
                  type="radio"
                  name="export-scope"
                  value="training"
                  checked={scope === "training"}
                  onChange={() => setScope("training")}
                  className="mt-0.5 accent-primary shrink-0"
                />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold leading-none">Training Style</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    ShareGPT format for Unsloth fine-tuning
                  </p>
                  <code className="mt-2 block w-full truncate rounded-md bg-muted px-2 py-1 font-mono text-ui-10 text-muted-foreground/60">
                    {`{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":""}]}`}
                  </code>
                </div>
              </label>
            </div>
          </div>

          {/* */}
          <div className="flex flex-col gap-2">
            <p className="text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground/60">
              Format
            </p>
            <div className="flex items-center gap-1 self-start rounded-lg bg-muted/60 p-1">
              {(["jsonl", "csv"] as ExportFormat[]).map((f) => {
                const disabled = f === "csv" && !csvAvailable;
                return (
                  <label
                    key={f}
                    className={cn(
                      "select-none rounded-md px-6 py-1.5 text-xs font-semibold uppercase tracking-wide transition-all",
                      disabled
                        ? "cursor-not-allowed opacity-40 text-muted-foreground"
                        : "cursor-pointer",
                      format === f && !disabled
                        ? "bg-background text-foreground shadow-sm ring-1 ring-border/40"
                        : !disabled && "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    <input
                      type="radio"
                      name="export-format"
                      value={f}
                      checked={format === f}
                      onChange={() => { if (!disabled) setFormat(f); }}
                      disabled={disabled}
                      className="sr-only"
                    />
                    {f.toUpperCase()}
                  </label>
                );
              })}
            </div>
            {!csvAvailable && (
              <p className="text-xs text-muted-foreground/60">
                CSV is not available for this export type
              </p>
            )}
          </div>
        </div>

        {/* */}
        <div className="flex items-center justify-end gap-2 border-t border-border/50 px-6 py-4">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" onClick={handleExport}>
            <DownloadIcon className="mr-1.5 size-3.5" />
            Download
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function PromptCard({
  entry,
  onUse,
  onExport,
  onRefresh,
}: {
  entry: PromptEntry;
  onUse: (text: string) => void;
  onExport: (entry: PromptEntry) => void;
  onRefresh: () => void;
}): ReactElement {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(entry.name);
  const [text, setText] = useState(entry.text);
  const pinnedPromptIds = usePlusMenuPrefsStore((s) => s.pinnedPromptIds);
  const togglePinnedPrompt = usePlusMenuPrefsStore((s) => s.togglePinnedPrompt);
  const isPinned = pinnedPromptIds.includes(entry.id);

  const handleSave = useCallback(async () => {
    const trimName = name.trim();
    const trimText = text.trim();
    if (!trimText) return;
    await savePromptEntry({ ...entry, name: trimName || "Untitled Prompt", text: trimText, updatedAt: now() });
    setEditing(false);
    onRefresh();
  }, [entry, name, text, onRefresh]);

  const handleDelete = useCallback(async () => {
    await deletePromptEntry(entry.id);
    onRefresh();
  }, [entry.id, onRefresh]);

  if (editing) {
    return (
      <div className="rounded-xl border border-border/50 bg-muted/30 p-4 flex flex-col gap-3">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Prompt name..."
          className="w-full rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Prompt text..."
          className="w-full resize-y rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow leading-relaxed"
        />
        <div className="flex gap-2 justify-end">
          <Button size="sm" variant="ghost" onClick={() => { setName(entry.name); setText(entry.text); setEditing(false); }}>
            <XIcon className="size-3.5 mr-1" />Cancel
          </Button>
          <Button size="sm" onClick={handleSave}>
            <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 mr-1" />Save
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="group rounded-xl border border-border/60 bg-card p-4 flex flex-col gap-2 hover:border-border hover:shadow-sm transition-all">
      <div className="flex items-center gap-2">
        {isPinned ? (
          <BookmarkIcon className="size-3.5 shrink-0 fill-primary text-primary" />
        ) : null}
        <span className="font-semibold text-sm flex-1 truncate tracking-tight">{entry.name}</span>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            type="button"
            onClick={() => onUse(entry.text)}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-2.5 py-1 text-xs font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
            title="Load into composer"
          >
            <PlayIcon className="size-3" />Use
          </button>
          <div className="mx-1 h-4 w-px bg-border/60" />
          <button
            type="button"
            onClick={() => togglePinnedPrompt(entry.id)}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
              isPinned
                ? "text-primary hover:bg-primary/10"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
            title={isPinned ? "Unpin from + menu" : "Pin to + menu"}
          >
            <BookmarkIcon
              className={cn("size-3.5", isPinned && "fill-primary")}
            />
          </button>
          <button
            type="button"
            onClick={() => onExport(entry)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title="Export"
          >
            <DownloadIcon className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title="Edit"
          >
            <PencilIcon className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={handleDelete}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
            title="Delete"
          >
            <Trash2Icon className="size-3.5" />
          </button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground line-clamp-3 leading-relaxed">{entry.text}</p>
    </div>
  );
}

function NewPromptForm({ onClose, onRefresh }: { onClose: () => void; onRefresh: () => void }): ReactElement {
  const [name, setName] = useState("");
  const [text, setText] = useState("");

  const handleSave = useCallback(async () => {
    const trimText = text.trim();
    if (!trimText) return;
    const ts = now();
    await savePromptEntry({
      id: newId(),
      name: name.trim() || "Untitled Prompt",
      text: trimText,
      createdAt: ts,
      updatedAt: ts,
    });
    onRefresh();
    onClose();
  }, [name, text, onClose, onRefresh]);

  return (
    <div className="rounded-xl border border-border/50 bg-muted/30 p-4 flex flex-col gap-3">
      <p className="text-xs font-semibold text-muted-foreground">New Prompt</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Prompt name (optional)..."
        autoFocus
        className="w-full rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
      />
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        placeholder="Write your prompt here..."
        className="w-full resize-y rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow leading-relaxed"
      />
      <div className="flex gap-2 justify-end">
        <Button size="sm" variant="ghost" onClick={onClose}>
          <XIcon className="size-3.5 mr-1" />Cancel
        </Button>
        <Button size="sm" onClick={handleSave} disabled={!text.trim()}>
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 mr-1" />Save Prompt
        </Button>
      </div>
    </div>
  );
}

function PromptListCard({
  entry,
  onRunList,
  onExport,
  onRefresh,
}: {
  entry: PromptListEntry;
  onRunList?: (items: string[]) => void;
  onExport: (entry: PromptListEntry) => void;
  onRefresh: () => void;
}): ReactElement {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(entry.name);
  const [items, setItems] = useState<string[]>(entry.items);

  const handleSave = useCallback(async () => {
    const filtered = items.filter((t) => t.trim());
    if (filtered.length === 0) return;
    await savePromptList({ ...entry, name: name.trim() || "Untitled List", items: filtered, updatedAt: now() });
    setEditing(false);
    onRefresh();
  }, [entry, name, items, onRefresh]);

  const handleDelete = useCallback(async () => {
    await deletePromptList(entry.id);
    onRefresh();
  }, [entry.id, onRefresh]);

  const addItem = useCallback(() => setItems((prev) => [...prev, ""]), []);
  const removeItem = useCallback(
    (i: number) => setItems((prev) => prev.filter((_, idx) => idx !== i)),
    [],
  );
  const updateItem = useCallback(
    (i: number, val: string) =>
      setItems((prev) => prev.map((v, idx) => (idx === i ? val : v))),
    [],
  );

  if (editing) {
    return (
      <div className="rounded-xl border border-border/50 bg-muted/30 p-4 flex flex-col gap-3">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="List name..."
          className="w-full rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
        />
        <p className="text-xs font-semibold text-muted-foreground">Prompts (sent in order)</p>
        <div className="flex flex-col gap-2">
          {items.map((item, i) => (
            <div key={i} className="flex items-start gap-2">
              <GripVerticalIcon className="size-4 mt-2.5 shrink-0 text-muted-foreground/30 cursor-grab" />
              <span className="text-xs font-medium text-muted-foreground/60 mt-2.5 w-5 shrink-0 text-right">{i + 1}.</span>
              <textarea
                value={item}
                onChange={(e) => updateItem(i, e.target.value)}
                rows={2}
                placeholder={`Prompt ${i + 1}...`}
                className="flex-1 resize-y rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow leading-relaxed"
              />
              <button
                type="button"
                onClick={() => removeItem(i)}
                className="flex h-7 w-7 mt-1 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors shrink-0"
                title="Remove"
              >
                <XIcon className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
        <button
          type="button"
          onClick={addItem}
          className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
        >
          <PlusIcon className="size-3.5" />Add prompt
        </button>
        <div className="flex gap-2 justify-end">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => { setName(entry.name); setItems(entry.items); setEditing(false); }}
          >
            <XIcon className="size-3.5 mr-1" />Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={items.filter((t) => t.trim()).length === 0}
          >
            <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 mr-1" />Save List
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="group rounded-xl border border-border/60 bg-card p-4 flex flex-col gap-2.5 hover:border-border hover:shadow-sm transition-all">
      <div className="flex items-center gap-2">
        <span className="font-semibold text-sm flex-1 truncate tracking-tight">{entry.name}</span>
        <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-ui-11 font-medium text-muted-foreground">
          {entry.items.length}
        </span>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {onRunList && (
            <button
              type="button"
              onClick={() => onRunList(entry.items)}
              className="flex h-7 items-center gap-1.5 rounded-lg px-2.5 text-xs font-semibold text-primary hover:bg-primary/10 transition-colors"
              title="Run list"
            >
              <PlayIcon className="size-3" />Run
            </button>
          )}
          <div className="mx-1 h-4 w-px bg-border/60" />
          <button
            type="button"
            onClick={() => onExport(entry)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title="Export"
          >
            <DownloadIcon className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title="Edit"
          >
            <PencilIcon className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={handleDelete}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
            title="Delete"
          >
            <Trash2Icon className="size-3.5" />
          </button>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        {entry.items.slice(0, 3).map((item, i) => (
          <p key={i} className="text-xs text-muted-foreground flex gap-2 leading-relaxed">
            <span className="text-muted-foreground/40 shrink-0 tabular-nums">{i + 1}.</span>
            <span className="line-clamp-1">{item}</span>
          </p>
        ))}
        {entry.items.length > 3 && (
          <p className="text-ui-11 text-muted-foreground/50 ml-5">
            +{entry.items.length - 3} more
          </p>
        )}
      </div>
    </div>
  );
}

function NewPromptListForm({ onClose, onRefresh }: { onClose: () => void; onRefresh: () => void }): ReactElement {
  const [name, setName] = useState("");
  const [items, setItems] = useState<string[]>(["", ""]);

  const handleSave = useCallback(async () => {
    const filtered = items.filter((t) => t.trim());
    if (filtered.length === 0) return;
    const ts = now();
    await savePromptList({
      id: newId(),
      name: name.trim() || "Untitled List",
      items: filtered,
      createdAt: ts,
      updatedAt: ts,
    });
    onRefresh();
    onClose();
  }, [name, items, onClose, onRefresh]);

  const addItem = useCallback(() => setItems((prev) => [...prev, ""]), []);
  const removeItem = useCallback(
    (i: number) => setItems((prev) => prev.filter((_, idx) => idx !== i)),
    [],
  );
  const updateItem = useCallback(
    (i: number, val: string) =>
      setItems((prev) => prev.map((v, idx) => (idx === i ? val : v))),
    [],
  );

  return (
    <div className="rounded-xl border border-border/50 bg-muted/30 p-4 flex flex-col gap-3">
      <p className="text-xs font-semibold text-muted-foreground">New Prompt List</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="List name..."
        autoFocus
        className="w-full rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
      />
      <p className="text-xs font-semibold text-muted-foreground">
        Prompts — loaded into the composer one at a time
      </p>
      <div className="flex flex-col gap-2">
        {items.map((item, i) => (
          <div key={i} className="flex items-start gap-2">
            <span className="text-xs font-medium text-muted-foreground/60 mt-2.5 w-5 shrink-0 text-right">{i + 1}.</span>
            <textarea
              value={item}
              onChange={(e) => updateItem(i, e.target.value)}
              rows={2}
              placeholder={`Prompt ${i + 1}...`}
              className="flex-1 resize-y rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow leading-relaxed"
            />
            <button
              type="button"
              onClick={() => removeItem(i)}
              disabled={items.length <= 1}
              className="flex h-7 w-7 mt-1 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors shrink-0 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Remove"
            >
              <XIcon className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={addItem}
        className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
      >
        <PlusIcon className="size-3.5" />Add another prompt
      </button>
      <div className="flex gap-2 justify-end">
        <Button size="sm" variant="ghost" onClick={onClose}>
          <XIcon className="size-3.5 mr-1" />Cancel
        </Button>
        <Button
          size="sm"
          onClick={handleSave}
          disabled={items.filter((t) => t.trim()).length === 0}
        >
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 mr-1" />Save Prompt List
        </Button>
      </div>
    </div>
  );
}

type Tab = "prompts" | "lists";

export function PromptStorageDialog({
  open,
  onOpenChange,
  onUse,
  onRunList,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUse: (text: string) => void;
  onRunList?: (items: string[]) => void;
}): ReactElement {
  const [activeTab, setActiveTab] = useState<Tab>("prompts");
  const [showNewPrompt, setShowNewPrompt] = useState(false);
  const [showNewList, setShowNewList] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [exportCtx, setExportCtx] = useState<ExportModalCtx | null>(null);
  const importRef = useRef<HTMLInputElement>(null);

  const [promptEntries, setPromptEntries] = useState<PromptEntry[]>([]);
  const [promptLists, setPromptLists] = useState<PromptListEntry[]>([]);

  const refreshEntries = useCallback(async () => {
    try { setPromptEntries(await listPromptEntries()); } catch {}
  }, []);
  const refreshLists = useCallback(async () => {
    try { setPromptLists(await listPromptLists()); } catch {}
  }, []);

  useEffect(() => {
    if (open) {
      void refreshEntries();
      void refreshLists();
    }
  }, [open, refreshEntries, refreshLists]);

  useEffect(() => {
    setSearchQuery("");
    setShowSuggestions(false);
    setShowNewPrompt(false);
    setShowNewList(false);
  }, [activeTab]);

  const filteredPrompts = useMemo(() => {
    const all = promptEntries ?? [];
    if (!searchQuery.trim()) return all;
    const q = searchQuery.toLowerCase();
    return all.filter(
      (e) => e.name.toLowerCase().includes(q) || e.text.toLowerCase().includes(q),
    );
  }, [promptEntries, searchQuery]);

  const filteredLists = useMemo(() => {
    const all = promptLists ?? [];
    if (!searchQuery.trim()) return all;
    const q = searchQuery.toLowerCase();
    return all.filter((e) => e.name.toLowerCase().includes(q));
  }, [promptLists, searchQuery]);

  const suggestions = useMemo(() => {
    if (!searchQuery.trim()) return [];
    const q = searchQuery.toLowerCase();
    const source: { name: string }[] =
      activeTab === "prompts" ? (promptEntries ?? []) : (promptLists ?? []);
    return source
      .filter((e) => e.name.toLowerCase().includes(q))
      .slice(0, 7)
      .map((e) => e.name);
  }, [searchQuery, activeTab, promptEntries, promptLists]);

  const handleUsePrompt = useCallback(
    (text: string) => {
      onUse(text);
      onOpenChange(false);
    },
    [onUse, onOpenChange],
  );

  const handleImportFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const text = await file.text();
      const isCsv = file.name.toLowerCase().endsWith(".csv");
      try {
        // Collection JSONL: first line carries a "type" field.
        if (!isCsv) {
          const firstLine = text.split("\n").find((l) => l.trim());
          if (firstLine) {
            try {
              const probe = JSON.parse(firstLine) as Record<string, unknown>;
              if (probe.type === "prompt" || probe.type === "prompt_list") {
                const result = await importCollectionFromText(text);
                const total = result.prompts + result.lists;
                void refreshEntries();
                void refreshLists();
                if (total > 0) {
                  toast.success(`Imported ${total} item${total !== 1 ? "s" : ""}`, {
                    description: `${result.prompts} prompt${result.prompts !== 1 ? "s" : ""}, ${result.lists} list${result.lists !== 1 ? "s" : ""}`,
                  });
                } else {
                  toast.warning("No items imported", {
                    description: "The file may be empty or in an unsupported format.",
                  });
                }
                e.target.value = "";
                return;
              }
            } catch {
              /* */
            }
          }
        }

        let count = 0;
        let skipped = 0;
        if (activeTab === "prompts") {
          ({ count, skipped } = await importPromptsFromText(text, isCsv));
          void refreshEntries();
        } else {
          ({ count, skipped } = await importListsFromText(text, isCsv));
          void refreshLists();
        }
        if (count > 0) {
          toast.success(`Imported ${count} item${count !== 1 ? "s" : ""}`, {
            description: skipped > 0 ? `${skipped} line${skipped !== 1 ? "s" : ""} skipped (unrecognised format)` : undefined,
          });
        } else {
          toast.warning("No items imported", {
            description: skipped > 0
              ? `${skipped} line${skipped !== 1 ? "s" : ""} could not be parsed.`
              : "The file may be empty or in an unsupported format.",
          });
        }
      } catch {
        toast.error("Import failed", { description: "Could not parse the file." });
      }
      e.target.value = "";
    },
    [activeTab, refreshEntries, refreshLists],
  );

  const bulkExportDisabled =
    (activeTab === "prompts" ? (promptEntries?.length ?? 0) : (promptLists?.length ?? 0)) === 0;

  const openBulkExport = useCallback(() => {
    setExportCtx({
      kind: "bulk",
      tab: activeTab,
      prompts: promptEntries ?? [],
      lists: promptLists ?? [],
    });
  }, [activeTab, promptEntries, promptLists]);

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent showCloseButton={false} className="sm:max-w-[min(1100px,88vw)] max-h-[94vh] flex flex-col gap-0 p-0 overflow-hidden">
          {/* */}
          <DialogHeader className="px-6 pt-5 pb-4 shrink-0 border-b border-border/50">
            <div className="flex items-center gap-3">
              <div className="flex-1 min-w-0">
                <DialogTitle className="text-base font-semibold tracking-tight">
                  Prompt Storage
                </DialogTitle>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Save and reuse prompts across conversations
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <input
                  ref={importRef}
                  type="file"
                  accept=".jsonl,.json,.csv"
                  className="hidden"
                  onChange={handleImportFile}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => importRef.current?.click()}
                  className="h-8 gap-1.5 text-xs"
                  title="Import from JSONL, CSV, or collection JSONL"
                >
                  <UploadIcon className="size-3.5" />
                  Import
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={bulkExportDisabled}
                  onClick={openBulkExport}
                  className="h-8 gap-1.5 text-xs"
                >
                  <DownloadIcon className="size-3.5" />
                  Export
                </Button>
                <div className="ml-1 h-5 w-px bg-border/60 shrink-0" />
                <DialogClose asChild>
                  <Button variant="ghost" size="icon-sm">
                    <XIcon className="size-4" />
                    <span className="sr-only">Close</span>
                  </Button>
                </DialogClose>
              </div>
            </div>
            <DialogDescription className="sr-only">
              Save and manage reusable prompts and prompt lists.
            </DialogDescription>
          </DialogHeader>

          {/* */}
          <div className="px-6 pt-4 pb-3 shrink-0 flex flex-col gap-3">
            {/* */}
            <div className="flex items-center gap-1 self-start rounded-lg bg-muted/60 p-1">
              {(["prompts", "lists"] as Tab[]).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveTab(tab)}
                  className={cn(
                    "rounded-md px-4 py-1.5 text-xs font-medium transition-all",
                    activeTab === tab
                      ? "bg-background text-foreground shadow-sm ring-1 ring-border/40"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {tab === "prompts" ? "Saved Prompts" : "Prompt Lists"}
                </button>
              ))}
            </div>

            {/* */}
            <div className="relative">
              <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/60" />
              <input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setShowSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
                placeholder={`Search ${activeTab === "prompts" ? "prompts by name or text" : "prompt lists by name"}…`}
                className="w-full rounded-lg border-0 bg-muted/50 pl-9 pr-3 py-2 text-sm outline-none focus:ring-1 focus:ring-ring placeholder:text-muted-foreground/60 transition-shadow"
              />
              {showSuggestions && searchQuery.trim() !== "" && suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 z-50 mt-1 rounded-xl border border-border/60 bg-popover shadow-lg overflow-hidden">
                  {suggestions.map((name) => (
                    <button
                      key={name}
                      type="button"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => { setSearchQuery(name); setShowSuggestions(false); }}
                      className="flex w-full items-center gap-2.5 px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground transition-colors text-left"
                    >
                      <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-3 shrink-0 text-muted-foreground/60" />
                      <span className="truncate">{name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* */}
          <div className="flex-1 min-h-0 overflow-y-auto px-6 pb-6 flex flex-col gap-2.5">
            {activeTab === "prompts" && (
              <>
                {!showNewPrompt ? (
                  <button
                    type="button"
                    onClick={() => setShowNewPrompt(true)}
                    className="flex items-center gap-2.5 rounded-xl border-2 border-dashed border-border/40 px-4 py-3 text-sm font-medium text-muted-foreground hover:border-border hover:text-foreground hover:bg-muted/50 transition-all"
                  >
                    <PlusIcon className="size-4" />New Prompt
                  </button>
                ) : (
                  <NewPromptForm onClose={() => setShowNewPrompt(false)} onRefresh={refreshEntries} />
                )}

                {filteredPrompts.length > 0 ? (
                  filteredPrompts.map((entry) => (
                    <PromptCard
                      key={entry.id}
                      entry={entry}
                      onUse={handleUsePrompt}
                      onExport={(e) => setExportCtx({ kind: "prompt", entry: e })}
                      onRefresh={refreshEntries}
                    />
                  ))
                ) : (
                  !showNewPrompt && (
                    <div className="flex flex-col items-center justify-center py-16 text-center gap-3">
                      {searchQuery.trim() ? (
                        <>
                          <div className="flex size-12 items-center justify-center rounded-2xl bg-muted/60">
                            <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-5 text-muted-foreground/40" />
                          </div>
                          <div className="flex flex-col gap-1">
                            <p className="text-sm font-medium text-muted-foreground">
                              No prompts match &ldquo;{searchQuery}&rdquo;
                            </p>
                            <button
                              type="button"
                              onClick={() => setSearchQuery("")}
                              className="text-xs text-primary hover:underline"
                            >
                              Clear search
                            </button>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex size-12 items-center justify-center rounded-2xl bg-muted/60">
                            <BookmarkIcon className="size-5 text-muted-foreground/40" />
                          </div>
                          <div className="flex flex-col gap-1">
                            <p className="text-sm font-medium text-muted-foreground">No saved prompts yet</p>
                            <p className="text-xs text-muted-foreground/60">
                              Save prompts you use often for quick reuse
                            </p>
                          </div>
                        </>
                      )}
                    </div>
                  )
                )}
              </>
            )}

            {activeTab === "lists" && (
              <>
                {!showNewList ? (
                  <button
                    type="button"
                    onClick={() => setShowNewList(true)}
                    className="flex items-center gap-2.5 rounded-xl border-2 border-dashed border-border/40 px-4 py-3 text-sm font-medium text-muted-foreground hover:border-border hover:text-foreground hover:bg-muted/50 transition-all"
                  >
                    <PlusIcon className="size-4" />New Prompt List
                  </button>
                ) : (
                  <NewPromptListForm onClose={() => setShowNewList(false)} onRefresh={refreshLists} />
                )}

                {filteredLists.length > 0 ? (
                  filteredLists.map((entry) => (
                    <PromptListCard
                      key={entry.id}
                      entry={entry}
                      onRunList={onRunList}
                      onExport={(e) => setExportCtx({ kind: "list", entry: e })}
                      onRefresh={refreshLists}
                    />
                  ))
                ) : (
                  !showNewList && (
                    <div className="flex flex-col items-center justify-center py-16 text-center gap-3">
                      {searchQuery.trim() ? (
                        <>
                          <div className="flex size-12 items-center justify-center rounded-2xl bg-muted/60">
                            <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-5 text-muted-foreground/40" />
                          </div>
                          <div className="flex flex-col gap-1">
                            <p className="text-sm font-medium text-muted-foreground">
                              No prompt lists match &ldquo;{searchQuery}&rdquo;
                            </p>
                            <button
                              type="button"
                              onClick={() => setSearchQuery("")}
                              className="text-xs text-primary hover:underline"
                            >
                              Clear search
                            </button>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex size-12 items-center justify-center rounded-2xl bg-muted/60">
                            <LayoutListIcon className="size-5 text-muted-foreground/40" />
                          </div>
                          <div className="flex flex-col gap-1">
                            <p className="text-sm font-medium text-muted-foreground">No prompt lists yet</p>
                            <p className="text-xs text-muted-foreground/60">
                              A prompt list queues a sequence of prompts for quick reuse
                            </p>
                          </div>
                        </>
                      )}
                    </div>
                  )
                )}
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {exportCtx && (
        <ExportModal ctx={exportCtx} onClose={() => setExportCtx(null)} />
      )}
    </>
  );
}
