// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
  EyeIcon,
  LayoutListIcon,
  PencilIcon,
  PlayIcon,
  PlusIcon,
  RotateCcwIcon,
  Trash2Icon,
  UploadIcon,
  XIcon,
} from "lucide-react";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { SortablePromptItems } from "./sortable-prompt-items";
import {
  acquire,
  lockKey,
  release,
  sameListDraft,
  samePromptDraft,
} from "./mutation-lock";
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
import { toolResultModelText } from "../api/chat-adapter";
import { toolCallReplayArguments } from "../tool-call-arguments";
import { usePlusMenuPrefsStore } from "../stores/plus-menu-prefs-store";
import type { ThreadRecord, MessageRecord } from "../types";
import {
  buildNamedConversationsMarkdown,
  createConversationMarkdownBuilder,
  createConversationMarkdownExporter,
} from "../utils/conversation-markdown-export";
import { parseCsv } from "../utils/csv-parse";
import {
  canMergeConversationExport,
  conversationJsonlBody,
  exportFormatIncludesSiblings,
  ndjsonBody,
  type ConversationJsonlLayout,
} from "../utils/ndjson";
import { orderByParentChain } from "../utils/message-order";
import { unwrapPastedTextContent } from "../utils/pasted-text.ts";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "../utils/conversation-markdown";
import { planChatItemSources } from "../utils/project-source-plan";
import { stripSearchImageTokens } from "../search-images/search-images.ts";
import { saveMarkdownAsProjectSource } from "@/features/rag";

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
    ndjsonBody([JSON.stringify({ name: entry.name, text: entry.text })]),
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
  const lines = entries.map((e) => JSON.stringify({ name: e.name, text: e.text }));
  return downloadBlob(ndjsonBody(lines), "prompts.jsonl", "application/x-ndjson");
}

function exportAllPromptsCsv(entries: PromptEntry[]): Promise<void> {
  const rows = entries.map((e) => `${csvEscape(e.name)},${csvEscape(e.text)}`).join("\n");
  return downloadBlob(`name,text\n${rows}`, "prompts.csv", "text/csv");
}

function exportListJsonl(entry: PromptListEntry): Promise<void> {
  return downloadBlob(
    ndjsonBody([JSON.stringify({ name: entry.name, items: entry.items })]),
    `${sanitizeFilename(entry.name)}.jsonl`,
    "application/x-ndjson",
  );
}

function exportAllListsJsonl(entries: PromptListEntry[]): Promise<void> {
  const lines = entries.map((e) => JSON.stringify({ name: e.name, items: e.items }));
  return downloadBlob(ndjsonBody(lines), "prompt-lists.jsonl", "application/x-ndjson");
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
        // Keep base64 image payloads and sandbox card metadata out of every export format: use the
        // model-visible text, matching chat replay.
        const result = toolResultModelText(
          p.result,
          typeof p.toolName === "string" ? p.toolName : undefined,
        );
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

async function loadConversationMessages(
  threadId: string,
  options: {
    emptyMessage?: string;
    includeSiblings?: boolean;
  } = {},
) {
  const {
    emptyMessage = "No messages in this conversation to export.",
    includeSiblings = true,
  } = options;
  const raw = await listStoredChatMessages(threadId);
  if (raw.length === 0) {
    toast.info(emptyMessage);
    return null;
  }
  // No parentId = legacy flat thread (already DB createdAt-sorted); walking the chain would invert
  // order, so keep raw order.
  const hasParentIds = raw.some((m) => (m as { parentId?: unknown }).parentId != null);
  if (!hasParentIds) return raw;
  return orderByParentChain(raw, { includeSiblings }) as typeof raw;
}

function exportTs(): string {
  return new Date().toISOString().slice(0, 19).replace(/:/g, "-");
}

// Attachments live in msg.attachments[].content, not msg.content, so flatten both here or they'd be dropped on export.
function messageToText(msg: { content: unknown; attachments?: unknown }): string {
  const parts: string[] = [];
  const main = contentBlocksToText(msg.content);
  if (main) parts.push(main);
  if (Array.isArray(msg.attachments)) {
    for (const attachment of msg.attachments as Array<{ content?: unknown }>) {
      if (!attachment?.content) continue;
      // A paste carries a wrapper the same text never had when it fitted inline, so strip it rather
      // than exporting the marker.
      const attText = unwrapPastedTextContent(
        contentBlocksToText(attachment.content),
      );
      if (attText) parts.push(attText);
    }
  }
  return parts.join("\n\n");
}

// Markdown counterpart to messageToText: same content and attachments, but each part keeps its
// shape so the renderer can fence tool calls and collapse thinking.
function messageToMarkdown(msg: { content: unknown; attachments?: unknown }): string {
  const normalizeToolResult = toolResultModelText;
  const blocks = contentBlocksToMarkdownBlocks(msg.content, normalizeToolResult);
  if (Array.isArray(msg.attachments)) {
    for (const attachment of msg.attachments as Array<{ content?: unknown }>) {
      if (!attachment?.content) continue;
      blocks.push(
        ...contentBlocksToMarkdownBlocks(
          attachment.content,
          normalizeToolResult,
        ).map((block) =>
          block.kind === "text"
            ? { ...block, text: unwrapPastedTextContent(block.text) }
            : block,
        ),
      );
    }
  }
  return renderConversationBlocks(blocks);
}

// OpenAI messages array (tool-calling + multimodal fine-tuning): tool calls to "tool_calls" plus
// separate "role":"tool" messages; images to "image_url" parts; audio dropped; thinking kept
// as a text part.

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
      if (!Array.isArray(att.content)) return [];
      // Attachment text only: a message body is verbatim, and the paste wrapper is not something the user wrote.
      return (att.content as Record<string, unknown>[]).map((part) =>
        part?.type === "text" && typeof part.text === "string"
          ? { ...part, text: unwrapPastedTextContent(part.text) }
          : part,
      );
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
      } else if (p.type === "image" && typeof p.image === "string" && p.image) {
        textParts.push("[image attachment]");
      } else if (p.type === "tool-call") {
        const id = typeof p.toolCallId === "string" ? p.toolCallId : `call_${toolCalls.length}`;
        const name = typeof p.toolName === "string" ? p.toolName : "unknown";
        const argsStr = toolCallReplayArguments(
          typeof p.argsText === "string" ? p.argsText : undefined,
          p.args,
        );
        toolCalls.push({ id, type: "function", function: { name, arguments: argsStr } });
        if (p.result !== undefined && p.result !== null) {
          // Keep base64 image payloads out of exports: MCP image results carry their model-visible text
          // alongside the data, so serialize the text instead of the full object.
          const modelText = toolResultModelText(p.result, name);
          const resultStr =
            typeof modelText === "string" ? modelText : JSON.stringify(modelText);
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
    ndjsonBody([JSON.stringify({ conversations })]),
    "conversation-" + exportTs() + ".jsonl",
    "application/x-ndjson",
  );
}

// OpenAI/ChatML JSONL: {"messages": [{"role","content"}, ...]} per conversation; Unsloth reads
// this as a ChatML dataset.
export async function exportConversationRawJsonl(threadId: string): Promise<void> {
  return exportConversationJsonl(threadId, "training");
}

export async function exportConversationMessagesJsonl(threadId: string): Promise<void> {
  return exportConversationJsonl(threadId, "messages");
}

async function exportConversationJsonl(
  threadId: string,
  layout: ConversationJsonlLayout,
): Promise<void> {
  const messages = await loadConversationMessages(threadId, {
    includeSiblings: exportFormatIncludesSiblings(
      layout === "training" ? "jsonl-raw" : "jsonl-messages",
    ),
  });
  if (!messages) return;

  const oaiMsgs: OAIMessage[] = messages.flatMap((msg) => messageToOpenAI(msg));
  if (oaiMsgs.length === 0) { toast.info("No exportable content."); return; }
  await downloadBlob(
    ndjsonBody([conversationJsonlBody(oaiMsgs, layout)]),
    `conversation${layout === "messages" ? "-messages" : ""}-${exportTs()}.jsonl`,
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

/** Same markdown the download produces, for the "Copy as Markdown" shortcut. */
export const buildConversationMarkdownForThread =
  createConversationMarkdownBuilder({
    loadMessages: loadConversationMessages,
    renderMessage: messageToMarkdown,
  });

export const exportConversationMarkdown = createConversationMarkdownExporter({
  loadMessages: loadConversationMessages,
  renderMessage: messageToMarkdown,
  download: downloadBlob,
  exportTimestamp: exportTs,
  notifyNoContent: () => toast.info("No exportable content."),
});

// "skipped" is an empty conversation, which has already said so and must not stop the rest of a
// pair; "failed" has toasted a reason, so stop there rather than stack a second one.
type SaveSourceOutcome = "saved" | "skipped" | "failed";

async function saveConversationAsProjectSource(
  threadId: string,
  projectId: string,
  title: string,
): Promise<SaveSourceOutcome> {
  const messages = await loadConversationMessages(threadId, {
    emptyMessage: "No messages in this conversation to save.",
  });
  if (!messages) return "skipped";
  const markdown = buildConversationMarkdown(
    messages.map((msg) => ({
      role: String(msg.role ?? ""),
      // As the markdown exporter does: a project source is retrieved back into context, so the
      // renderer's tokens must not be saved as prose.
      content: stripSearchImageTokens(messageToMarkdown(msg)),
    })),
  );
  if (!markdown) {
    toast.info("No content to save.");
    return "skipped";
  }
  const saved = await saveMarkdownAsProjectSource(projectId, markdown, title, {
    quiet: true,
  });
  return saved ? "saved" : "failed";
}

export async function saveChatItemAsProjectSource(
  item: { id: string; title: string; type: string },
  projectId: string,
): Promise<void> {
  const plans = planChatItemSources(
    item,
    item.type === "single" ? [] : await listStoredChatThreads({ pairId: item.id }),
  );
  let saved = 0;
  for (const plan of plans) {
    const outcome = await saveConversationAsProjectSource(
      plan.id,
      projectId,
      plan.title,
    );
    if (outcome === "failed") break;
    if (outcome === "saved") saved += 1;
  }
  // One toast per click, not one per thread in the pair.
  if (saved === 1) toast.success("Saved to project sources.");
  else if (saved > 1) {
    toast.success(`Saved ${saved} chats to project sources.`);
  }
}

/** A sidebar row as one markdown document. The halves of a compare pair are named after their
 *  models, as saving them to project sources does: the two arrive in whichever order they
 *  last answered in, so position alone would label them wrong. */
export async function buildChatItemMarkdown(item: {
  id: string;
  title: string;
  type: string;
}): Promise<string> {
  const plans = planChatItemSources(
    item,
    item.type === "single" ? [] : await listStoredChatThreads({ pairId: item.id }),
  );
  return buildNamedConversationsMarkdown(
    plans,
    buildConversationMarkdownForThread,
  );
}

export type ConvExportFormat = "jsonl-raw" | "jsonl-messages" | "csv" | "sharegpt";

const EXPORT_FORMAT_LABELS: Record<ConvExportFormat, string> = {
  "jsonl-raw": "Training JSONL",
  "jsonl-messages": "Message JSONL",
  csv: "CSV",
  sharegpt: "ShareGPT JSONL",
};

export const EXPORT_FORMATS_LIST = (
  Object.keys(EXPORT_FORMAT_LABELS) as ConvExportFormat[]
).map((fmt) => ({ fmt, label: EXPORT_FORMAT_LABELS[fmt] }));

export const COMBINED_EXPORT_FORMATS_LIST = EXPORT_FORMATS_LIST.filter(
  ({ fmt }) => canMergeConversationExport(fmt),
);

async function buildThreadContent(
  threadId: string,
  format: ConvExportFormat,
): Promise<string | null> {
  const messages = await loadConversationMessages(threadId, {
    includeSiblings: exportFormatIncludesSiblings(format),
  });
  if (!messages) return null;

  if (format === "jsonl-raw" || format === "jsonl-messages") {
    const oaiMsgs: OAIMessage[] = messages.flatMap((msg) => messageToOpenAI(msg));
    if (oaiMsgs.length === 0) return null;
    return conversationJsonlBody(
      oaiMsgs,
      format === "jsonl-messages" ? "messages" : "training",
    );
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
  if (!canMergeConversationExport(format) && threadIds.length > 1) {
    toast.info("Message JSONL is available per chat.");
    return;
  }

  const parts: string[] = [];
  const header = csvHeader(format);

  for (const id of threadIds) {
    const content = await buildThreadContent(id, format);
    if (content) parts.push(content);
  }

  if (parts.length === 0) { toast.info("No exportable content."); return; }

  const body = header
    ? header + "\n" + parts.join("\n")
    : ndjsonBody(parts);

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
    const body = header ? header + "\n" + content : ndjsonBody([content]);
    files[`${id}.${ext}`] = strToU8(body);
  }

  if (Object.keys(files).length === 0) { toast.info("No exportable content."); return; }

  const zipped = zipSync(files);
  await downloadBlob(zipped, `${basename}.zip`, "application/zip");
}

// Scope-level bulk export shared by the sidebar Recents menu and Settings > Chat > Data.
// "recents" = chats outside projects; "all" adds project chats.
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

// One JSONL line per conversation, string-only content in system/user/assistant turns: Unsloth's
// training tab detects this as ChatML natively and it works with train-on-completions
// masking. Reasoning, tool calls and images are dropped for clean SFT targets.

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
  // Only attachment text is unwrapped: a message body is verbatim, and may legitimately quote the
  // wrapper syntax in a code sample.
  const collect = (blocks: unknown, fromAttachment = false) => {
    const normalize = fromAttachment
      ? unwrapPastedTextContent
      : (text: string) => text;
    // Legacy and imported histories can store content as a plain string.
    if (typeof blocks === "string") {
      if (blocks.trim()) parts.push(normalize(blocks));
      return;
    }
    if (!Array.isArray(blocks)) return;
    for (const b of blocks) {
      if (!b || typeof b !== "object") {
        continue;
      }
      const block = b as Record<string, unknown>;
      if (block.type === "text" && typeof block.text === "string" && block.text) {
        parts.push(normalize(block.text));
      }
    }
  };
  collect(msg.content);
  if (Array.isArray(msg.attachments)) {
    for (const attachment of msg.attachments as Array<{ content?: unknown }>) {
      collect(attachment?.content, true);
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

/** Conversation turns for fine-tuning, or null when the thread has no usable user + assistant
 *  exchange. Consecutive same-role turns merge, assistant turns before the first user turn
 *  drop (a target with no prompt teaches nothing), and trailing non-assistant turns drop. */
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

/** JSONL lines for one conversation in the chosen format. Alpaca is single-turn, so each user to
 *  assistant pair becomes its own record with the system prompt and earlier exchange in the
 *  input field. */
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
    // Chain only: retries/regenerations leave sibling branches, and mixing alternate replies into one
    // conversation corrupts the training targets.
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
    ndjsonBody(lines),
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

// ShareGPT training exports: prompt to one record (human turn + empty gpt slot); list to one
// multi-turn record, each item a human turn.
function exportPromptTrainingJsonl(entry: PromptEntry): Promise<void> {
  const record = {
    conversations: [
      { from: "human", value: entry.text },
      { from: "gpt", value: "" },
    ],
  };
  return downloadBlob(
    ndjsonBody([JSON.stringify(record)]),
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
    );
  return downloadBlob(ndjsonBody(lines), "prompts-training.jsonl", "application/x-ndjson");
}

function exportListTrainingJsonl(entry: PromptListEntry): Promise<void> {
  const conversations = entry.items.flatMap((text) => [
    { from: "human", value: text },
    { from: "gpt", value: "" },
  ]);
  return downloadBlob(
    ndjsonBody([JSON.stringify({ conversations })]),
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
    });
  return downloadBlob(ndjsonBody(lines), "prompt-lists-training.jsonl", "application/x-ndjson");
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

// Unsaved edits live in the parent keyed by entry id, so switching rows in the rail never
// silently throws away what you typed.
type PromptDraft = { name: string; text: string };
type ListDraft = { name: string; items: string[] };

// Factories, not shared constants: every reset would otherwise hand back the same `items` array,
// one in-place edit from corrupting it for good.
const emptyPromptDraft = (): PromptDraft => ({ name: "", text: "" });
const emptyListDraft = (): ListDraft => ({ name: "", items: ["", ""] });

// Selecting the Lists tab auto-selects its first row, and one controlled textarea per item makes
// that cost grow faster than the item count: in Chromium 100 items settle in 247ms, 200 in 568ms,
// 500 in 2.5s and 2000 in 36s, against a flat 80ms on the collapsed card this replaced. The backend
// accepts 10000 items in one list (routes/prompts.py), which an import can produce, so past this
// many the editor waits to be asked for; save, run and export all still read the full items.
const EDITOR_ROW_LIMIT = 100;

function relativeTime(ts: number): string {
  const secs = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

// `current` drives the roving tabindex and stays true while a New form covers the pane: without
// it every row is tabbable, so reaching the editor means tabbing through the library.
function RailRow({
  title,
  preview,
  selected,
  current,
  dirty,
  badge,
  leading,
  onSelect,
}: {
  title: string;
  preview: string;
  selected: boolean;
  current: boolean;
  dirty: boolean;
  badge?: string;
  leading?: ReactElement | null;
  onSelect: () => void;
}): ReactElement {
  return (
    <button
      type="button"
      data-rail-row=""
      tabIndex={current ? 0 : -1}
      aria-current={selected ? "true" : undefined}
      onClick={onSelect}
      className={cn(
        "w-full rounded-lg px-2.5 py-2 text-left transition-colors border",
        "focus-visible:ring-1 focus-visible:ring-ring outline-none",
        selected
          ? "bg-muted/70 border-border"
          : "border-transparent hover:bg-muted/40",
      )}
    >
      <div className="flex items-center gap-1.5">
        {leading}
        <span className="flex-1 truncate text-xs font-medium tracking-tight">{title}</span>
        {dirty && (
          <span className="size-1.5 shrink-0 rounded-full bg-primary" title="Unsaved changes" />
        )}
        {badge && (
          <span className="shrink-0 rounded-full bg-muted px-1.5 text-ui-11 tabular-nums text-muted-foreground">
            {badge}
          </span>
        )}
      </div>
      <p className="mt-0.5 truncate text-ui-11 text-muted-foreground/70">{preview}</p>
    </button>
  );
}

function EmptyDetail({
  icon,
  title,
  hint,
  onClearSearch,
}: {
  icon: ReactElement;
  title: string;
  hint?: string;
  onClearSearch?: () => void;
}): ReactElement {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
      <div className="flex size-12 items-center justify-center rounded-2xl bg-muted/60">{icon}</div>
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium text-muted-foreground">{title}</p>
        {hint && <p className="text-xs text-muted-foreground/60">{hint}</p>}
        {onClearSearch && (
          <button type="button" onClick={onClearSearch} className="text-xs text-primary hover:underline">
            Clear search
          </button>
        )}
      </div>
    </div>
  );
}

// Deleting a dirty entry destroys the stored copy and the only copy of the unsaved draft at once,
// so that case asks first.
function UnsavedDeleteConfirm({
  open,
  onOpenChange,
  kind,
  name,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  kind: "prompt" | "list";
  name: string;
  onConfirm: () => void;
}): ReactElement {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            Delete {kind} with unsaved changes
          </AlertDialogTitle>
          <AlertDialogDescription>
            <span className="font-medium text-foreground">&quot;{name}&quot;</span> has
            unsaved edits. Deleting discards the saved {kind} and those edits together.
            This cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onConfirm}>
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function PromptDetail({
  entry,
  draft,
  onDraftChange,
  onUse,
  onExport,
  onRefresh,
  onDeleted,
  onSaved,
  pending,
  runMutation,
}: {
  entry: PromptEntry;
  draft: PromptDraft | undefined;
  onDraftChange: (draft: PromptDraft | undefined) => void;
  onUse: (text: string) => void;
  onExport: (entry: PromptEntry) => void;
  onRefresh: () => Promise<void>;
  onDeleted: (deletedId: string) => void;
  onSaved: (submitted: PromptDraft) => void;
  pending: boolean;
  runMutation: (id: string, fn: () => Promise<void>) => Promise<void>;
}): ReactElement {
  const pinnedPromptIds = usePlusMenuPrefsStore((s) => s.pinnedPromptIds);
  const togglePinnedPrompt = usePlusMenuPrefsStore((s) => s.togglePinnedPrompt);
  const isPinned = pinnedPromptIds.includes(entry.id);
  const [preview, setPreview] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const name = draft?.name ?? entry.name;
  const text = draft?.text ?? entry.text;
  const dirty = draft !== undefined;

  const update = useCallback(
    (patch: Partial<PromptDraft>) => {
      const next = { name, text, ...patch };
      if (next.name === entry.name && next.text === entry.text) onDraftChange(undefined);
      else onDraftChange(next);
    },
    [name, text, entry.name, entry.text, onDraftChange],
  );

  // One mutation at a time, and the lock lives in the parent: this pane is keyed by row id, so
  // selecting another row unmounts it and a local lock would come back false. PUT is an
  // unconditional upsert, so a Save still in flight can land after a DELETE.
  const handleSave = useCallback(async () => {
    const trimText = text.trim();
    if (!trimText) return;
    // Snapshot what is going to the server: the editor stays usable while the request is in flight,
    // so the draft can move on before it resolves.
    const submitted: PromptDraft = { name, text };
    await runMutation(lockKey("prompt", entry.id), async () => {
      try {
        await savePromptEntry({
          ...entry,
          name: name.trim() || "Untitled Prompt",
          text: trimText,
          updatedAt: now(),
        });
        // Refresh before dropping the draft. Dropping it first uncovers the pre-save copy this pane still
        // holds, so the editor flashes the old text until the fetch lands.
        await onRefresh();
        onSaved(submitted);
      } catch (err) {
        toast.error("Could not save prompt", {
          description: err instanceof Error ? err.message : "Please try again.",
        });
      }
    });
  }, [entry, name, text, onSaved, onRefresh, runMutation]);

  const handleDelete = useCallback(async () => {
    await runMutation(lockKey("prompt", entry.id), async () => {
      try {
        await deletePromptEntry(entry.id);
      } catch (err) {
        toast.error("Could not delete prompt", {
          description: err instanceof Error ? err.message : "Please try again.",
        });
        return;
      }
      onDraftChange(undefined);
      // Refresh before clearing, or the keep-a-row-selected pass runs against a list that still holds
      // this row and can select the deleted entry back.
      await onRefresh();
      // Id goes up so the parent only clears if this row is still selected: the delete is awaited, and
      // a click elsewhere meanwhile would be undone.
      onDeleted(entry.id);
    });
  }, [entry.id, onDraftChange, onDeleted, onRefresh, runMutation]);

  // Export what the pane shows, normalised as saving would, or a dirty entry writes a file that does
  // not match the screen.
  const exportValue: PromptEntry = dirty
    ? { ...entry, name: name.trim() || "Untitled Prompt", text: text.trim() }
    : entry;

  return (
    <>
    <div className="flex h-full min-h-0 flex-col gap-3">
      <input
        value={name}
        onChange={(e) => update({ name: e.target.value })}
        placeholder="Prompt name..."
        className="w-full shrink-0 rounded-lg border-0 bg-background/80 px-3 py-2 text-sm font-medium ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
      />
      {preview ? (
        <MarkdownPreview
          markdown={text}
          className="min-h-0 max-h-none flex-1 rounded-lg border-border/60 bg-background/40 p-3 text-sm"
        />
      ) : (
        <textarea
          value={text}
          onChange={(e) => update({ text: e.target.value })}
          placeholder="Prompt text..."
          className="min-h-0 flex-1 w-full resize-none rounded-lg border-0 bg-background/80 px-3 py-2.5 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow leading-relaxed"
        />
      )}
      <div className="flex shrink-0 items-center gap-2 text-ui-11 text-muted-foreground/70">
        <span className="tabular-nums">{text.length.toLocaleString()} characters</span>
        <span className="text-muted-foreground/30">·</span>
        <span>updated {relativeTime(entry.updatedAt)}</span>
        {dirty && (
          <>
            <span className="text-muted-foreground/30">·</span>
            <span className="text-primary">unsaved changes</span>
          </>
        )}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-1 border-t border-border/50 pt-3">
        <button
          type="button"
          onClick={() => togglePinnedPrompt(entry.id)}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
            isPinned
              ? "text-primary hover:bg-primary/10"
              : "text-muted-foreground hover:bg-muted hover:text-foreground",
          )}
          title={isPinned ? "Unpin from + menu" : "Pin to + menu"}
        >
          <BookmarkIcon className={cn("size-4", isPinned && "fill-primary")} />
        </button>
        <button
          type="button"
          onClick={() => onExport(exportValue)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          title="Export"
        >
          <DownloadIcon className="size-4" />
        </button>
        <button
          type="button"
          disabled={pending}
          onClick={() => (dirty ? setConfirmingDelete(true) : void handleDelete())}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-muted-foreground"
          title="Delete"
        >
          <Trash2Icon className="size-4" />
        </button>
        <div className="flex-1" />
        {dirty && (
          <Button size="sm" variant="ghost" onClick={() => onDraftChange(undefined)}>
            <RotateCcwIcon className="size-3.5 mr-1" />Revert
          </Button>
        )}
        <Button size="sm" variant="outline" onClick={() => setPreview((v) => !v)}>
          {preview ? (
            <><PencilIcon className="size-3.5 mr-1" />Edit</>
          ) : (
            <><EyeIcon className="size-3.5 mr-1" />Preview</>
          )}
        </Button>
        <Button size="sm" variant="outline" disabled={pending || !dirty || !text.trim()} onClick={handleSave}>
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 mr-1" />Save
        </Button>
        <Button size="sm" onClick={() => onUse(text)} disabled={!text.trim()}>
          <PlayIcon className="size-3 mr-1" />Use
        </Button>
      </div>
    </div>
    <UnsavedDeleteConfirm
      open={confirmingDelete}
      onOpenChange={setConfirmingDelete}
      kind="prompt"
      name={name.trim() || "Untitled Prompt"}
      onConfirm={() => {
        setConfirmingDelete(false);
        void handleDelete();
      }}
    />
    </>
  );
}

// A create has no row id yet, so it cannot use the by-id mutation lock above. The ref is the
// authority because React state is not readable straight after scheduling it. Call this above
// the New forms, never inside one: selecting a rail row unmounts the form, and a guard that
// dies with it comes back false while its request is still out.
function useCreateGuard(): {
  creating: boolean;
  create: (fn: () => Promise<void>) => Promise<void>;
} {
  const creatingRef = useRef(false);
  const [creating, setCreating] = useState(false);
  const create = useCallback(async (fn: () => Promise<void>) => {
    if (creatingRef.current) return;
    creatingRef.current = true;
    setCreating(true);
    try {
      await fn();
    } finally {
      creatingRef.current = false;
      setCreating(false);
    }
  }, []);
  return { creating, create };
}

// Draft lives in the parent, like the edit drafts: selecting a row hides this form, and local
// state would be discarded with it.
function NewPromptForm({
  draft,
  onDraftChange,
  onClose,
  onRefresh,
  onCreated,
  creating,
  create,
}: {
  draft: PromptDraft;
  onDraftChange: (draft: PromptDraft) => void;
  onClose: () => void;
  onRefresh: () => Promise<void>;
  onCreated: (id: string, submitted: PromptDraft, fromOpenForm: boolean) => void;
  creating: boolean;
  create: (fn: () => Promise<void>) => Promise<void>;
}): ReactElement {
  const { name, text } = draft;
  const setName = (value: string) => onDraftChange({ ...draft, name: value });
  const setText = (value: string) => onDraftChange({ ...draft, text: value });
  // A create outlives the form that started it, so this answers "is the form the user is looking at
  // still mine". Set in setup, not just at the ref: StrictMode replays setup, cleanup, setup,
  // and the flag would stay false from that first cleanup for the pane's whole life.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const handleSave = useCallback(
    () =>
      create(async () => {
        const trimText = text.trim();
        if (!trimText) return;
        const ts = now();
        const id = newId();
        // What the request carries: the fields stay editable while it is out, so the draft can move on
        // before it resolves.
        const submitted: PromptDraft = { name, text };
        try {
          await savePromptEntry({
            id,
            name: name.trim() || "Untitled Prompt",
            text: trimText,
            createdAt: ts,
            updatedAt: ts,
          });
        } catch (err) {
          // Without this the rejection is unhandled and the form just sits there, so the prompt looks saved
          // until the dialog is reopened.
          toast.error("Could not create prompt", {
            description: err instanceof Error ? err.message : "Please try again.",
          });
          return;
        }
        // Await the refresh first, or the keep-a-row-selected effect runs against a list without the new
        // id and bounces the selection off it.
        await onRefresh();
        // The parent resets the draft if it still holds what was sent, and moves the view to the new row
        // only while this form is still on screen.
        onCreated(id, submitted, mounted.current);
      }),
    [name, text, create, onRefresh, onCreated],
  );

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <p className="shrink-0 text-xs font-semibold text-muted-foreground">New Prompt</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Prompt name (optional)..."
        autoFocus
        className="w-full shrink-0 rounded-lg border-0 bg-background/80 px-3 py-2 text-sm font-medium ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
      />
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Write your prompt here..."
        className="min-h-0 flex-1 w-full resize-none rounded-lg border-0 bg-background/80 px-3 py-2.5 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow leading-relaxed"
      />
      <div className="flex shrink-0 flex-wrap gap-2 justify-end border-t border-border/50 pt-3">
        <Button size="sm" variant="ghost" onClick={onClose}>
          <XIcon className="size-3.5 mr-1" />Cancel
        </Button>
        <Button size="sm" onClick={handleSave} disabled={creating || !text.trim()}>
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 mr-1" />Save Prompt
        </Button>
      </div>
    </div>
  );
}

function PromptListDetail({
  entry,
  draft,
  onDraftChange,
  onRunList,
  onExport,
  onRefresh,
  onDeleted,
  onSaved,
  pending,
  runMutation,
}: {
  entry: PromptListEntry;
  draft: ListDraft | undefined;
  onDraftChange: (draft: ListDraft | undefined) => void;
  onRunList?: (items: string[]) => void;
  onExport: (entry: PromptListEntry) => void;
  onRefresh: () => Promise<void>;
  onDeleted: (deletedId: string) => void;
  onSaved: (submitted: ListDraft) => void;
  pending: boolean;
  runMutation: (id: string, fn: () => Promise<void>) => Promise<void>;
}): ReactElement {
  const [preview, setPreview] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const name = draft?.name ?? entry.name;
  const items = draft?.items ?? entry.items;
  // Decided once per list and latched: this pane is keyed by row id, so selecting another list
  // re-decides it, but Add prompt taking a 100-item list to 101 must not unmount the editor.
  const [editorMounted, setEditorMounted] = useState(
    () => items.length <= EDITOR_ROW_LIMIT,
  );
  const dirty = draft !== undefined;
  const savable = items.filter((t) => t.trim()).length > 0;

  const update = useCallback(
    (patch: Partial<ListDraft>) => {
      const next = { name, items, ...patch };
      const same =
        next.name === entry.name &&
        next.items.length === entry.items.length &&
        next.items.every((v, i) => v === entry.items[i]);
      if (same) onDraftChange(undefined);
      else onDraftChange(next);
    },
    [name, items, entry.name, entry.items, onDraftChange],
  );

  // See PromptDetail: a Save landing after a DELETE would upsert the row back, and the lock is the
  // parent's because this pane unmounts on a row switch.
  const handleSave = useCallback(async () => {
    const filtered = items.filter((t) => t.trim());
    if (filtered.length === 0) return;
    // See PromptDetail: the editor stays usable while the request is in flight.
    const submitted: ListDraft = { name, items };
    await runMutation(lockKey("list", entry.id), async () => {
      try {
        await savePromptList({
          ...entry,
          name: name.trim() || "Untitled List",
          items: filtered,
          updatedAt: now(),
        });
        // See PromptDetail: the draft is what covers the pre-save entry.
        await onRefresh();
        onSaved(submitted);
      } catch (err) {
        toast.error("Could not save list", {
          description: err instanceof Error ? err.message : "Please try again.",
        });
      }
    });
  }, [entry, name, items, onSaved, onRefresh, runMutation]);

  const handleDelete = useCallback(async () => {
    await runMutation(lockKey("list", entry.id), async () => {
      try {
        await deletePromptList(entry.id);
      } catch (err) {
        toast.error("Could not delete list", {
          description: err instanceof Error ? err.message : "Please try again.",
        });
        return;
      }
      onDraftChange(undefined);
      // See PromptDetail: clearing first can reselect the row being deleted.
      await onRefresh();
      // See PromptDetail: only the parent knows whether this row is still current.
      onDeleted(entry.id);
    });
  }, [entry.id, onDraftChange, onDeleted, onRefresh, runMutation]);

  // Run what the editor shows. Off entry.items, deleting every draft item left the button enabled
  // on the stored length and ran the old list.
  const runnableItems = items.filter((t) => t.trim());

  // See PromptDetail: export the visible draft, not the last saved copy.
  const exportValue: PromptListEntry = dirty
    ? {
        ...entry,
        name: name.trim() || "Untitled List",
        items: items.filter((t) => t.trim()),
      }
    : entry;

  return (
    <>
    <div className="flex h-full min-h-0 flex-col gap-3">
      <input
        value={name}
        onChange={(e) => update({ name: e.target.value })}
        placeholder="List name..."
        className="w-full shrink-0 rounded-lg border-0 bg-background/80 px-3 py-2 text-sm font-medium ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
      />
      <p className="shrink-0 text-xs font-semibold text-muted-foreground">
        Prompts, loaded into the composer one at a time
      </p>
      <div className="min-h-0 flex-1 overflow-y-auto pr-1 flex flex-col gap-2">
        {editorMounted ? (
          <>
            <SortablePromptItems
              items={items}
              onChange={(next) => update({ items: next })}
              preview={preview}
            />
            {preview ? null : (
              <button
                type="button"
                onClick={() => update({ items: [...items, ""] })}
                className="flex items-center gap-1.5 self-start text-xs font-medium text-primary hover:text-primary/80 transition-colors"
              >
                <PlusIcon className="size-3.5" />Add prompt
              </button>
            )}
          </>
        ) : (
          <>
            <div className="flex flex-col gap-1.5">
              {items.slice(0, 5).map((item, i) => (
                <p key={i} className="flex gap-2 text-xs leading-relaxed text-muted-foreground">
                  <span className="shrink-0 tabular-nums text-muted-foreground/40">{i + 1}.</span>
                  <span className="line-clamp-1">{item}</span>
                </p>
              ))}
            </div>
            <p className="text-ui-11 text-muted-foreground/60">
              {items.length - 5} more. Opening the editor for a list this long takes a
              while, so it waits for you.
            </p>
            <button
              type="button"
              onClick={() => setEditorMounted(true)}
              className="flex items-center gap-1.5 self-start text-xs font-medium text-primary hover:text-primary/80 transition-colors"
            >
              <PencilIcon className="size-3.5" />Edit all {items.length} prompts
            </button>
          </>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2 text-ui-11 text-muted-foreground/70">
        <span className="tabular-nums">{items.length} prompts</span>
        <span className="text-muted-foreground/30">·</span>
        <span>updated {relativeTime(entry.updatedAt)}</span>
        {dirty && (
          <>
            <span className="text-muted-foreground/30">·</span>
            <span className="text-primary">unsaved changes</span>
          </>
        )}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-1 border-t border-border/50 pt-3">
        <button
          type="button"
          onClick={() => onExport(exportValue)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          title="Export"
        >
          <DownloadIcon className="size-4" />
        </button>
        <button
          type="button"
          disabled={pending}
          onClick={() => (dirty ? setConfirmingDelete(true) : void handleDelete())}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-muted-foreground"
          title="Delete"
        >
          <Trash2Icon className="size-4" />
        </button>
        <div className="flex-1" />
        {dirty && (
          <Button size="sm" variant="ghost" onClick={() => onDraftChange(undefined)}>
            <RotateCcwIcon className="size-3.5 mr-1" />Revert
          </Button>
        )}
        {/* The deferred summary renders neither editor nor preview, so the toggle would only relabel itself. */}
        {editorMounted && (
          <Button size="sm" variant="outline" onClick={() => setPreview((v) => !v)}>
            {preview ? (
              <><PencilIcon className="size-3.5 mr-1" />Edit</>
            ) : (
              <><EyeIcon className="size-3.5 mr-1" />Preview</>
            )}
          </Button>
        )}
        <Button size="sm" variant="outline" disabled={pending || !dirty || !savable} onClick={handleSave}>
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 mr-1" />Save
        </Button>
        {onRunList && (
          <Button size="sm" onClick={() => onRunList(runnableItems)} disabled={runnableItems.length === 0}>
            <PlayIcon className="size-3 mr-1" />Run
          </Button>
        )}
      </div>
    </div>
    <UnsavedDeleteConfirm
      open={confirmingDelete}
      onOpenChange={setConfirmingDelete}
      kind="list"
      name={name.trim() || "Untitled List"}
      onConfirm={() => {
        setConfirmingDelete(false);
        void handleDelete();
      }}
    />
    </>
  );
}

// See NewPromptForm: the in-progress list lives in the parent so clicking a row in the rail
// cannot silently discard a partially authored list.
function NewPromptListForm({
  draft,
  onDraftChange,
  onClose,
  onRefresh,
  onCreated,
  creating,
  create,
}: {
  draft: ListDraft;
  onDraftChange: (draft: ListDraft) => void;
  onClose: () => void;
  onRefresh: () => Promise<void>;
  onCreated: (id: string, submitted: ListDraft, fromOpenForm: boolean) => void;
  creating: boolean;
  create: (fn: () => Promise<void>) => Promise<void>;
}): ReactElement {
  const { name, items } = draft;
  const setName = (value: string) => onDraftChange({ ...draft, name: value });
  const setItems = (value: string[]) => onDraftChange({ ...draft, items: value });
  // See NewPromptForm, including why the flag is set in setup.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const handleSave = useCallback(
    () =>
      create(async () => {
        const filtered = items.filter((t) => t.trim());
        if (filtered.length === 0) return;
        const ts = now();
        const id = newId();
        // See NewPromptForm: the editor stays usable while the request is out.
        const submitted: ListDraft = { name, items };
        try {
          await savePromptList({
            id,
            name: name.trim() || "Untitled List",
            items: filtered,
            createdAt: ts,
            updatedAt: ts,
          });
        } catch (err) {
          // See NewPromptForm: an unhandled rejection reads as a saved list.
          toast.error("Could not create list", {
            description: err instanceof Error ? err.message : "Please try again.",
          });
          return;
        }
        // See NewPromptForm: select only once the refreshed list contains the id.
        await onRefresh();
        onCreated(id, submitted, mounted.current);
      }),
    [name, items, create, onRefresh, onCreated],
  );

  const addItem = () => setItems([...items, ""]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <p className="shrink-0 text-xs font-semibold text-muted-foreground">New Prompt List</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="List name..."
        autoFocus
        className="w-full shrink-0 rounded-lg border-0 bg-background/80 px-3 py-2 text-sm font-medium ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow"
      />
      <p className="shrink-0 text-xs font-semibold text-muted-foreground">
        Prompts, loaded into the composer one at a time
      </p>
      <div className="min-h-0 flex-1 overflow-y-auto pr-1 flex flex-col gap-2">
        <SortablePromptItems items={items} onChange={setItems} minItems={1} />
        <button
          type="button"
          onClick={addItem}
          className="flex items-center gap-1.5 self-start text-xs font-medium text-primary hover:text-primary/80 transition-colors"
        >
          <PlusIcon className="size-3.5" />Add another prompt
        </button>
      </div>
      <div className="flex shrink-0 flex-wrap gap-2 justify-end border-t border-border/50 pt-3">
        <Button size="sm" variant="ghost" onClick={onClose}>
          <XIcon className="size-3.5 mr-1" />Cancel
        </Button>
        <Button
          size="sm"
          onClick={handleSave}
          disabled={creating || items.filter((t) => t.trim()).length === 0}
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
  const pinnedPromptIds = usePlusMenuPrefsStore((s) => s.pinnedPromptIds);

  const [promptEntries, setPromptEntries] = useState<PromptEntry[]>([]);
  const [promptLists, setPromptLists] = useState<PromptListEntry[]>([]);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [selectedListId, setSelectedListId] = useState<string | null>(null);
  // Maps, not plain objects: ids are arbitrary strings, so one named "constructor" would resolve to
  // an inherited prototype member.
  const [promptDrafts, setPromptDrafts] = useState<Map<string, PromptDraft>>(
    () => new Map(),
  );
  const [listDrafts, setListDrafts] = useState<Map<string, ListDraft>>(
    () => new Map(),
  );

  // In-progress new entries, held here rather than inside the forms so that hiding a form does not destroy the work.
  const [newPromptDraft, setNewPromptDraft] = useState<PromptDraft>(emptyPromptDraft);
  const [newListDraft, setNewListDraft] = useState<ListDraft>(emptyListDraft);
  const newPromptStarted =
    newPromptDraft.name.trim() !== "" || newPromptDraft.text.trim() !== "";
  const newListStarted =
    newListDraft.name.trim() !== "" || newListDraft.items.some((t) => t.trim() !== "");

  const setPromptDraft = useCallback((id: string, draft: PromptDraft | undefined) => {
    setPromptDrafts((prev) => {
      if (!draft) {
        if (!prev.has(id)) return prev;
        const next = new Map(prev);
        next.delete(id);
        return next;
      }
      return new Map(prev).set(id, draft);
    });
  }, []);

  const setListDraft = useCallback((id: string, draft: ListDraft | undefined) => {
    setListDrafts((prev) => {
      if (!draft) {
        if (!prev.has(id)) return prev;
        const next = new Map(prev);
        next.delete(id);
        return next;
      }
      return new Map(prev).set(id, draft);
    });
  }, []);

  // Mutation locks live here rather than in the detail panes, which are keyed by row id and unmount
  // the moment another row is selected. Held by id until the request settles, or a Save in
  // flight during a row switch comes back unlocked and its PUT can land after a DELETE.
  const [mutatingIds, setMutatingIds] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  );
  // The ref is the authority, not the state: a functional updater is not guaranteed to run during
  // setState, so reading the outcome straight after scheduling one can see a stale answer, skip
  // the request, and still acquire the id later during render with nothing to release it.
  const mutatingRef = useRef<ReadonlySet<string>>(new Set<string>());
  const runMutation = useCallback(
    async (id: string, fn: () => Promise<void>) => {
      const [held, started] = acquire(mutatingRef.current, id);
      // The buttons are disabled while locked, but a second caller reaching here anyway must not run and
      // must not clear the first one's lock.
      if (!started) return;
      mutatingRef.current = held;
      setMutatingIds(held);
      try {
        await fn();
      } finally {
        mutatingRef.current = release(mutatingRef.current, id);
        setMutatingIds(mutatingRef.current);
      }
    },
    [],
  );

  // Above the New forms, not inside them: selecting a rail row unmounts the form while its create is
  // still out, and a guard mounted with it would let reopening New mint a second id.
  const promptCreate = useCreateGuard();
  const listCreate = useCreateGuard();

  // Only drop the draft if it still holds what the request carried: anything typed while the save
  // was in flight is the user's most recent intent.
  const clearPromptDraftIfSaved = useCallback(
    (id: string, submitted: PromptDraft) => {
      setPromptDrafts((prev) => {
        const current = prev.get(id);
        if (!current || !samePromptDraft(current, submitted)) return prev;
        const next = new Map(prev);
        next.delete(id);
        return next;
      });
    },
    [],
  );
  const clearListDraftIfSaved = useCallback(
    (id: string, submitted: ListDraft) => {
      setListDrafts((prev) => {
        const current = prev.get(id);
        if (!current || !sameListDraft(current, submitted)) return prev;
        const next = new Map(prev);
        next.delete(id);
        return next;
      });
    },
    [],
  );

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

  // In the handler, not an effect keyed to activeTab: an effect leaves one render of the new tab
  // still holding the old query, and the keep-a-row-selected pass runs in it and drops the row
  // that tab had selected.
  const selectTab = useCallback((tab: Tab) => {
    setActiveTab(tab);
    setSearchQuery("");
    setShowSuggestions(false);
    setShowNewPrompt(false);
    setShowNewList(false);
  }, []);

  // Clear the search too: an active one the new entry does not match keeps it out of the filtered
  // rail. The draft resets only when it still holds what was sent, since the fields stay
  // editable while the create is out.
  const selectCreatedPrompt = useCallback(
    (id: string, submitted: PromptDraft, fromOpenForm: boolean) => {
      setNewPromptDraft((prev) =>
        samePromptDraft(prev, submitted) ? emptyPromptDraft() : prev,
      );
      // The user left the form while the request was out, so they are looking at something else on
      // purpose. Take the refresh, leave the view alone.
      if (!fromOpenForm) return;
      setSearchQuery("");
      setSelectedPromptId(id);
      setShowNewPrompt(false);
    },
    [],
  );

  const selectCreatedList = useCallback(
    (id: string, submitted: ListDraft, fromOpenForm: boolean) => {
      setNewListDraft((prev) =>
        sameListDraft(prev, submitted) ? emptyListDraft() : prev,
      );
      if (!fromOpenForm) return;
      setSearchQuery("");
      setSelectedListId(id);
      setShowNewList(false);
    },
    [],
  );

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

  // Keep a row selected whenever the filtered rail is non-empty. During render, not in an effect:
  // an effect paints the blank pane once before correcting it. The visible tab only, since
  // searchQuery is shared and correcting against the hidden one dropped the row selected there.
  if (activeTab === "prompts") {
    if (filteredPrompts.length === 0) {
      if (selectedPromptId !== null) setSelectedPromptId(null);
    } else if (!filteredPrompts.some((e) => e.id === selectedPromptId)) {
      setSelectedPromptId(filteredPrompts[0].id);
    }
  } else if (filteredLists.length === 0) {
    if (selectedListId !== null) setSelectedListId(null);
  } else if (!filteredLists.some((e) => e.id === selectedListId)) {
    setSelectedListId(filteredLists[0].id);
  }

  const selectedPrompt = useMemo(
    () => promptEntries.find((e) => e.id === selectedPromptId) ?? null,
    [promptEntries, selectedPromptId],
  );
  const selectedList = useMemo(
    () => promptLists.find((e) => e.id === selectedListId) ?? null,
    [promptLists, selectedListId],
  );

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

  // The rail is one tab stop; arrows move within it. Clicking rather than setting the id keeps the
  // two tabs on one handler.
  const handleRailKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const rows = Array.from(
      e.currentTarget.querySelectorAll<HTMLButtonElement>("[data-rail-row]"),
    );
    const at = rows.indexOf(document.activeElement as HTMLButtonElement);
    if (at < 0) return;
    const next = rows[at + (e.key === "ArrowDown" ? 1 : -1)];
    if (!next) return;
    e.preventDefault();
    next.focus();
    next.click();
  }, []);

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
        <DialogContent showCloseButton={false} className="sm:max-w-[min(1100px,88vw)] max-h-[94dvh] flex flex-col gap-0 p-0 overflow-hidden">
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
                  onClick={() => selectTab(tab)}
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
          {/* min-h-0, not a floor: the grid rows below already carry the minimum each pane's chrome needs,
              and a floor here has to guess the height of the header and search above it, whose text
              wraps on a narrow dialog. */}
          <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 pb-4 sm:pb-6 grid gap-2 sm:gap-4 grid-cols-1 grid-rows-[minmax(132px,30%)_minmax(272px,1fr)] sm:grid-cols-[200px_minmax(0,1fr)] sm:grid-rows-1 lg:grid-cols-[248px_minmax(0,1fr)]">
            {/* */}
            <div className="flex min-h-[132px] flex-col gap-2 rounded-xl border border-border/50 bg-muted/20 p-2">
              <button
                type="button"
                onClick={() => {
                  if (activeTab === "prompts") setShowNewPrompt(true);
                  else setShowNewList(true);
                }}
                className="flex shrink-0 items-center justify-center gap-1.5 rounded-lg border border-dashed border-border/60 px-3 py-2 text-xs font-medium text-muted-foreground hover:border-border hover:bg-muted/50 hover:text-foreground transition-all"
              >
                <PlusIcon className="size-3.5" />
                {activeTab === "prompts" ? "New prompt" : "New prompt list"}
                {(activeTab === "prompts" ? newPromptStarted : newListStarted) && (
                  <span
                    className="size-1.5 shrink-0 rounded-full bg-primary"
                    title="Unsaved draft"
                  />
                )}
              </button>

              <div
                role="group"
                aria-label={activeTab === "prompts" ? "Saved prompts" : "Prompt lists"}
                onKeyDown={handleRailKeyDown}
                className="min-h-0 flex-1 overflow-y-auto flex flex-col gap-0.5"
              >
                {activeTab === "prompts" &&
                  filteredPrompts.map((entry) => (
                    <RailRow
                      key={entry.id}
                      title={entry.name}
                      preview={entry.text.replace(/\s+/g, " ").trim()}
                      selected={!showNewPrompt && entry.id === selectedPromptId}
                      current={entry.id === selectedPromptId}
                      dirty={promptDrafts.has(entry.id)}
                      leading={
                        pinnedPromptIds.includes(entry.id) ? (
                          <BookmarkIcon className="size-3 shrink-0 fill-primary text-primary" />
                        ) : null
                      }
                      onSelect={() => {
                        setShowNewPrompt(false);
                        setSelectedPromptId(entry.id);
                      }}
                    />
                  ))}

                {activeTab === "lists" &&
                  filteredLists.map((entry) => (
                    <RailRow
                      key={entry.id}
                      title={entry.name}
                      preview={(entry.items[0] ?? "").replace(/\s+/g, " ").trim()}
                      badge={String(entry.items.length)}
                      selected={!showNewList && entry.id === selectedListId}
                      current={entry.id === selectedListId}
                      dirty={listDrafts.has(entry.id)}
                      onSelect={() => {
                        setShowNewList(false);
                        setSelectedListId(entry.id);
                      }}
                    />
                  ))}

                {activeTab === "prompts" && filteredPrompts.length === 0 && (
                  <p className="px-2 py-6 text-center text-ui-11 text-muted-foreground/60">
                    {searchQuery.trim() ? "No prompts match" : "No saved prompts yet"}
                  </p>
                )}
                {activeTab === "lists" && filteredLists.length === 0 && (
                  <p className="px-2 py-6 text-center text-ui-11 text-muted-foreground/60">
                    {searchQuery.trim() ? "No lists match" : "No prompt lists yet"}
                  </p>
                )}
              </div>

              <p className="shrink-0 px-1 pb-0.5 text-ui-11 tabular-nums text-muted-foreground/50">
                {activeTab === "prompts"
                  ? `${filteredPrompts.length} of ${promptEntries.length} prompts`
                  : `${filteredLists.length} of ${promptLists.length} lists`}
              </p>
            </div>

            {/* */}
            <div className="min-h-[272px] rounded-xl border border-border/60 bg-card p-4">
              {activeTab === "prompts" &&
                (showNewPrompt ? (
                  <NewPromptForm
                    draft={newPromptDraft}
                    onDraftChange={setNewPromptDraft}
                    onClose={() => {
                      setNewPromptDraft(emptyPromptDraft());
                      setShowNewPrompt(false);
                    }}
                    onRefresh={refreshEntries}
                    onCreated={selectCreatedPrompt}
                    creating={promptCreate.creating}
                    create={promptCreate.create}
                  />
                ) : selectedPrompt ? (
                  <PromptDetail
                    key={selectedPrompt.id}
                    entry={selectedPrompt}
                    draft={promptDrafts.get(selectedPrompt.id)}
                    onDraftChange={(d) => setPromptDraft(selectedPrompt.id, d)}
                    onUse={handleUsePrompt}
                    onExport={(e) => setExportCtx({ kind: "prompt", entry: e })}
                    onRefresh={refreshEntries}
                    onDeleted={(deletedId) =>
                      setSelectedPromptId((prev) => (prev === deletedId ? null : prev))
                    }
                    onSaved={(submitted) =>
                      clearPromptDraftIfSaved(selectedPrompt.id, submitted)
                    }
                    pending={mutatingIds.has(lockKey("prompt", selectedPrompt.id))}
                    runMutation={runMutation}
                  />
                ) : (
                  <EmptyDetail
                    icon={<BookmarkIcon className="size-5 text-muted-foreground/40" />}
                    title={searchQuery.trim() ? `No prompts match “${searchQuery}”` : "No saved prompts yet"}
                    hint={searchQuery.trim() ? undefined : "Save prompts you use often for quick reuse"}
                    onClearSearch={searchQuery.trim() ? () => setSearchQuery("") : undefined}
                  />
                ))}

              {activeTab === "lists" &&
                (showNewList ? (
                  <NewPromptListForm
                    draft={newListDraft}
                    onDraftChange={setNewListDraft}
                    onClose={() => {
                      setNewListDraft(emptyListDraft());
                      setShowNewList(false);
                    }}
                    onRefresh={refreshLists}
                    onCreated={selectCreatedList}
                    creating={listCreate.creating}
                    create={listCreate.create}
                  />
                ) : selectedList ? (
                  <PromptListDetail
                    key={selectedList.id}
                    entry={selectedList}
                    draft={listDrafts.get(selectedList.id)}
                    onDraftChange={(d) => setListDraft(selectedList.id, d)}
                    onRunList={onRunList}
                    onExport={(e) => setExportCtx({ kind: "list", entry: e })}
                    onRefresh={refreshLists}
                    onDeleted={(deletedId) =>
                      setSelectedListId((prev) => (prev === deletedId ? null : prev))
                    }
                    onSaved={(submitted) =>
                      clearListDraftIfSaved(selectedList.id, submitted)
                    }
                    pending={mutatingIds.has(lockKey("list", selectedList.id))}
                    runMutation={runMutation}
                  />
                ) : (
                  <EmptyDetail
                    icon={<LayoutListIcon className="size-5 text-muted-foreground/40" />}
                    title={searchQuery.trim() ? `No prompt lists match “${searchQuery}”` : "No prompt lists yet"}
                    hint={searchQuery.trim() ? undefined : "A prompt list queues a sequence of prompts for quick reuse"}
                    onClearSearch={searchQuery.trim() ? () => setSearchQuery("") : undefined}
                  />
                ))}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {exportCtx && (
        <ExportModal ctx={exportCtx} onClose={() => setExportCtx(null)} />
      )}
    </>
  );
}
