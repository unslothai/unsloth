// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** JSON records stream individually so large exports never become one JS string. */

import {
  ChatThreadWriteError,
  listChatProjects,
  notifyChatHistoryUpdated,
  saveChatProject,
} from "../api/chat-api";
import type { MessageRecord, ParsedConversation, ThreadRecord } from "../types";
import {
  deleteStoredChatThreads,
  saveStoredChatThread,
  syncStoredChatMessages,
} from "./chat-history-storage";
import { parseCsv } from "./csv-parse";
import {
  decodeTextChunks,
  fileImportSource,
  readAllText,
  streamJsonRecords,
  type ImportSource,
} from "./json-record-stream";
import {
  isOpenWebUIRecord,
  openWebUIRecordToConversation,
} from "./openwebui-import";
import {
  isOpenAIMessageRecord,
  messageJsonlConversationRecord,
} from "./ndjson";
import {
  isStudioChatBackup,
  studioBackupProjects,
  studioBackupToConversations,
} from "./studio-backup-import";
import { parseConversationMarkdownDocument } from "./conversation-markdown-import";

const WHOLE_FILE_MAX_BYTES = 64 * 1024 * 1024;

/** Matches MAX_CHAT_IMPORT_CHUNK_BYTES in src-tauri/src/native_file_dialogs.rs. */
const NATIVE_CHUNK_BYTES = 8 * 1024 * 1024;

const WRITE_CONCURRENCY = 6;

/** Byte cadence too, so exports with a few huge chats still show progress. */
const PROGRESS_BYTES = 4 * 1024 * 1024;

export interface ImportProgress {
  imported: number;
  failed: number;
  bytesRead: number;
  totalBytes?: number;
}

export interface ImportOptions {
  onProgress?: (progress: ImportProgress) => void;
  onSaved?: (rowId: string) => void;
}

export interface ImportResult {
  imported: number;
  failed: number;
}

export { fileImportSource, type ImportSource };

async function* nativeBytes(handle: {
  name: string;
  size: number;
  token: string;
}): AsyncGenerator<Uint8Array> {
  const { readNativeChatImportChunk } = await import("@/lib/native-files");
  let offset = 0;
  while (offset < handle.size) {
    const bytes = await readNativeChatImportChunk(
      handle.token,
      offset,
      // Never past the recorded size: appended bytes are not part of the chosen export.
      Math.min(NATIVE_CHUNK_BYTES, handle.size - offset),
    );
    // A short read means the file shrank; do not pass a partial export as whole.
    if (bytes.byteLength === 0) {
      throw new Error(
        `${handle.name} ended after ${offset} of ${handle.size} bytes; it changed after it was picked.`,
      );
    }
    offset += bytes.byteLength;
    yield bytes;
  }
}

export function nativeImportSource(handle: {
  name: string;
  size: number;
  token: string;
}): ImportSource {
  return {
    name: handle.name,
    size: handle.size,
    // Fatal decoding: invalid UTF-8 is rejected rather than saved as U+FFFD.
    chunks: () => decodeTextChunks(nativeBytes(handle), true),
  };
}

// role:"tool" results fold into the preceding tool-call part's `result`.
function oaiContentToParts(raw: unknown): unknown[] {
  if (!Array.isArray(raw)) {
    return typeof raw === "string" && raw.trim()
      ? [{ type: "text", text: raw }]
      : [];
  }

  return raw.flatMap((value): unknown[] => {
    if (typeof value !== "object" || value === null) return [];
    const part = value as Record<string, unknown>;
    if (part.type === "text" && typeof part.text === "string") {
      return [{ type: "text", text: part.text }];
    }
    if (part.type === "image_url") {
      const imageUrl =
        typeof part.image_url === "object" && part.image_url !== null
          ? (part.image_url as Record<string, unknown>).url
          : undefined;
      return typeof imageUrl === "string" && imageUrl
        ? [{ type: "image", image: imageUrl }]
        : [];
    }
    return [];
  });
}

function oaiMessagesToRecords(
  oaiMsgs: unknown[],
  threadId: string,
  baseTs: number,
): MessageRecord[] {
  const toolResults = new Map<string, string>();
  const pending: Array<{ turn: number; part: { toolCallId: string; result?: string } }> = [];

  const records: MessageRecord[] = [];
  let prevId: string | null = null;
  let idx = 0;

  for (const m of oaiMsgs) {
    const msg = m as Record<string, unknown>;
    const role = msg.role as string;
    if (role === "tool") {
      if (typeof msg.tool_call_id !== "string") continue;
      const result = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? "");
      let at = pending.length - 1;
      while (at >= 0 && pending[at].part.toolCallId !== msg.tool_call_id) at--;
      if (at === -1) {
        toolResults.set(msg.tool_call_id, result);
      } else {
        const { turn } = pending[at];
        at = pending.findIndex((entry) => entry.turn === turn && entry.part.toolCallId === msg.tool_call_id);
        pending[at].part.result = result;
        pending.splice(at, 1);
      }
      continue;
    }

    const id = crypto.randomUUID();

    let content: unknown[];

    if (role === "assistant") {
      const parts = oaiContentToParts(msg.content);
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          const tcObj = tc as Record<string, unknown>;
          const fn = (tcObj.function as Record<string, unknown>) ?? {};
          const tcId = typeof tcObj.id === "string" ? tcObj.id : crypto.randomUUID();
          const name = typeof fn.name === "string" ? fn.name : "unknown";
          const argsStr = typeof fn.arguments === "string" ? fn.arguments : "{}";
          let args: unknown = {};
          // _raw matches what the stream adapter and backend keep for invalid JSON arguments.
          try { args = JSON.parse(argsStr); } catch { args = { _raw: argsStr }; }
          const part = {
            type: "tool-call",
            toolCallId: tcId,
            toolName: name,
            args,
            argsText: argsStr,
          };
          parts.push(part);
          pending.push({ turn: idx, part });
        }
      }
      content = parts;
    } else {
      content = oaiContentToParts(msg.content);
    }

    if (content.length === 0) continue;

    records.push({
      id,
      threadId,
      parentId: prevId,
      role: (role === "developer" ? "system" : role) as MessageRecord["role"],
      content: content as MessageRecord["content"],
      createdAt: baseTs + idx,
      metadata: { createdAtEstimated: true },
    });
    prevId = id;
    idx++;
  }

  for (const { part } of pending) {
    const result = toolResults.get(part.toolCallId);
    if (result !== undefined) part.result = result;
  }

  return records;
}

const SHAREGPT_ROLES = new Map<string, MessageRecord["role"]>([
  ["human", "user"],
  ["user", "user"],
  ["gpt", "assistant"],
  ["assistant", "assistant"],
  ["system", "system"],
]);

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
    const role = SHAREGPT_ROLES.get(from.trim().toLowerCase()) ?? "assistant";
    const id = crypto.randomUUID();
    records.push({
      id,
      threadId,
      parentId: prevId,
      role,
      content: [{ type: "text", text: value }] as MessageRecord["content"],
      createdAt: baseTs + idx,
      metadata: { createdAtEstimated: true },
    });
    prevId = id;
    idx++;
  }
  return records;
}

function markdownToRecords(
  messages: Array<{ role: string; content: string }>,
  threadId: string,
  baseTs: number,
): MessageRecord[] {
  const records: MessageRecord[] = [];
  let prevId: string | null = null;
  let idx = 0;
  for (const { role, content } of messages) {
    if (!content.trim()) continue;
    const normalizedRole = role.trim().toLowerCase();
    const validRole =
      normalizedRole === "user" ||
      normalizedRole === "assistant" ||
      normalizedRole === "system"
        ? normalizedRole
        : normalizedRole.length > 0
          ? normalizedRole
          : "user";
    const id = crypto.randomUUID();
    records.push({
      id,
      threadId,
      parentId: prevId,
      role: validRole as MessageRecord["role"],
      content: [{ type: "text", text: content }] as MessageRecord["content"],
      createdAt: baseTs + idx,
      metadata: { createdAtEstimated: true },
    });
    prevId = id;
    idx++;
  }
  return records;
}

function csvToRecords(csvText: string, threadId: string, baseTs: number): MessageRecord[] {
  const rows = parseCsv(csvText).slice(1);
  const records: MessageRecord[] = [];
  let prevId: string | null = null;
  let idx = 0;
  for (const row of rows) {
    if (row.length < 2) continue;
    const role = row[0]?.trim().toLowerCase();
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
      metadata: { createdAtEstimated: true },
    });
    prevId = id;
    idx++;
  }
  return records;
}

export function recordToConversation(
  record: unknown,
  fallbackTitle: string,
): ParsedConversation | null {
  if (isOpenWebUIRecord(record)) {
    return openWebUIRecordToConversation(record, fallbackTitle);
  }

  if (typeof record !== "object" || record === null) return null;
  const obj = record as Record<string, unknown>;

  // Fresh ID: reusing the exported thread_id would clobber an existing thread.
  const threadId = crypto.randomUUID();
  const title = typeof obj.title === "string" ? obj.title : fallbackTitle;
  const baseTs = typeof obj.created_at === "number" ? obj.created_at : Date.now();

  let messages: MessageRecord[] = [];
  if (Array.isArray(obj.messages)) {
    messages = oaiMessagesToRecords(obj.messages, threadId, baseTs);
  } else if (Array.isArray(obj.conversations)) {
    messages = sharegptToRecords(obj.conversations, threadId, baseTs);
  }

  if (messages.length === 0) return null;
  return { title, threadId, messages };
}

export function parseImportText(
  text: string,
  filename: string,
): ParsedConversation[] {
  const basename = filename.replace(/\.[^.]+$/, "");
  if (/\.(?:md|markdown)$/i.test(filename)) {
    const baseTs = Date.now();
    return parseConversationMarkdownDocument(text, basename).flatMap(
      ({ title, messages }) => {
        const threadId = crypto.randomUUID();
        const records = markdownToRecords(messages, threadId, baseTs);
        return records.length > 0 ? [{ title, threadId, messages: records }] : [];
      },
    );
  }
  if (/\.csv$/i.test(filename)) {
    const threadId = crypto.randomUUID();
    const messages = csvToRecords(text, threadId, Date.now());
    return messages.length > 0 ? [{ title: basename, threadId, messages }] : [];
  }

  const results: ParsedConversation[] = [];
  const messageRecords: Record<string, unknown>[] = [];
  let index = 0;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    index++;
    if (isStudioChatBackup(record)) {
      results.push(...studioBackupToConversations(record, basename));
      continue;
    }
    if (isOpenAIMessageRecord(record)) {
      messageRecords.push(record);
      continue;
    }
    const parsed = recordToConversation(record, `${basename} ${index}`);
    if (parsed) results.push(parsed);
  }
  const messageConversation = messageJsonlConversationRecord(messageRecords);
  if (messageConversation) {
    const parsed = recordToConversation(messageConversation, basename);
    if (parsed) results.push(parsed);
  }
  return results;
}

async function writeConversation(
  conversation: ParsedConversation,
  projectId: string | null | undefined,
): Promise<void> {
  const { title, threadId, messages } = conversation;
  const thread: ThreadRecord = {
    id: threadId,
    title,
    modelType: "base",
    archived: conversation.archived ?? false,
    createdAt: messages[0]?.createdAt ?? conversation.createdAt ?? Date.now(),
    ...conversation.thread,
    // undefined lets the backup's grouping decide; null is an explicit Recents choice.
    projectId:
      projectId === undefined
        ? (conversation.thread?.projectId ?? null)
        : projectId,
  };
  try {
    await saveStoredChatThread(thread);
  } catch (error) {
    // Newer settings can fail strict validation; on a rejection only (not timeout/5xx), retry
    // without settings.
    const { settings, ...withoutSettings } = thread;
    const rejected =
      error instanceof ChatThreadWriteError && error.status === 422;
    if (!rejected || settings === undefined || settings === null) throw error;
    await saveStoredChatThread(withoutSettings);
  }
  try {
    await syncStoredChatMessages(threadId, messages, { pruneMissing: false });
  } catch (error) {
    // Remove the blank thread row, or the user must delete it and a retry adds another.
    await deleteStoredChatThreads([threadId]).catch(() => {});
    throw error;
  }
}

async function restoreBackupProjects(
  backup: Record<string, unknown>,
): Promise<Set<string>> {
  const projects = studioBackupProjects(backup);
  if (projects.length === 0) return new Set();
  const known = new Set(
    (await listChatProjects({ includeArchived: true })).map(({ id }) => id),
  );
  for (const project of projects) {
    if (known.has(project.id)) continue;
    try {
      await saveChatProject(project);
      known.add(project.id);
    } catch {
      // Its chats still import, ungrouped.
    }
  }
  return known;
}

export async function importConversationsFromSource(
  source: ImportSource,
  projectId?: string | null,
  options: ImportOptions = {},
): Promise<ImportResult> {
  const basename = source.name.replace(/\.[^.]+$/, "");
  const progress: ImportProgress = {
    imported: 0,
    failed: 0,
    bytesRead: 0,
    totalBytes: source.size,
  };
  const report = () => options.onProgress?.({ ...progress });
  const saved = (conversation: { threadId: string; thread?: { pairId?: string } }) =>
    options.onSaved?.(conversation.thread?.pairId ?? conversation.threadId);

  if (/\.(?:csv|md|markdown)$/i.test(source.name)) {
    const label = /\.csv$/i.test(source.name) ? "CSV" : "Markdown";
    const text = await readAllText(source, WHOLE_FILE_MAX_BYTES, label);
    for (const conversation of parseImportText(text, source.name)) {
      try {
        await writeConversation(conversation, projectId);
        progress.imported++;
        saved(conversation);
      } catch {
        progress.failed++;
      }
    }
    if (progress.imported > 0) notifyChatHistoryUpdated();
    report();
    return { imported: progress.imported, failed: progress.failed };
  }

  const inFlight = new Set<Promise<void>>();
  const messageRecords: Record<string, unknown>[] = [];
  let index = 0;
  let failure: unknown;
  let reportedBytes = 0;

  try {
    for await (const record of streamJsonRecords(source.chunks(), {
      onBytes: (bytes) => {
        progress.bytesRead += bytes;
        if (progress.bytesRead - reportedBytes >= PROGRESS_BYTES) {
          reportedBytes = progress.bytesRead;
          report();
        }
      },
      onMalformed: () => {
        progress.failed++;
      },
    })) {
      index++;
      if (isOpenAIMessageRecord(record)) {
        messageRecords.push(record);
        continue;
      }
      const conversations = isStudioChatBackup(record)
        ? studioBackupToConversations(
            record,
            basename,
            projectId === undefined
              ? await restoreBackupProjects(record).catch(() => new Set<string>())
              : undefined,
          )
        : [recordToConversation(record, `${basename} ${index}`)];

      for (const conversation of conversations) {
        if (!conversation) continue;
        const task = writeConversation(conversation, projectId)
          .then(() => {
            progress.imported++;
            saved(conversation);
          })
          .catch(() => {
            progress.failed++;
          })
          .finally(() => {
            inFlight.delete(task);
            if ((progress.imported + progress.failed) % 25 === 0) report();
          });
        inFlight.add(task);
        if (inFlight.size >= WRITE_CONCURRENCY) await Promise.race(inFlight);
      }
    }
  } catch (error) {
    failure = error;
  }

  if (failure === undefined) {
    const messageConversation = messageJsonlConversationRecord(messageRecords);
    const conversation = messageConversation
      ? recordToConversation(messageConversation, basename)
      : null;
    if (conversation) {
      try {
        await writeConversation(conversation, projectId);
        progress.imported++;
        saved(conversation);
      } catch {
        progress.failed++;
      }
    }
  }

  await Promise.allSettled(inFlight);
  // Notify even on failure, or saved chats stay hidden and a retry duplicates them.
  if (progress.imported > 0) notifyChatHistoryUpdated();
  report();

  if (failure !== undefined) {
    const reason = failure instanceof Error ? failure.message : String(failure);
    throw new Error(
      progress.imported > 0
        ? `${reason} ${progress.imported} conversation${progress.imported === 1 ? " was" : "s were"} imported before it stopped.`
        : reason,
    );
  }
  return { imported: progress.imported, failed: progress.failed };
}

export async function importConversationsFromFile(
  file: File,
  projectId?: string | null,
  options: ImportOptions = {},
): Promise<ImportResult> {
  return importConversationsFromSource(
    fileImportSource(file),
    projectId,
    options,
  );
}
