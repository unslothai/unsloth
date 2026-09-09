// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Imports Open WebUI JSON arrays, OpenAI/ShareGPT JSONL, and role/content CSV. JSON records
 *  stream individually so large exports never become one JS string. */

import { notifyChatHistoryUpdated } from "../api/chat-api";
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

/** CSV has no record framing to stream on, so it is still read whole. */
const CSV_MAX_BYTES = 64 * 1024 * 1024;

/** Matches MAX_CHAT_IMPORT_CHUNK_BYTES in src-tauri/src/native_file_dialogs.rs. */
const NATIVE_CHUNK_BYTES = 8 * 1024 * 1024;

/** Limit concurrent writes for exports that may contain thousands of chats. */
const WRITE_CONCURRENCY = 6;

/** Report progress at least this often by bytes as well as by conversations. A count-only
 *  cadence leaves the toast reading "0 so far (0%)" for the entire read of an export made of
 *  a few very large chats, which is the case the toast exists for. */
const PROGRESS_BYTES = 4 * 1024 * 1024;

export interface ImportProgress {
  /** Conversations written so far. */
  imported: number;
  /** Conversations whose write failed; the import continues past them. */
  failed: number;
  bytesRead: number;
  totalBytes?: number;
}

export interface ImportOptions {
  onProgress?: (progress: ImportProgress) => void;
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
      // Never past the size the picker recorded: bytes appended to a file still being written are not
      // part of the export that was chosen.
      Math.min(NATIVE_CHUNK_BYTES, handle.size - offset),
    );
    // The picker recorded the size, so a short read means the file shrank since then. Stopping
    // quietly would pass a partial export off as the whole one.
    if (bytes.byteLength === 0) {
      throw new Error(
        `${handle.name} ended after ${offset} of ${handle.size} bytes; it changed after it was picked.`,
      );
    }
    offset += bytes.byteLength;
    yield bytes;
  }
}

/** Desktop: the file stays on disk and is redeemed one range at a time. */
export function nativeImportSource(handle: {
  name: string;
  size: number;
  token: string;
}): ImportSource {
  return {
    name: handle.name,
    size: handle.size,
    // Decoding is fatal here because the native reader it replaced rejected invalid UTF-8 outright
    // rather than saving a chat full of U+FFFD.
    chunks: () => decodeTextChunks(nativeBytes(handle), true),
  };
}

// Record conversion

// role:"tool" results are absorbed into the preceding assistant tool-call part's `result` field
// rather than becoming separate records.
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
      const parts = oaiContentToParts(msg.content);
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          const tcObj = tc as Record<string, unknown>;
          const fn = (tcObj.function as Record<string, unknown>) ?? {};
          const tcId = typeof tcObj.id === "string" ? tcObj.id : crypto.randomUUID();
          const name = typeof fn.name === "string" ? fn.name : "unknown";
          const argsStr = typeof fn.arguments === "string" ? fn.arguments : "{}";
          let args: unknown = {};
          // _raw matches what the stream adapter and the backend keep for arguments the model did not
          // emit as valid JSON.
          try { args = JSON.parse(argsStr); } catch { args = { _raw: argsStr }; }
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
  // parseCsv handles quoted newlines, so multi-line message content round-trips from the exporter.
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
    });
    prevId = id;
    idx++;
  }
  return records;
}

/** One streamed record -> one conversation, or null when it holds no messages. */
export function recordToConversation(
  record: unknown,
  fallbackTitle: string,
): ParsedConversation | null {
  if (isOpenWebUIRecord(record)) {
    return openWebUIRecordToConversation(record, fallbackTitle);
  }

  if (typeof record !== "object" || record === null) return null;
  const obj = record as Record<string, unknown>;

  // Fresh ID: reusing the exported thread_id would clobber an existing thread on import.
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

/** Kept for the CSV path and for callers that already hold the whole text. */
export function parseImportText(
  text: string,
  filename: string,
): ParsedConversation[] {
  const basename = filename.replace(/\.[^.]+$/, "");
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

// Persistence

async function writeConversation(
  conversation: ParsedConversation,
  projectId: string | null,
): Promise<void> {
  const { title, threadId, messages } = conversation;
  const thread: ThreadRecord = {
    id: threadId,
    title,
    modelType: "base",
    projectId: projectId ?? null,
    archived: conversation.archived ?? false,
    createdAt: messages[0]?.createdAt ?? conversation.createdAt ?? Date.now(),
  };
  await saveStoredChatThread(thread);
  try {
    await syncStoredChatMessages(threadId, messages, { pruneMissing: false });
  } catch (error) {
    // The thread row is already in the sidebar. Left behind it is a blank chat the user has to
    // delete by hand, and a retry adds another one.
    await deleteStoredChatThreads([threadId]).catch(() => {});
    throw error;
  }
}

export async function importConversationsFromSource(
  source: ImportSource,
  projectId: string | null = null,
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

  if (/\.csv$/i.test(source.name)) {
    const text = await readAllText(source, CSV_MAX_BYTES, "CSV");
    for (const conversation of parseImportText(text, source.name)) {
      await writeConversation(conversation, projectId);
      progress.imported++;
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
      // Count a malformed record and continue with the rest of the export.
      onMalformed: () => {
        progress.failed++;
      },
    })) {
      index++;
      if (isOpenAIMessageRecord(record)) {
        messageRecords.push(record);
        continue;
      }
      const conversation = recordToConversation(record, `${basename} ${index}`);
      if (!conversation) continue;

      const task = writeConversation(conversation, projectId)
        .then(() => {
          progress.imported++;
        })
        .catch(() => {
          // Keep importing after one conversation fails to save.
          progress.failed++;
        })
        .finally(() => {
          inFlight.delete(task);
          if ((progress.imported + progress.failed) % 25 === 0) report();
        });
      inFlight.add(task);
      if (inFlight.size >= WRITE_CONCURRENCY) await Promise.race(inFlight);
    }
  } catch (error) {
    // A read that dies partway still leaves earlier chats saved.
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
      } catch {
        progress.failed++;
      }
    }
  }

  await Promise.allSettled(inFlight);
  // Those chats have to reach the sidebar even when the read failed, or the UI stays empty until
  // a reload and a retry duplicates every one of them.
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
  projectId: string | null = null,
  options: ImportOptions = {},
): Promise<ImportResult> {
  return importConversationsFromSource(
    fileImportSource(file),
    projectId,
    options,
  );
}
