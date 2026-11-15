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
import { COMBINED_CSV_HEADER } from "./csv-export";
import { parseCsv } from "./csv-parse";
import {
  decodeTextChunks,
  fileImportSource,
  readAllText,
  streamJsonRecords,
  type ImportSource,
} from "./json-record-stream";
import {
  CLIENT_RUN_CALLS,
  isOpenWebUIRecord,
  openWebUIRecordToConversation,
  reasoningText,
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

// role:"tool" and role:"function" results fold into a tool-call part's `result`.
function oaiContentToParts(raw: unknown): unknown[] {
  if (!Array.isArray(raw)) {
    return typeof raw === "string" && raw.trim()
      ? [{ type: "text", text: raw }]
      : [];
  }

  return raw.flatMap((value): unknown[] => {
    if (typeof value !== "object" || value === null) return [];
    const part = value as Record<string, unknown>;
    if (
      (part.type === "text" ||
        part.type === "input_text" ||
        part.type === "output_text") &&
      typeof part.text === "string"
    ) {
      return [{ type: "text", text: part.text }];
    }
    if (part.type === "refusal" && typeof part.refusal === "string") {
      return [{ type: "text", text: part.refusal }];
    }
    if (part.type === "reasoning" && typeof part.text === "string" && part.text.trim()) {
      return [{ type: "reasoning", text: part.text }];
    }
    if (part.type === "image_url" || part.type === "input_image") {
      // Chat Completions nests {image_url:{url}}; Responses input_image is a string.
      const imageUrl =
        typeof part.image_url === "string"
          ? part.image_url
          : typeof part.image_url === "object" && part.image_url !== null
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
  // Waiting results under id:, name:, bare: (named, no id) and anon keys; cursors keep lookups linear.
  type Stashed = { result: unknown; used?: boolean };
  const buckets = new Map<string, { items: Stashed[]; start: number }>();
  const stash = (result: unknown, id?: string, name?: string) => {
    const entry: Stashed = { result };
    const keys =
      id === undefined && name === undefined
        ? ["anon"]
        : [id !== undefined && `id:${id}`, name !== undefined && `name:${name}`, id === undefined && `bare:${name}`];
    for (const key of keys) {
      if (!key) continue;
      const bucket = buckets.get(key);
      if (bucket) bucket.items.push(entry);
      else buckets.set(key, { items: [entry], start: 0 });
    }
  };
  const take = (key: string) => {
    const bucket = buckets.get(key);
    if (!bucket) return undefined;
    while (bucket.start < bucket.items.length && bucket.items[bucket.start].used) bucket.start++;
    const entry = bucket.items[bucket.start];
    if (entry) entry.used = true;
    return entry?.result;
  };
  const pending: Array<{
    turn: number;
    hasExplicitId: boolean;
    part: { toolCallId: string; toolName: string; result?: unknown };
  }> = [];
  const bindResult = (
    pred: (entry: (typeof pending)[number]) => boolean,
    result: unknown,
  ) => {
    for (let at = pending.length - 1; at >= 0; at--) {
      if (!pred(pending[at])) continue;
      const { turn } = pending[at];
      const first = pending.findIndex((entry) => entry.turn === turn && pred(entry));
      pending[first].part.result = result;
      pending.splice(first, 1);
      return true;
    }
    return false;
  };

  const records: MessageRecord[] = [];
  let prevId: string | null = null;
  let idx = 0;

  for (const m of oaiMsgs) {
    if (m === null || typeof m !== "object") continue;
    const msg = m as Record<string, unknown>;
    const role = msg.role as string;
    if (role === "tool" || role === "function") {
      const result =
        (msg as { [STRUCTURED_RESULT]?: unknown })[STRUCTURED_RESULT] ??
        (typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? ""));
      const idKey =
        typeof msg.tool_call_id === "string" ? msg.tool_call_id : undefined;
      const nameKey = typeof msg.name === "string" ? msg.name : undefined;
      // An id always wins over a name; unnamed results bind in call order within the newest turn.
      const bound =
        idKey !== undefined
          ? bindResult((entry) => entry.part.toolCallId === idKey, result)
          : nameKey !== undefined
            ? bindResult((entry) => entry.part.toolName === nameKey, result)
            : bindResult(() => true, result);
      if (!bound) stash(result, idKey, nameKey);
      continue;
    }

    const id = crypto.randomUUID();

    let content: unknown[];

    if (role === "assistant") {
      const parts = oaiContentToParts(msg.content);
      if (parts.length === 0) parts.push(...oaiContentToParts(msg.refusal));
      const argumentsText = (fn: Record<string, unknown>) =>
        typeof fn.arguments === "string"
          ? fn.arguments
          : JSON.stringify(fn.arguments ?? {});
      const parseArgs = (argsStr: string): unknown => {
        // _raw matches what the stream adapter and backend keep for invalid JSON arguments.
        try { return JSON.parse(argsStr); } catch { return { _raw: argsStr }; }
      };
      const callParts: unknown[] = [];
      const addCall = (fn: Record<string, unknown>, callId: unknown) => {
        const argsStr = argumentsText(fn);
        const part: { type: string; toolCallId: string; toolName: string; args: unknown; argsText: string; result?: unknown } = {
          type: "tool-call",
          toolCallId: typeof callId === "string" ? callId : crypto.randomUUID(),
          toolName: typeof fn.name === "string" ? fn.name : "unknown",
          args: parseArgs(argsStr),
          argsText: argsStr,
        };
        callParts.push(part);
        // An earlier result binds now (id, else id-less name, else anonymous), in arrival order.
        const early =
          (typeof callId === "string" ? take(`id:${callId}`) : undefined) ??
          take(`bare:${part.toolName}`) ??
          take("anon");
        if (early !== undefined) part.result = early;
        else pending.push({ turn: idx, part, hasExplicitId: typeof callId === "string" });
      };
      // LangChain nests both call shapes under additional_kwargs.
      const additional =
        typeof msg.additional_kwargs === "object" && msg.additional_kwargs !== null
          ? (msg.additional_kwargs as Record<string, unknown>)
          : undefined;
      const toolCalls = Array.isArray(msg.tool_calls)
        ? msg.tool_calls
        : additional !== undefined && Array.isArray(additional.tool_calls)
          ? additional.tool_calls
          : undefined;
      if (toolCalls !== undefined) {
        for (const tc of toolCalls) {
          if (tc === null || typeof tc !== "object") continue;
          const tcObj = tc as Record<string, unknown>;
          const fn = tcObj.function;
          addCall(
            (typeof fn === "object" && fn !== null ? fn : tcObj) as Record<string, unknown>,
            tcObj.id ?? tcObj.call_id,
          );
        }
      }
      const legacyCall =
        (typeof msg.function_call === "object" && msg.function_call) ||
        additional?.function_call;
      if (typeof legacyCall === "object" && legacyCall !== null) {
        const fn = legacyCall as Record<string, unknown>;
        const name = typeof fn.name === "string" ? fn.name : "unknown";
        const args = JSON.stringify(parseArgs(argumentsText(fn)));
        // Converters that upgraded the call to tool_calls sometimes leave the copy, maybe reserialized.
        const alreadyListed = callParts.some((part) => {
          const p = part as { type?: string; toolName?: string; toolCallId?: string; args?: unknown };
          return (
            p.type === "tool-call" &&
            p.toolName === name &&
            (typeof fn.id === "string" ? p.toolCallId === fn.id : JSON.stringify(p.args) === args)
          );
        });
        if (!alreadyListed) addCall(fn, fn.id);
      }
      const order = (msg as { [ITEM_ORDER]?: unknown[] })[ITEM_ORDER];
      content = order
        ? order.flatMap((item) =>
            typeof item === "number" ? callParts.slice(item, item + 1) : oaiContentToParts([item]),
          )
        : [...parts, ...callParts];
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

  // A keyed result binds by name only to a call without its own id; keyed leftovers drop.
  for (const { part, hasExplicitId } of pending) {
    if (part.result !== undefined) continue;
    part.result = take(`${hasExplicitId ? "bare" : "name"}:${part.toolName}`);
  }

  return records;
}

const SHAREGPT_ROLES = new Map<string, MessageRecord["role"]>([
  ["human", "user"],
  ["user", "user"],
  ["input", "user"],
  ["gpt", "assistant"],
  ["assistant", "assistant"],
  ["output", "assistant"],
  ["developer", "system"],
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
    if (typeof c !== "object" || c === null) continue;
    const conv = c as Record<string, unknown>;
    // role/content beats from/value (backend formatter order); an unmapped role key is skipped, not mislabeled.
    const roleKey =
      typeof conv.role === "string" && conv.role.trim() ? conv.role : "";
    const from =
      roleKey || (typeof conv.from === "string" ? conv.from : "");
    const value =
      typeof conv.content === "string" && conv.content.trim()
        ? conv.content
        : typeof conv.value === "string"
          ? conv.value
          : "";
    if (!value.trim()) continue;
    const role =
      SHAREGPT_ROLES.get(from.trim().toLowerCase()) ??
      (roleKey ? null : "assistant");
    if (role === null) continue;
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
      normalizedRole === "developer"
        ? "system"
        : normalizedRole === "user" ||
            normalizedRole === "assistant" ||
            normalizedRole === "system"
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

// A combined export starts each row with the chat's number and title; a new number starts a new chat.
function csvToConversations(csvText: string, basename: string): ParsedConversation[] {
  const parsed = parseCsv(csvText);
  const [header = [], ...rows] = parsed;
  const baseTs = Date.now();
  if (header.map((cell) => cell.trim().toLowerCase()).join(",") !== COMBINED_CSV_HEADER) {
    const threadId = crypto.randomUUID();
    const first = header[0]?.trim().toLowerCase();
    // Row 0 is data only in a headerless file, where it starts with a role.
    const body = first === "user" || first === "assistant" || first === "system" ? parsed : rows;
    const messages = csvToRecords(body, threadId, baseTs);
    return messages.length > 0 ? [{ title: basename, threadId, messages }] : [];
  }
  const chats: Array<{ title: string; rows: string[][] }> = [];
  let chatKey: string | undefined;
  for (const row of rows) {
    if (row.length < 4) continue;
    if (row[0] !== chatKey) {
      chatKey = row[0];
      chats.push({ title: row[1]?.trim() || `${basename} ${chats.length + 1}`, rows: [] });
    }
    chats[chats.length - 1].rows.push(row.slice(2));
  }
  return chats.flatMap(({ title, rows: chatRows }) => {
    const threadId = crypto.randomUUID();
    const messages = csvToRecords(chatRows, threadId, baseTs);
    return messages.length > 0 ? [{ title, threadId, messages }] : [];
  });
}

function csvToRecords(rows: string[][], threadId: string, baseTs: number): MessageRecord[] {
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

// Alpaca input follows the instruction, as in the backend's dataset formatter.
function pairToTurns(obj: Record<string, unknown>): unknown[] {
  const isAlpaca =
    typeof obj.instruction === "string" && typeof obj.output === "string";
  const isPair =
    typeof obj.prompt === "string" && typeof obj.completion === "string";
  if (!isAlpaca && !isPair) return [];
  // Trim only to test emptiness: completion prompts can end in meaningful whitespace.
  const input = typeof obj.input === "string" && obj.input.trim() ? obj.input : "";
  const userText = isAlpaca
    ? input ? `${obj.instruction}\n\n${input}` : (obj.instruction as string)
    : (obj.prompt as string);
  const assistantText = (isAlpaca ? obj.output : obj.completion) as string;
  if (!userText.trim() || !assistantText.trim()) return [];
  return [
    ...(typeof obj.system === "string" ? [{ role: "system", content: obj.system }] : []),
    { role: "user", content: userText },
    { role: "assistant", content: assistantText },
  ];
}

// Live parser's {text, images} tool result; symbol-keyed so imported JSON cannot set it.
const STRUCTURED_RESULT = Symbol("structuredResult");
// Item order of a merged Responses assistant run: content parts and tool_calls indices.
const ITEM_ORDER = Symbol("itemOrder");

function toolTurn(callId: string, output: unknown): Record<string | symbol, unknown> {
  if (typeof output === "string") return { role: "tool", tool_call_id: callId, content: output };
  const parts = (Array.isArray(output) ? output : typeof output === "object" && output !== null ? [output] : []) as Array<Record<string, unknown> | null>;
  const texts: string[] = [];
  const images: { data: string; mimeType: string }[] = [];
  const known = parts.length > 0 && parts.every((part) => {
    const source = part?.image_url ?? part?.url;
    const url =
      typeof source === "string" ? source : (source as { url?: unknown } | undefined)?.url;
    const value = typeof part?.text === "string" ? part.text : part?.logs;
    if (typeof value === "string") texts.push(value);
    else if (typeof url !== "string") return false;
    else {
      const inline = /^data:([^;,]+);base64,(.*)$/s.exec(url);
      if (inline) images.push({ mimeType: inline[1], data: inline[2] });
      else texts.push(url);
    }
    return true;
  });
  const text = known ? texts.join("\n") : JSON.stringify(output ?? "");
  return {
    role: "tool",
    tool_call_id: callId,
    content: text,
    ...(images.length > 0 ? { [STRUCTURED_RESULT]: { text, images } } : {}),
  };
}

// Studio's names for provider-run Responses tools (chat-adapter SERVER_SIDE_BUILTIN_TOOL_NAMES).
const BUILTIN_TOOL_NAMES: Record<string, string> = {
  web_search_call: "web_search",
  code_interpreter_call: "code_execution",
  shell_call: "code_execution",
};

// Adjacent assistant items share one turn, as in the backend's _normalise_responses_input.
function responsesToTurns(obj: Record<string, unknown>): unknown[] {
  // A lone typed item (a raw function_call_output line) is not a transcript.
  if (typeof obj.type === "string") return [];
  const turns: Record<string | symbol, unknown>[] = [];
  const built = new Set<Record<string | symbol, unknown>>();
  type Built = { content: unknown[]; tool_calls: unknown[]; [ITEM_ORDER]: unknown[] };
  const assistantTurn = (): Built => {
    const last = turns[turns.length - 1];
    if (last && built.has(last)) return last as unknown as Built;
    const turn = { role: "assistant", content: [], tool_calls: [], [ITEM_ORDER]: [] };
    built.add(turn);
    turns.push(turn);
    return turn;
  };
  const addContent = (...parts: unknown[]) => {
    const turn = assistantTurn();
    turn.content.push(...parts);
    turn[ITEM_ORDER].push(...parts);
  };
  // Hosted results flush after the run: mid-run they split it and replay prunes the tool-only half.
  const hosted = new Set<string>();
  const deferred: Record<string | symbol, unknown>[] = [];
  const push = (turn: Record<string | symbol, unknown>) => turns.push(...deferred.splice(0), turn);
  const text = (value: unknown) =>
    typeof value === "string" && value.trim() ? value : undefined;
  const instructions = text(obj.instructions);
  if (instructions) turns.push({ role: "system", content: instructions });
  const input = text(obj.input);
  if (input) turns.push({ role: "user", content: input });
  const items = [obj.input, obj.output].flatMap((list) =>
    Array.isArray(list)
      ? list.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
      : [],
  );
  const answered = new Set(
    items.flatMap((item) =>
      typeof item.type === "string" && item.type.endsWith("_call_output") ? [item.call_id ?? item.id] : [],
    ),
  );
  for (const entry of items) {
    const type = typeof entry.type === "string" ? entry.type : "";
    if (entry.role === "assistant") {
      addContent(
        ...(Array.isArray(entry.content)
          ? entry.content
          : text(entry.content)
            ? [{ type: "text", text: entry.content }]
            : []),
      );
    } else if (typeof entry.role === "string") {
      push(entry);
    } else if (type === "reasoning") {
      const reasoning = reasoningText(entry.summary) || reasoningText(entry.content);
      if (reasoning) addContent({ type: "reasoning", text: reasoning });
    } else if (type === "image_generation_call") {
      const encoded = text(entry.result) ?? text(entry.b64_json);
      if (encoded) {
        addContent({
          type: "image_url",
          image_url: /^(?:https?:|data:)/i.test(encoded)
            ? encoded
            : `data:image/${text(entry.output_format) ?? "png"};base64,${encoded}`,
        });
      }
    } else if (type.endsWith("_call")) {
      const id = text(entry.call_id) ?? text(entry.id) ?? crypto.randomUUID();
      const builtin = !CLIENT_RUN_CALLS.has(type);
      if (builtin) hosted.add(id);
      const commands = (entry.action as { commands?: unknown } | undefined)?.commands;
      const name =
        BUILTIN_TOOL_NAMES[type] ??
        text(entry.name) ??
        (type === "function_call" || type === "custom_tool_call" ? "unknown" : type.replace(/_call$/, ""));
      // mcp_call names its server in server_label.
      const namespace = text(entry.namespace) ?? text(entry.server_label);
      // Hosted tools use the shapes Studio's own cards render (code_execution reads command).
      const payload =
        typeof entry.code === "string"
          ? { command: entry.code }
          : type === "shell_call" && Array.isArray(commands)
            ? { kind: "bash", command: commands.join("\n") }
            : Array.isArray(entry.queries)
              ? { queries: entry.queries }
              : entry.input !== undefined
                ? { input: entry.input }
                : (entry.arguments ?? entry.action ?? {});
      const turn = assistantTurn();
      turn[ITEM_ORDER].push(turn.tool_calls.length);
      turn.tool_calls.push({
        id,
        type: "function",
        function: {
          // namespace__name and {input} wrapping match the backend.
          name: namespace ? `${namespace}__${name}` : name,
          arguments:
            typeof entry.arguments === "string"
              ? entry.arguments
              : JSON.stringify(
                  // Marked like Studio's own hosted-tool cards, so replay strips them.
                  builtin && typeof payload === "object" && payload !== null
                    ? { ...payload, _server_tool: true }
                    : payload,
                ),
        },
      });
      const done =
        entry.output ??
        (Array.isArray(entry.outputs) && entry.outputs.length > 0 ? entry.outputs : undefined) ??
        (Array.isArray(entry.results) && entry.results.length > 0 ? entry.results : undefined) ??
        entry.error ??
        (builtin && !answered.has(id) ? (text(entry.status) ?? "completed") : undefined);
      if (done != null) {
        if (builtin) deferred.push(toolTurn(id, done));
        else push(toolTurn(id, done));
      }
    } else if (type.endsWith("_call_output")) {
      // local_shell_call_output names its call in id rather than call_id.
      const callId = text(entry.call_id) ?? text(entry.id);
      if (callId === undefined) continue;
      const turn = toolTurn(callId, entry.output);
      if (hosted.has(callId)) deferred.push(turn);
      else push(turn);
    }
  }
  const output = text(obj.output);
  if (output) addContent({ type: "text", text: output });
  turns.push(...deferred);
  return turns;
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
  // Saved Response objects stamp created_at in seconds; chats use ms.
  const createdAt = typeof obj.created_at === "number" ? obj.created_at : NaN;
  const baseTs = Number.isNaN(createdAt)
    ? Date.now()
    : createdAt < 1e12
      ? createdAt * 1000
      : createdAt;

  let messages: MessageRecord[] = [];
  if (Array.isArray(obj.messages) && obj.messages.length > 0) {
    messages = oaiMessagesToRecords(obj.messages, threadId, baseTs);
  }
  if (
    messages.length === 0 &&
    Array.isArray(obj.conversations) &&
    obj.conversations.length > 0
  ) {
    messages = sharegptToRecords(obj.conversations, threadId, baseTs);
  }
  if (messages.length === 0) {
    messages = oaiMessagesToRecords(pairToTurns(obj), threadId, baseTs);
  }
  if (messages.length === 0) {
    messages = oaiMessagesToRecords(responsesToTurns(obj), threadId, baseTs);
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
    return csvToConversations(text, basename);
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
