// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type { MessageRecord, ModelType, ThreadRecord } from "../types";
import type {
  AudioGenerationResponse,
  InferenceStatusResponse,
  ListLorasResponse,
  ListModelsResponse,
  LoadModelRequest,
  LoadModelResponse,
  OpenAIChatChunk,
  OpenAIChatCompletionsRequest,
  UnloadModelRequest,
  ValidateModelResponse,
} from "../types/api";

export const CHAT_HISTORY_UPDATED_EVENT = "unsloth-chat-history-updated";

export function notifyChatHistoryUpdated(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT));
  }
}

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    const formatted = formatFastApiDetail(detail);
    if (status === 405) {
      return `${formatted || "Method Not Allowed"} - the Studio backend did not accept this API method. Restart Studio so the frontend and backend are on the same build.`;
    }
    if (formatted) return formatted;
    const message = (body as { message?: unknown }).message;
    if (typeof message === "string" && message) return message;
  }
  if (status === 405) {
    return "Method Not Allowed - the Studio backend did not accept this API method. Restart Studio so the frontend and backend are on the same build.";
  }
  return `Request failed (${status})`;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(parseErrorText(response.status, body));
  }
  return body as T;
}

export async function listModels(): Promise<ListModelsResponse> {
  const response = await authFetch("/api/models/list");
  return parseJsonOrThrow<ListModelsResponse>(response);
}

export async function listLoras(
  outputsDir?: string,
): Promise<ListLorasResponse> {
  const query = outputsDir
    ? `?${new URLSearchParams({ outputs_dir: outputsDir }).toString()}`
    : "";
  const response = await authFetch(`/api/models/loras${query}`);
  return parseJsonOrThrow<ListLorasResponse>(response);
}

export async function getInferenceStatus(): Promise<InferenceStatusResponse> {
  const response = await authFetch("/api/inference/status");
  return parseJsonOrThrow<InferenceStatusResponse>(response);
}

export async function loadModel(
  payload: LoadModelRequest,
): Promise<LoadModelResponse> {
  const response = await authFetch("/api/inference/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      native_path_lease: payload.nativePathLease ?? null,
      nativePathLease: undefined,
    }),
  });
  return parseJsonOrThrow<LoadModelResponse>(response);
}

export async function validateModel(
  payload: LoadModelRequest,
): Promise<ValidateModelResponse> {
  const response = await authFetch("/api/inference/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: payload.model_path,
      native_path_lease: payload.nativePathLease ?? null,
      hf_token: payload.hf_token,
      gguf_variant: payload.gguf_variant ?? null,
      model_format: payload.model_format ?? null,
      local_files_only: payload.local_files_only ?? false,
      local_path: payload.local_path ?? null,
    }),
  });
  return parseJsonOrThrow<ValidateModelResponse>(response);
}

export async function unloadModel(payload: UnloadModelRequest): Promise<void> {
  const response = await authFetch("/api/inference/unload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await parseJsonOrThrow<unknown>(response);
}

export {
  cancelDatasetDownload,
  cancelModelDownload,
  getActiveModelDownloads,
  getDatasetDownloadProgress,
  getDatasetDownloadStatus,
  getDatasetTransportStatus,
  getDownloadProgress,
  getGgufDownloadProgress,
  getModelDownloadStatus,
  getModelTransportStatus,
  startDatasetDownload,
  startModelDownload,
  type ActiveModelDownload,
  type DownloadJobState,
  type DownloadJobStatus,
  type DownloadProgressResponse,
  type DownloadStartResult,
  type DownloadStartState,
  type TransportMarker,
  type TransportStatus,
} from "@/features/download-jobs";

export {
  addScanFolder,
  browseFolders,
  deleteCachedModel,
  invalidateGgufVariantsCache,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  listLocalModels,
  listScanFolders,
  removeScanFolder,
  type BrowseFoldersResponse,
  type CachedGgufRepo,
  type CachedModelRepo,
  type GgufVariantDetail,
  type GgufVariantsResponse,
  type LocalModelInfo,
  type LocalModelListResponse,
  type ScanFolderInfo,
} from "@/features/inventory";

export type ModelLoadPhase = "mmap" | "ready" | null;

export interface LoadProgressResponse {
  /**
   * Load phase: ``"mmap"`` while the llama-server subprocess is paging
   * weight shards into RAM, ``"ready"`` once it has reported healthy,
   * or ``null`` when no load is in flight.
   */
  phase: ModelLoadPhase;
  bytes_loaded: number;
  bytes_total: number;
  fraction: number;
}

/**
 * Fetch the active GGUF load's mmap/upload progress. Complements
 * ``getDownloadProgress`` / ``getGgufDownloadProgress`` for the window
 * between "download complete" and "chat ready", which for large MoE
 * models can be several minutes of otherwise-opaque spinning.
 */
export async function getLoadProgress(): Promise<LoadProgressResponse> {
  const response = await authFetch(`/api/inference/load-progress`);
  return parseJsonOrThrow(response);
}

export interface DeleteFineTunedModelResult {
  status: string;
  path: string;
  deleted_run_ids?: string[];
}

export async function deleteFineTunedModel(args: {
  modelPath: string;
  source: "training" | "exported";
  exportType?: "lora" | "merged" | "gguf";
  ggufVariant?: string;
  deleteRunRecord?: boolean;
}): Promise<DeleteFineTunedModelResult> {
  const response = await authFetch("/api/models/delete-finetuned", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: args.modelPath,
      source: args.source,
      export_type: args.exportType ?? null,
      gguf_variant: args.ggufVariant ?? null,
      delete_run_record: args.deleteRunRecord ?? false,
    }),
  });
  return parseJsonOrThrow<DeleteFineTunedModelResult>(response);
}

export async function listChatThreads(
  args: {
    modelType?: ModelType;
    pairId?: string;
    includeArchived?: boolean;
  } = {},
): Promise<ThreadRecord[]> {
  const params = new URLSearchParams();
  if (args.modelType) params.set("model_type", args.modelType);
  if (args.pairId) params.set("pair_id", args.pairId);
  if (args.includeArchived !== undefined) {
    params.set("include_archived", String(args.includeArchived));
  }
  const qs = params.toString();
  const response = await authFetch(`/api/chat/threads${qs ? `?${qs}` : ""}`);
  const data = await parseJsonOrThrow<{ threads: ThreadRecord[] }>(response);
  return data.threads;
}

export async function getChatThread(
  threadId: string,
): Promise<ThreadRecord | null> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}`,
  );
  if (response.status === 404) return null;
  return parseJsonOrThrow<ThreadRecord>(response);
}

export async function saveChatThread(
  thread: ThreadRecord,
): Promise<ThreadRecord> {
  const response = await authFetch("/api/chat/threads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(thread),
  });
  const savedThread = await parseJsonOrThrow<ThreadRecord>(response);
  notifyChatHistoryUpdated();
  return savedThread;
}

export async function updateChatThread(
  threadId: string,
  patch: Partial<ThreadRecord>,
): Promise<ThreadRecord> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    },
  );
  const thread = await parseJsonOrThrow<ThreadRecord>(response);
  notifyChatHistoryUpdated();
  return thread;
}

export async function deleteChatThreads(threadIds: string[]): Promise<void> {
  if (threadIds.length === 0) return;
  const response = await authFetch("/api/chat/threads", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: threadIds }),
  });
  await parseJsonOrThrow<unknown>(response);
  notifyChatHistoryUpdated();
}

export async function listChatMessages(
  threadId: string,
): Promise<MessageRecord[]> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}/messages`,
  );
  if (response.status === 404) return [];
  const data = await parseJsonOrThrow<{ messages: MessageRecord[] }>(response);
  return data.messages;
}

/**
 * Fetch messages for many threads in one HTTP call. Falls back to
 * per-thread listChatMessages on 404/405 (older servers without the
 * batch route).
 */
export async function batchListChatMessages(
  threadIds: string[],
): Promise<Map<string, MessageRecord[]>> {
  const out = new Map<string, MessageRecord[]>();
  if (threadIds.length === 0) return out;
  const response = await authFetch("/api/chat/messages:batch", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ threadIds }),
  });
  if (response.status === 404 || response.status === 405) {
    // Older server: fall back to per-thread fetches.
    const per = await Promise.all(
      threadIds.map(async (id) => [id, await listChatMessages(id)] as const),
    );
    for (const [id, msgs] of per) out.set(id, msgs);
    return out;
  }
  const data = await parseJsonOrThrow<{
    messagesByThreadId: Record<string, MessageRecord[]>;
  }>(response);
  for (const id of threadIds) {
    out.set(id, data.messagesByThreadId[id] ?? []);
  }
  return out;
}

export async function getChatMessage(
  threadId: string,
  messageId: string,
): Promise<MessageRecord | null> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}/messages/${encodeURIComponent(messageId)}`,
  );
  if (response.status === 404) return null;
  return parseJsonOrThrow<MessageRecord>(response);
}

export async function saveChatMessage(
  message: MessageRecord,
): Promise<MessageRecord> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(message.threadId)}/messages/${encodeURIComponent(message.id)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message),
    },
  );
  const savedMessage = await parseJsonOrThrow<MessageRecord>(response);
  notifyChatHistoryUpdated();
  return savedMessage;
}

export async function syncChatMessages(
  threadId: string,
  messages: MessageRecord[],
  options: { pruneMissing?: boolean } = {},
): Promise<MessageRecord[]> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}/messages`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        pruneMissing: options.pruneMissing ?? false,
      }),
    },
  );
  const data = await parseJsonOrThrow<{ messages: MessageRecord[] }>(response);
  notifyChatHistoryUpdated();
  return data.messages;
}

export async function countBackendChats(): Promise<number> {
  const response = await authFetch("/api/chat/count");
  const data = await parseJsonOrThrow<{ count: number }>(response);
  return data.count;
}

export async function clearBackendChats(
  options: { notify?: boolean } = {},
): Promise<void> {
  const response = await authFetch("/api/chat", { method: "DELETE" });
  await parseJsonOrThrow<unknown>(response);
  if (options.notify !== false) {
    notifyChatHistoryUpdated();
  }
}

export async function buildBackendChatExport(): Promise<{
  exportedAt: string;
  version: number;
  threadCount: number;
  threads: ThreadRecord[];
  messages: MessageRecord[];
}> {
  const response = await authFetch("/api/chat/export");
  return parseJsonOrThrow(response);
}

// Legacy-Dexie import ledger. The server-side source of truth that
// replaces the boolean localStorage sentinel
// (`unsloth_chat_legacy_imported_to_studio_db`) so a studio.db wipe
// makes the import recoverable.
export async function listChatImportLedger(): Promise<Set<string>> {
  const response = await authFetch("/api/chat/import-ledger");
  // Backend deployments that don't have this endpoint yet behave the
  // same as an empty ledger -- caller treats every legacy thread as
  // un-imported and tries to import. The UPSERT semantics in
  // syncChatMessages prevent duplicates, so this fallback is safe.
  if (response.status === 404 || response.status === 405) return new Set();
  const data = await parseJsonOrThrow<{ threadIds: string[] }>(response);
  return new Set(data.threadIds);
}

export interface RecordChatImportLedgerResult {
  accepted: number;
  inserted: number;
  // false when the backend predates /api/chat/import-ledger (404/405/501)
  // so the caller can avoid poisoning the localStorage perf hint -- the
  // next launch will retry the (idempotent) import.
  supported: boolean;
}

export async function recordChatImportLedger(
  threadIds: string[],
): Promise<RecordChatImportLedgerResult> {
  if (threadIds.length === 0) {
    return { accepted: 0, inserted: 0, supported: true };
  }
  const response = await authFetch("/api/chat/import-ledger", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threadIds }),
  });
  if (
    response.status === 404 ||
    response.status === 405 ||
    response.status === 501
  ) {
    return { accepted: 0, inserted: 0, supported: false };
  }
  const data = await parseJsonOrThrow<{ accepted: number; inserted: number }>(
    response,
  );
  return {
    accepted: data.accepted,
    inserted: data.inserted,
    supported: true,
  };
}

export async function listRecommendedFolders(): Promise<string[]> {
  const response = await authFetch("/api/models/recommended-folders");
  const data = await parseJsonOrThrow<{ folders: string[] }>(response);
  return data.folders;
}

function parseSseEvent(rawEvent: string): string[] {
  const dataLines: string[] = [];
  for (const line of rawEvent.split(/\r?\n/)) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  return dataLines;
}

export async function* streamChatCompletions(
  payload: OpenAIChatCompletionsRequest,
  signal: AbortSignal,
): AsyncGenerator<OpenAIChatChunk> {
  const response = await authFetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }

  if (!response.body) {
    throw new Error("Stream response missing body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    let separatorIndex = buffer.search(/\r?\n\r?\n/);
    while (separatorIndex >= 0) {
      const rawEvent = buffer.slice(0, separatorIndex);
      const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
      buffer = buffer.slice(separatorIndex + separatorLength);

      const dataLines = parseSseEvent(rawEvent);
      if (dataLines.length === 0) {
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }

      const dataText = dataLines.join("\n");
      if (dataText === "[DONE]") {
        return;
      }

      const parsed = JSON.parse(dataText) as
        | OpenAIChatChunk
        | { type?: string; content?: string; error?: { message?: string } };
      if ("error" in parsed && parsed.error) {
        throw new Error(parsed.error.message || "Stream error");
      }
      // Tool status events are custom SSE payloads, not OpenAI chunks
      if ("type" in parsed && parsed.type === "tool_status") {
        yield {
          _toolStatus: parsed.content ?? "",
        } as unknown as OpenAIChatChunk;
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }
      // Tool start/end events carry full input/output for the tool outputs panel
      if (
        "type" in parsed &&
        (parsed.type === "tool_start" || parsed.type === "tool_end")
      ) {
        yield { _toolEvent: parsed } as unknown as OpenAIChatChunk;
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }
      yield parsed as OpenAIChatChunk;
      separatorIndex = buffer.search(/\r?\n\r?\n/);
    }
  }
}

export async function generateAudio(
  payload: OpenAIChatCompletionsRequest,
  signal: AbortSignal,
): Promise<AudioGenerationResponse> {
  const response = await authFetch("/api/inference/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: false }),
    signal,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }

  return (await response.json()) as AudioGenerationResponse;
}
