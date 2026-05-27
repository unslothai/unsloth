// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type { MessageRecord, ModelType, ThreadRecord } from "../types";
import type {
  AudioGenerationResponse,
  GgufVariantsResponse,
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
import { setExtractionBackendLimit } from "../utils/extraction-queue";

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
    if (formatted) return formatted;
    const message = (body as { message?: unknown }).message;
    if (typeof message === "string" && message) return message;
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
  signal?: AbortSignal,
): Promise<LoadModelResponse> {
  const response = await authFetch("/api/inference/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      native_path_lease: payload.nativePathLease ?? null,
      nativePathLease: undefined,
    }),
    ...(signal ? { signal } : {}),
  });
  return parseJsonOrThrow<LoadModelResponse>(response);
}

export async function validateModel(
  payload: LoadModelRequest,
  signal?: AbortSignal,
): Promise<ValidateModelResponse> {
  const response = await authFetch("/api/inference/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: payload.model_path,
      native_path_lease: payload.nativePathLease ?? null,
      hf_token: payload.hf_token,
      gguf_variant: payload.gguf_variant ?? null,
      trust_remote_code: payload.trust_remote_code ?? false,
    }),
    ...(signal ? { signal } : {}),
  });
  return parseJsonOrThrow<ValidateModelResponse>(response);
}

export async function unloadModel(
  payload: UnloadModelRequest,
  signal?: AbortSignal,
): Promise<void> {
  const response = await authFetch("/api/inference/unload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    ...(signal ? { signal } : {}),
  });
  await parseJsonOrThrow<unknown>(response);
}

export interface CachedGgufRepo {
  repo_id: string;
  size_bytes: number;
  cache_path: string;
}

export async function getGgufDownloadProgress(
  repoId: string,
  variant: string,
  expectedBytes: number,
): Promise<{
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
}> {
  const params = new URLSearchParams({
    repo_id: repoId,
    variant,
    expected_bytes: String(expectedBytes),
  });
  const response = await authFetch(
    `/api/models/gguf-download-progress?${params}`,
  );
  return parseJsonOrThrow(response);
}

export interface DownloadProgressResponse {
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
  /**
   * Resolved on-disk path of the snapshot dir (or cache repo root if no
   * snapshot exists yet). Null when nothing has been written to the
   * cache for this repo.
   */
  cache_path: string | null;
}

export async function getDownloadProgress(
  repoId: string,
): Promise<DownloadProgressResponse> {
  const params = new URLSearchParams({ repo_id: repoId });
  const response = await authFetch(`/api/models/download-progress?${params}`);
  return parseJsonOrThrow(response);
}

export async function getDatasetDownloadProgress(
  repoId: string,
): Promise<DownloadProgressResponse> {
  const params = new URLSearchParams({ repo_id: repoId });
  const response = await authFetch(`/api/datasets/download-progress?${params}`);
  return parseJsonOrThrow(response);
}

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

export interface LocalModelInfo {
  id: string;
  display_name: string;
  path: string;
  source: "models_dir" | "hf_cache" | "lmstudio" | "custom";
  model_id?: string | null;
  updated_at?: number | null;
}

interface LocalModelListResponse {
  models_dir: string;
  hf_cache_dir?: string | null;
  lmstudio_dirs: string[];
  models: LocalModelInfo[];
}

export async function listLocalModels(): Promise<LocalModelListResponse> {
  const response = await authFetch("/api/models/local");
  return parseJsonOrThrow<LocalModelListResponse>(response);
}

export async function listCachedGguf(): Promise<CachedGgufRepo[]> {
  const response = await authFetch("/api/models/cached-gguf");
  const data = await parseJsonOrThrow<{ cached: CachedGgufRepo[] }>(response);
  return data.cached;
}

export interface CachedModelRepo {
  repo_id: string;
  size_bytes: number;
}

export async function listCachedModels(): Promise<CachedModelRepo[]> {
  const response = await authFetch("/api/models/cached-models");
  const data = await parseJsonOrThrow<{ cached: CachedModelRepo[] }>(response);
  return data.cached;
}

export async function deleteCachedModel(
  repoId: string,
  variant?: string,
): Promise<void> {
  const payload: Record<string, string> = { repo_id: repoId };
  if (variant) payload.variant = variant;
  const response = await authFetch("/api/models/delete-cached", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await parseJsonOrThrow<unknown>(response);
}

export async function deleteFineTunedModel(args: {
  modelPath: string;
  source: "training" | "exported";
  exportType?: "lora" | "merged" | "gguf";
  ggufVariant?: string;
}): Promise<void> {
  const response = await authFetch("/api/models/delete-finetuned", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: args.modelPath,
      source: args.source,
      export_type: args.exportType ?? null,
      gguf_variant: args.ggufVariant ?? null,
    }),
  });
  await parseJsonOrThrow<unknown>(response);
}

export interface ScanFolderInfo {
  id: number;
  path: string;
  created_at: string;
}

export async function listScanFolders(): Promise<ScanFolderInfo[]> {
  const response = await authFetch("/api/models/scan-folders");
  const data = await parseJsonOrThrow<{ folders: ScanFolderInfo[] }>(response);
  return data.folders;
}

export async function addScanFolder(path: string): Promise<ScanFolderInfo> {
  const response = await authFetch("/api/models/scan-folders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  return parseJsonOrThrow<ScanFolderInfo>(response);
}

export async function removeScanFolder(id: number): Promise<void> {
  const response = await authFetch(`/api/models/scan-folders/${id}`, {
    method: "DELETE",
  });
  await parseJsonOrThrow<unknown>(response);
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

export interface BrowseEntry {
  name: string;
  has_models: boolean;
  hidden: boolean;
}

export interface BrowseFoldersResponse {
  current: string;
  parent: string | null;
  entries: BrowseEntry[];
  suggestions: string[];
  truncated?: boolean;
  model_files_here?: number;
}

export async function listRecommendedFolders(): Promise<string[]> {
  const response = await authFetch("/api/models/recommended-folders");
  const data = await parseJsonOrThrow<{ folders: string[] }>(response);
  return data.folders;
}

export async function browseFolders(
  path?: string,
  showHidden = false,
  signal?: AbortSignal,
): Promise<BrowseFoldersResponse> {
  const params = new URLSearchParams();
  if (path !== undefined && path !== null) params.set("path", path);
  if (showHidden) params.set("show_hidden", "true");
  const qs = params.toString();
  // Forward the AbortSignal through authFetch -> fetch so that a
  // navigation cancelled in the FolderBrowser (rapid breadcrumb / row /
  // hidden-toggle clicks) actually cancels the in-flight HTTP request
  // server-side, instead of merely dropping the response client-side
  // while the backend keeps walking large directory trees.
  const response = await authFetch(
    `/api/models/browse-folders${qs ? `?${qs}` : ""}`,
    signal ? { signal } : undefined,
  );
  return parseJsonOrThrow<BrowseFoldersResponse>(response);
}

export async function listGgufVariants(
  repoId: string,
  hfToken?: string,
): Promise<GgufVariantsResponse> {
  const params = new URLSearchParams({ repo_id: repoId });
  if (hfToken) params.set("hf_token", hfToken);
  const response = await authFetch(`/api/models/gguf-variants?${params}`);
  return parseJsonOrThrow<GgufVariantsResponse>(response);
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

/** Options accepted by {@link extractDocument}. */
export interface ExtractDocumentOptions {
  describeImages?: boolean;
  /** Render full-page visual payloads for scanned PDFs when a vision model is loaded. */
  useVlmOcr?: boolean;
  /** Maximum figure/page references to list in extracted document text. */
  maxFigures?: number;
  /** Maximum extracted image payloads to keep for vision-capable sends. */
  maxVisualPayloads?: number;
  tokenBudget?: number;
}

/** Streamed progress events emitted by the extraction endpoint. */
export type ExtractDocumentProgressEvent =
  | { stage: "parsing" }
  | { stage: "done" }
  | {
      stage: "captioning";
      current: number;
      total: number;
      page: number | null;
      total_pages: number;
    };

/**
 * Upload a document (PDF / DOCX / HTML / MD / TXT) and receive
 * layout-aware Markdown plus optional figure captions produced by the
 * currently-loaded vision model. A 501 from the backend means the
 * extraction extras are not installed server-side.
 *
 * The endpoint streams NDJSON: zero or more `{stage, ...}` progress
 * events followed by a final `{stage:"result", data}` or
 * `{stage:"error", status_code, detail}` line. Pass `onProgress` to
 * receive intermediate events (e.g. captioning progress). Pass an
 * `AbortSignal` to cancel; abortion rejects with
 * `DOMException("Aborted", "AbortError")`.
 */
export function extractDocument(
  file: File,
  options: ExtractDocumentOptions = {},
  signal?: AbortSignal,
  onProgress?: (event: ExtractDocumentProgressEvent) => void,
): Promise<import("../types").ExtractedDocument> {
  const buildForm = (): FormData => {
    const form = new FormData();
    form.append("file", file, file.name);
    if (options.describeImages !== undefined) {
      form.append("describe_images", options.describeImages ? "true" : "false");
    }
    if (options.useVlmOcr !== undefined) {
      form.append("use_vlm_ocr", options.useVlmOcr ? "true" : "false");
    }
    if (options.maxFigures !== undefined) {
      form.append("max_figures", String(options.maxFigures));
    }
    if (options.maxVisualPayloads !== undefined) {
      form.append("max_visual_payloads", String(options.maxVisualPayloads));
    }
    if (options.tokenBudget !== undefined) {
      form.append("token_budget", String(options.tokenBudget));
    }
    return form;
  };

  type StreamOutcome =
    | {
        kind: "result";
        data: import("../types").ExtractedDocument;
      }
    | {
        kind: "error";
        status: number;
        detail: string;
      }
    | {
        kind: "http-error";
        status: number;
        body: unknown;
      };

  const sendOnce = async (): Promise<StreamOutcome> => {
    if (signal?.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }

    const response = await authFetch("/api/inference/chat/extract-document", {
      method: "POST",
      headers: {
        Accept: "application/x-ndjson",
      },
      body: buildForm(),
      signal,
    });

    if (!response.ok) {
      let body: unknown = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      return { kind: "http-error", status: response.status, body };
    }

    if (!response.body) {
      throw new Error("Response stream unavailable");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const handleLine = (line: string): StreamOutcome | null => {
      if (!line) return null;
      let event: { stage?: string; [key: string]: unknown };
      try {
        event = JSON.parse(line);
      } catch {
        return null;
      }
      if (event.stage === "result") {
        return {
          kind: "result",
          data: event.data as import("../types").ExtractedDocument,
        };
      }
      if (event.stage === "error") {
        return {
          kind: "error",
          status:
            typeof event.status_code === "number" ? event.status_code : 500,
          detail:
            typeof event.detail === "string" ? event.detail : "Extraction failed",
        };
      }
      onProgress?.(event as ExtractDocumentProgressEvent);
      return null;
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl = buffer.indexOf("\n");
        while (nl !== -1) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          const outcome = handleLine(line);
          if (outcome) return outcome;
          nl = buffer.indexOf("\n");
        }
      }
      const tail = buffer.trim();
      if (tail) {
        const outcome = handleLine(tail);
        if (outcome) return outcome;
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // ignore — already closed
      }
    }
    throw new Error("Extraction stream ended without a result");
  };

  return (async () => {
    let outcome: StreamOutcome;
    try {
      outcome = await sendOnce();
    } catch (err) {
      if (
        err instanceof DOMException &&
        (err.name === "AbortError" || err.message === "Aborted")
      ) {
        throw err;
      }
      throw err;
    }
    if (outcome.kind === "result") {
      return outcome.data;
    }
    if (outcome.kind === "error") {
      throw new Error(outcome.detail);
    }
    throw new Error(parseErrorText(outcome.status, outcome.body));
  })();
}

/**
 * Probe the server for document-extraction support and the currently
 * loaded model's vision capability. Polled by the Chat settings card
 * to drive the "describe figures" toggle state + tooltip.
 */
export async function getDocumentSupport(
  signal?: AbortSignal,
): Promise<import("../types").DocumentSupport> {
  const response = await authFetch("/api/inference/chat/document-support", {
    signal,
  });
  return parseJsonOrThrow<import("../types").DocumentSupport>(response);
}

const DOCUMENT_SUPPORT_TTL_MS = 30_000;
let documentSupportCache: {
  value: import("../types").DocumentSupport;
  expiresAt: number;
} | null = null;
let documentSupportInflight: Promise<
  import("../types").DocumentSupport
> | null = null;
let documentSupportCacheGeneration = 0;

function rememberDocumentSupport(
  value: import("../types").DocumentSupport,
  generation: number,
): void {
  if (generation === documentSupportCacheGeneration) {
    documentSupportCache = {
      value,
      expiresAt: Date.now() + DOCUMENT_SUPPORT_TTL_MS,
    };
    setExtractionBackendLimit(value.max_extract_concurrency);
  }
}

export function invalidateDocumentSupportCache(): void {
  documentSupportCacheGeneration += 1;
  documentSupportCache = null;
  documentSupportInflight = null;
  setExtractionBackendLimit(null);
}

export async function getCachedDocumentSupport(
  signal?: AbortSignal,
): Promise<import("../types").DocumentSupport> {
  const now = Date.now();
  if (documentSupportCache && documentSupportCache.expiresAt > now) {
    setExtractionBackendLimit(documentSupportCache.value.max_extract_concurrency);
    return documentSupportCache.value;
  }
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
  if (signal) {
    const generation = documentSupportCacheGeneration;
    const value = await getDocumentSupport(signal);
    if (!signal.aborted && generation === documentSupportCacheGeneration) {
      rememberDocumentSupport(value, generation);
    }
    return value;
  }
  if (!documentSupportInflight) {
    const generation = documentSupportCacheGeneration;
    documentSupportInflight = getDocumentSupport()
      .then((value) => {
        rememberDocumentSupport(value, generation);
        return value;
      })
      .finally(() => {
        if (generation === documentSupportCacheGeneration) {
          documentSupportInflight = null;
        }
      });
  }
  return documentSupportInflight;
}
