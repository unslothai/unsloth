// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { prepareHfTokenForUse } from "@/features/hf-auth";
// These helpers are deliberately API-layer-only, not part of their features' public barrels.
// eslint-disable-next-line no-restricted-imports
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
// eslint-disable-next-line no-restricted-imports
import { isHuggingFaceOffline } from "@/features/hub/lib/network";
// eslint-disable-next-line no-restricted-imports
import { consumeNativePathToken } from "@/features/native-intents/api";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type {
  MessageRecord,
  ModelType,
  ProjectRecord,
  ThreadRecord,
} from "../types";
import type {
  ApiMonitorEntry,
  ApiMonitorResponse,
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
import {
  type GgufVariantsRequestOptions,
  ggufVariantsQuery,
  runBoundedVariantsRequest,
} from "./gguf-variants-request";
import { assertCompletedPaddedBody } from "./padded-response";

export const CHAT_HISTORY_UPDATED_EVENT = "unsloth-chat-history-updated";
export const CHAT_PROJECTS_UPDATED_EVENT = "unsloth-chat-projects-updated";

/**
* Thrown when the chat SSE stream ends without a terminal signal (`[DONE]` or a finish_reason
* chunk): the connection dropped mid-generation, surfaced as an explicit interrupted state.
*/
export class StreamInterruptedError extends Error {
  constructor() {
    super(
      "Response interrupted: the connection dropped before the model finished. " +
        "Use Retry to regenerate.",
    );
    this.name = "StreamInterruptedError";
  }
}

/**
* Thrown when a reasoning model consumes its output budget before emitting any standard
* content, so the chat UI can explain a completed stream holding only a thinking panel.
*/
export class GenerationLengthError extends Error {
  constructor() {
    super(
      "The model reached the Max Tokens limit before producing a final answer. " +
        "Increase Max Tokens or disable thinking, then retry.",
    );
    this.name = "GenerationLengthError";
  }
}

export function notifyChatHistoryUpdated(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT));
  }
}

function notifyChatProjectsUpdated(): void {
  notifyChatHistoryUpdated();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(CHAT_PROJECTS_UPDATED_EVENT));
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

/**
* `/api/inference/load` and `/unload` pad their body so a proxy cannot time the request out,
* which commits the status early: a later failure can only arrive in-band as `_deferred_error`.
*/
function deferredError(body: unknown): { status: number; message: string } | null {
  const deferred =
    body && typeof body === "object"
      ? (body as { _deferred_error?: { status_code?: unknown; detail?: unknown } })
          ._deferred_error
      : undefined;
  if (!deferred || typeof deferred !== "object") return null;
  const status =
    typeof deferred.status_code === "number" ? deferred.status_code : 500;
  return { status, message: parseErrorText(status, { detail: deferred.detail }) };
}

/**
* `paddedLabel` opts a caller into `assertCompletedPaddedBody`; only the two padded routes may,
* since a truncated body means unfinished there but is legitimate elsewhere. */
async function parseJsonOrThrow<T>(
  response: Response,
  paddedLabel?: string,
): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(parseErrorText(response.status, body));
  }
  const deferred = deferredError(body);
  if (deferred) {
    throw new Error(deferred.message);
  }
  if (paddedLabel !== undefined) {
    assertCompletedPaddedBody(body, paddedLabel);
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

export async function getApiMonitor(): Promise<ApiMonitorResponse> {
  const response = await authFetch("/api/inference/monitor");
  return parseJsonOrThrow<ApiMonitorResponse>(response);
}

export async function getApiMonitorEntry(id: string): Promise<ApiMonitorEntry> {
  const response = await authFetch(
    `/api/inference/monitor/${encodeURIComponent(id)}`,
  );
  return parseJsonOrThrow<ApiMonitorEntry>(response);
}

export async function clearApiMonitor(): Promise<void> {
  const response = await authFetch("/api/inference/monitor", {
    method: "DELETE",
  });
  await parseJsonOrThrow<{ cleared: boolean }>(response);
}

export interface ActiveGenerationsResponse {
  count: number;
  /** Conversations with a generation in flight. Shorter than `count` when a
   *  first turn started before its thread id was persisted. */
  thread_ids: string[];
  /** One entry per in-flight request. `kind` is "chat" unless it is an
   *  embeddings / completions / audio call, which has no conversation. */
  active?: { thread_id: string | null; kind?: string }[];
  parallel_slots: number;
}

/**
* Chats generating on the backend right now. Authoritative where `runningByThreadId` is not:
* that map is per-tab, empty after a reload and blind to a second tab, and /load 409s on these.
*/
export async function getActiveGenerations(): Promise<ActiveGenerationsResponse> {
  const response = await authFetch("/api/inference/active-generations");
  return parseJsonOrThrow<ActiveGenerationsResponse>(response);
}

export async function loadModel(
  payload: LoadModelRequest,
): Promise<LoadModelResponse> {
  const preparedToken = await prepareHfTokenForUse(payload.hf_token);
  // Tagged so auto-load can tell a user cancellation from a backend rejection.
  if (!preparedToken.proceed)
    throw Object.assign(new Error("Model load cancelled."), {
      unslothUserCancelled: true,
    });
  const response = await authFetch("/api/inference/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      hf_token: preparedToken.token,
      native_path_lease: payload.nativePathLease ?? null,
      nativePathLease: undefined,
    }),
  });
  return parseJsonOrThrow<LoadModelResponse>(response, "Model load");
}

export async function countChatInputTokens(payload: {
  model: string;
  messages: OpenAIChatCompletionsRequest["messages"];
  enable_thinking?: boolean;
  reasoning_effort?: OpenAIChatCompletionsRequest["reasoning_effort"];
  preserve_thinking?: boolean;
  enable_tools?: boolean;
  enabled_tools?: string[];
  mcp_enabled?: boolean;
  rag_scope?: Record<string, unknown>;
  auto_heal_tool_calls?: boolean;
  // `model` is informational: the endpoint counts with whatever is resident and reports which.
}): Promise<{ input_tokens: number; model?: string }> {
  const response = await authFetch("/api/inference/chat/count_tokens", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJsonOrThrow<{ input_tokens: number; model?: string }>(response);
}

export async function validateModel(
  payload: LoadModelRequest,
): Promise<ValidateModelResponse> {
  const preparedToken = await prepareHfTokenForUse(payload.hf_token);
  if (!preparedToken.proceed)
    throw Object.assign(new Error("Model load cancelled."), {
      unslothUserCancelled: true,
    });
  const response = await authFetch("/api/inference/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: payload.model_path,
      native_path_lease: payload.nativePathLease ?? null,
      hf_token: preparedToken.token,
      gguf_variant: payload.gguf_variant ?? null,
      // Intended load settings so validate's preflight matches the follow-up /load.
      max_seq_length: payload.max_seq_length,
      load_in_4bit: payload.load_in_4bit,
      cache_type_kv: payload.cache_type_kv ?? null,
      tensor_parallel: payload.tensor_parallel ?? false,
      gpu_ids: payload.gpu_ids,
      // Manual placement is an explicit override: Auto layers use llama.cpp --fit, a pinned
      // layer count is owned by the user. Tell validate so it applies the same policy as /load.
      gpu_memory_mode: payload.gpu_memory_mode,
      // Only 0 changes the verdict: a zero-layer DiffusionGemma split places no layers.
      gpu_layers: payload.gpu_layers,
      // Slots scale the KV estimate; keep validate sized like the load.
      n_parallel: payload.n_parallel,
      // The estimate charges a drafter whose size differs by kind (a DSpark
      // sidecar is ~11 GB), so omitting the mode makes this preflight disagree
      // with /load in both directions.
      speculative_type: payload.speculative_type ?? null,
      spec_draft_n_max: payload.spec_draft_n_max ?? null,
    }),
  });
  return parseJsonOrThrow<ValidateModelResponse>(response);
}

/**
* Read a GGUF's header dims (native context length, layer count, MoE expert-layer count) from
* its local file (no GPU load, no download). All null when the file isn't downloaded, isn't a
* GGUF, or is gated. For a native (drag-drop) file, pass `nativePathToken`. Used by the
* deferred-load staging flow to size the context, GPU-layers and MoE sliders before the load.
*/
export async function fetchGgufStagedMetadata(payload: {
  model_path: string;
  gguf_variant?: string | null;
  hf_token?: string | null;
  nativePathToken?: string | null;
}): Promise<{
  contextLength: number | null;
  layerCount: number | null;
  moeLayerCount: number | null;
  isDiffusion: boolean;
  /** Unclassifiable, so `isDiffusion: false` above means "not known to be diffusion":
   *  callers picking a GPU split must assume possibly-diffusion. */
  diffusionUnknown: boolean;
}> {
  let nativePathLease: string | null = null;
  if (payload.nativePathToken) {
    try {
      nativePathLease = (
        await consumeNativePathToken(payload.nativePathToken, "validate-model")
      ).nativePathLease;
    } catch {
      // Lease expired / revoked: degrade to no metadata (the load can re-mint). Nothing was read, so diffusion is unknown, not false.
      return {
        contextLength: null,
        layerCount: null,
        moeLayerCount: null,
        isDiffusion: false,
        diffusionUnknown: true,
      };
    }
  }
  const response = await authFetch("/api/inference/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: payload.model_path,
      gguf_variant: payload.gguf_variant ?? null,
      hf_token: payload.hf_token ?? null,
      native_path_lease: nativePathLease,
      include_context_length: true,
    }),
  });
  const res = await parseJsonOrThrow<ValidateModelResponse>(response);
  return {
    contextLength: res.context_length ?? null,
    layerCount: res.layer_count ?? null,
    moeLayerCount: res.moe_layer_count ?? null,
    isDiffusion: res.is_diffusion ?? false,
    // Absent on a pre-#7575 backend, which never reported the inconclusive case.
    diffusionUnknown: res.diffusion_unknown ?? false,
  };
}

export async function unloadModel(payload: UnloadModelRequest): Promise<void> {
  const response = await authFetch("/api/inference/unload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await parseJsonOrThrow<unknown>(response, "Model unload");
}

/**
* Allow or deny a tool call paused awaiting user confirmation. Identified by the backend
* ``approvalId`` echoed in the tool_start event; ``sessionId`` is a scope check. Resolves to
* ``true`` only when the backend matched a pending call, so a stale post can offer a retry.
*/
export async function resolveToolConfirmation(
  sessionId: string,
  approvalId: string,
  decision: "allow" | "deny",
): Promise<boolean> {
  const response = await authFetch("/api/inference/tool-confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      approval_id: approvalId,
      decision,
    }),
  });
  const parsed = await parseJsonOrThrow<{ resolved?: boolean }>(response);
  return parsed.resolved === true;
}

export interface CachedGgufRepo {
  repo_id: string;
  load_id?: string | null;
  size_bytes: number;
  cache_path: string;
  /** Epoch seconds of the newest downloaded quant; sorts Downloaded
   * newest-first. Optional for older-backend compatibility. */
  last_modified?: number;
  /** True when the repo ships an mmproj adapter (image inputs). Optional for
   * older-backend compatibility. */
  has_vision?: boolean;
  /** HF pipeline task inferred from the GGUF architecture ("text-to-image" for diffusion), so the Images picker can show only diffusion GGUFs. Optional for older backends. */
  task?: string | null;
  /** True when some quant has a download manifest or cancel marker. Optional
   * for older-backend compatibility. */
  has_variant_state?: boolean;
  partial?: boolean;
  capabilities?: CachedRepoCapabilities | null;
}

/** The subset of a row's capabilities auto-load acts on; Hub view models have a wider type. */
export interface CachedRepoCapabilities {
  can_chat?: boolean;
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
  /**
   * Finalized-blob bytes only. Bytes still landing in a `.incomplete` blob count
   * toward `downloaded_bytes` but not here, so the two are equal exactly when
   * nothing is in flight.
   */
  completed_bytes: number;
  expected_bytes: number;
  progress: number;
  /**
  * On-disk path of the snapshot dir (or cache repo root if no snapshot yet); null when
  * nothing has been written to the cache for this repo. */
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
  const response = await authFetch(
    `/api/hub/datasets/download-progress?${params}`,
  );
  return parseJsonOrThrow(response);
}

export type ModelLoadPhase = "mmap" | "ready" | null;

export interface LoadProgressResponse {
  /**
  * Load phase: "mmap" while llama-server pages weight shards into RAM, "ready" once
  * healthy, or null when no load is in flight. */
  phase: ModelLoadPhase;
  bytes_loaded: number;
  bytes_total: number;
  fraction: number;
}

/**
* Fetch the active GGUF load's mmap/upload progress. Complements the download progress
* endpoints for the "download complete" -> "chat ready" window, minutes for large MoE models.
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
  // Backend-detected weights format ("gguf" when known), for folders whose name lacks -GGUF.
  model_format?: string | null;
  // Set when a cached snapshot holds an incomplete download, so consumers skip unloadable weights.
  partial?: boolean;
  updated_at?: number | null;
  // HF pipeline task inferred from the GGUF architecture, so the Images picker can filter to diffusion ("text-to-image"). Optional for older backends.
  task?: string | null;
}

interface LocalModelListResponse {
  models_dir: string;
  hf_cache_dir?: string | null;
  lmstudio_dirs: string[];
  models: LocalModelInfo[];
}

export async function listLocalModels(
  signal?: AbortSignal,
): Promise<LocalModelListResponse> {
  const response = await authFetch("/api/models/local", { signal });
  return parseJsonOrThrow<LocalModelListResponse>(response);
}

export async function listCachedGguf(
  signal?: AbortSignal,
): Promise<CachedGgufRepo[]> {
  const response = await authFetch("/api/hub/cached-gguf", { signal });
  const data = await parseJsonOrThrow<{ cached: CachedGgufRepo[] }>(response);
  return data.cached;
}

export interface CachedModelRepo {
  repo_id: string;
  load_id?: string | null;
  size_bytes: number;
  /** Epoch seconds of the newest downloaded weight file; sorts Downloaded
   * newest-first. Optional for older-backend compatibility. */
  last_modified?: number;
  /** HF pipeline task: "text-to-image" for a cached diffusers pipeline repo (model_index.json present), so the chat picker can hide it. Absent = chat. */
  task?: string | null;
  /** True when the snapshot is incomplete (a cancelled/partial download): such a repo must not count as downloaded, or a click re-downloads the full weights. */
  partial?: boolean;
  /** True for a diffusion repo with no model_index.json: a single-file checkpoint loadable only via from_single_file, so task pickers must not offer it as a pipeline load unless the curated catalog carries its artifact. */
  single_file?: boolean;
  /** True for an sd.cpp companion mirror (VAE / text encoders, no denoiser): listed so it can be seen and deleted, never offered as a load. */
  companion?: boolean;
  /** Owning cache dir; sent so a delete targets this copy, not the active
   * cache. Optional for older-backend compatibility. */
  cache_path?: string | null;
  capabilities?: CachedRepoCapabilities | null;
}

export async function listCachedModels(
  hfToken?: string | null,
  signal?: AbortSignal,
): Promise<CachedModelRepo[]> {
  const response = await authFetch("/api/hub/cached-models", {
    headers: hubTokenHeader(hfToken),
    signal,
  });
  const data = await parseJsonOrThrow<{ cached: CachedModelRepo[] }>(response);
  return data.cached;
}

export interface CachedModelPath {
  path: string;
  is_dir: boolean;
}

/** Absolute on-disk path of a cached repo or one of its GGUF variants. */
export async function getCachedModelPath(
  repoId: string,
  variant?: string,
): Promise<CachedModelPath> {
  const params = new URLSearchParams({ repo_id: repoId });
  if (variant) params.set("variant", variant);
  const response = await authFetch(
    `/api/models/cached-model-path?${params.toString()}`,
  );
  return parseJsonOrThrow<CachedModelPath>(response);
}

/** Reveal a cached repo (or one GGUF variant's file) in the OS file manager. */
export async function revealCachedModel(
  repoId: string,
  variant?: string,
): Promise<void> {
  const payload: Record<string, string> = { repo_id: repoId };
  if (variant) payload.variant = variant;
  const response = await authFetch("/api/models/reveal-cached-model", {
    method: "POST",
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
    projectId?: string | null;
    includeArchived?: boolean;
  } = {},
): Promise<ThreadRecord[]> {
  const params = new URLSearchParams();
  if (args.modelType) params.set("model_type", args.modelType);
  if (args.pairId) params.set("pair_id", args.pairId);
  if (args.projectId) params.set("project_id", args.projectId);
  if (args.includeArchived !== undefined) {
    params.set("include_archived", String(args.includeArchived));
  }
  const qs = params.toString();
  const response = await authFetch(`/api/chat/threads${qs ? `?${qs}` : ""}`);
  const data = await parseJsonOrThrow<{ threads: ThreadRecord[] }>(response);
  // Always hand back an array: an older or misbehaving backend may omit it or send a non-array.
  return Array.isArray(data.threads) ? data.threads : [];
}

/** One chat message attachment, as listed for the settings uploaded-files view. */
export interface ChatAttachmentRecord {
  id: string;
  messageId: string;
  threadId: string;
  pairId?: string | null;
  threadTitle?: string | null;
  name: string;
  type?: string | null;
  contentType?: string | null;
  sizeBytes?: number | null;
  createdAt?: number | null;
}

export interface ChatAttachmentPage {
  attachments: ChatAttachmentRecord[];
  nextOffset: number | null;
}

export async function listChatAttachments(
  offset = 0,
  limit = 50,
): Promise<ChatAttachmentPage> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  const response = await authFetch(`/api/chat/attachments?${params}`);
  const data = await parseJsonOrThrow<{
    attachments: ChatAttachmentRecord[];
    nextOffset: number | null;
  }>(response);
  return {
    attachments: Array.isArray(data.attachments) ? data.attachments : [],
    nextOffset:
      typeof data.nextOffset === "number" && Number.isFinite(data.nextOffset)
        ? data.nextOffset
        : null,
  };
}

/** Stored attachment content (image bytes or extracted text) as a Blob. */
export async function fetchChatAttachmentBlob(
  messageId: string,
  attachmentId: string,
): Promise<Blob> {
  const response = await authFetch(
    `/api/chat/attachments/${encodeURIComponent(messageId)}/${encodeURIComponent(attachmentId)}/file`,
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }
  return response.blob();
}

export async function deleteChatAttachment(
  messageId: string,
  attachmentId: string,
): Promise<void> {
  const response = await authFetch(
    `/api/chat/attachments/${encodeURIComponent(messageId)}/${encodeURIComponent(attachmentId)}`,
    { method: "DELETE" },
  );
  await parseJsonOrThrow<{ ok: boolean }>(response);
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

export interface UpdateChatThreadOptions {
  /** Apply only while the row still holds this title, else 409. */
  expectedTitle?: string;
  /** And only while this is still the thread's opening user message. */
  expectedOpeningMessageId?: string;
}

export async function updateChatThread(
  threadId: string,
  patch: Partial<ThreadRecord>,
  options: UpdateChatThreadOptions = {},
): Promise<ThreadRecord> {
  const body: Record<string, unknown> = { ...patch };
  if (options.expectedTitle !== undefined) {
    body.expectedTitle = options.expectedTitle;
  }
  if (options.expectedOpeningMessageId !== undefined) {
    body.expectedOpeningMessageId = options.expectedOpeningMessageId;
  }
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  const thread = await parseJsonOrThrow<ThreadRecord>(response);
  notifyChatHistoryUpdated();
  return thread;
}

export interface ForkChatThreadResult {
  thread: ThreadRecord;
  messages: MessageRecord[];
  containerSnapshotWarning: string | null;
}

export async function forkChatThread(
  threadId: string,
  args: { messageId: string; newThreadId: string; createdAt: number },
): Promise<ForkChatThreadResult> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}/fork`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(args),
    },
  );
  const data = await parseJsonOrThrow<{
    thread: ThreadRecord;
    messages: MessageRecord[];
    containerSnapshotWarning: string | null;
  }>(response);
  notifyChatHistoryUpdated();
  return data;
}

export async function getForkCount(
  threadId: string,
  messageId: string,
): Promise<number> {
  const response = await authFetch(
    `/api/chat/threads/${encodeURIComponent(threadId)}/messages/${encodeURIComponent(messageId)}/forks`,
  );
  if (response.status === 404) return 0;
  const data = await parseJsonOrThrow<{ count: number }>(response);
  return data.count;
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

export async function listChatProjects(
  args: { includeArchived?: boolean } = {},
): Promise<ProjectRecord[]> {
  const params = new URLSearchParams();
  if (args.includeArchived !== undefined) {
    params.set("include_archived", String(args.includeArchived));
  }
  const qs = params.toString();
  const response = await authFetch(`/api/chat/projects${qs ? `?${qs}` : ""}`);
  const data = await parseJsonOrThrow<{ projects: ProjectRecord[] }>(response);
  // Always hand back an array: an older or misbehaving backend may omit it or send a non-array.
  return Array.isArray(data.projects) ? data.projects : [];
}

export async function getChatProject(
  projectId: string,
): Promise<ProjectRecord | null> {
  const response = await authFetch(
    `/api/chat/projects/${encodeURIComponent(projectId)}`,
  );
  if (response.status === 404) return null;
  return parseJsonOrThrow<ProjectRecord>(response);
}

export async function saveChatProject(
  project: ProjectRecord,
): Promise<ProjectRecord> {
  const response = await authFetch("/api/chat/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(project),
  });
  const saved = await parseJsonOrThrow<ProjectRecord>(response);
  notifyChatProjectsUpdated();
  return saved;
}

export async function updateChatProject(
  projectId: string,
  patch: Partial<ProjectRecord>,
): Promise<ProjectRecord> {
  const response = await authFetch(
    `/api/chat/projects/${encodeURIComponent(projectId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    },
  );
  const project = await parseJsonOrThrow<ProjectRecord>(response);
  notifyChatProjectsUpdated();
  return project;
}

export async function deleteChatProject(
  projectId: string,
  args: { deleteFiles?: boolean } = {},
): Promise<void> {
  const params = new URLSearchParams();
  if (args.deleteFiles) params.set("delete_files", "true");
  const qs = params.toString();
  const response = await authFetch(
    `/api/chat/projects/${encodeURIComponent(projectId)}${qs ? `?${qs}` : ""}`,
    { method: "DELETE" },
  );
  await parseJsonOrThrow<ProjectRecord>(response);
  notifyChatProjectsUpdated();
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
* Fetch messages for many threads in one HTTP call. Falls back to per-thread
* listChatMessages on 404/405 (older servers without the batch route). */
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
  projects?: ProjectRecord[];
  threads: ThreadRecord[];
  messages: MessageRecord[];
}> {
  const response = await authFetch("/api/chat/export");
  return parseJsonOrThrow(response);
}

// Legacy-Dexie import ledger: a server-side source of truth replacing the localStorage
// sentinel, so a studio.db wipe keeps the import recoverable.
export async function listChatImportLedger(): Promise<Set<string>> {
  const response = await authFetch("/api/chat/import-ledger");
  // Backends without this endpoint behave like an empty ledger -- the caller re-imports every
  // legacy thread, and syncChatMessages UPSERTs prevent duplicates.
  if (response.status === 404 || response.status === 405) return new Set();
  const data = await parseJsonOrThrow<{ threadIds: string[] }>(response);
  return new Set(data.threadIds);
}

export interface RecordChatImportLedgerResult {
  accepted: number;
  inserted: number;
  // false when the backend predates /api/chat/import-ledger, so the caller does not poison the
  // localStorage perf hint; the next launch retries the (idempotent) import.
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
  // Forward the AbortSignal through authFetch so a cancelled FolderBrowser navigation also cancels the server-side walk.
  const response = await authFetch(
    `/api/models/browse-folders${qs ? `?${qs}` : ""}`,
    signal ? { signal } : undefined,
  );
  return parseJsonOrThrow<BrowseFoldersResponse>(response);
}

export async function listGgufVariants(
  repoId: string,
  hfToken?: string,
  options?: GgufVariantsRequestOptions,
): Promise<GgufVariantsResponse> {
  const params = ggufVariantsQuery(repoId, options, isHuggingFaceOffline());
  return runBoundedVariantsRequest(options?.signal, async (signal) => {
    const response = await authFetch(`/api/models/gguf-variants?${params}`, {
      headers: hubTokenHeader(hfToken),
      signal,
    });
    return parseJsonOrThrow<GgufVariantsResponse>(response);
  });
}

export interface KvCacheEstimate {
  kv_bytes: number | null;
  weights_bytes: number | null;
  native_context: number | null;
}

/** Estimate KV cache + weight bytes for a downloaded quant at a context length,
 * for the load dialog's memory warning. */
export async function estimateKvCache(
  repoId: string,
  quant: string,
  nCtx: number,
  cacheTypeKv?: string | null,
  signal?: AbortSignal,
): Promise<KvCacheEstimate> {
  const params = new URLSearchParams({
    repo_id: repoId,
    quant,
    n_ctx: String(nCtx),
  });
  if (cacheTypeKv) params.set("cache_type_kv", cacheTypeKv);
  const response = await authFetch(
    `/api/models/kv-cache-estimate?${params}`,
    signal ? { signal } : undefined,
  );
  return parseJsonOrThrow<KvCacheEstimate>(response);
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

function hasNonWhitespaceText(value: unknown): boolean {
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.some((item) => hasNonWhitespaceText(item));
  }
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  return ["thinking", "text", "content", "reasoning", "summary"].some(
    (key) => key in record && hasNonWhitespaceText(record[key]),
  );
}

function classifyStructuredDeltaContent(content: unknown): {
  hasAssistantContent: boolean;
  hasReasoningContent: boolean;
} {
  if (typeof content === "string") {
    return {
      hasAssistantContent: hasNonWhitespaceText(content),
      hasReasoningContent: false,
    };
  }
  if (!Array.isArray(content)) {
    return {
      hasAssistantContent: false,
      hasReasoningContent: false,
    };
  }

  let hasAssistantContent = false;
  let hasReasoningContent = false;
  for (const part of content) {
    if (typeof part === "string") {
      hasAssistantContent ||= hasNonWhitespaceText(part);
      continue;
    }
    if (!part || typeof part !== "object") {
      continue;
    }
    const record = part as Record<string, unknown>;
    if (record.type === "thinking" || record.type === "reasoning") {
      hasReasoningContent ||= hasNonWhitespaceText(record);
    } else if (record.type === "text" || record.type === "output_text") {
      const text =
        typeof record.text === "string" ? record.text : record.content;
      hasAssistantContent ||= hasNonWhitespaceText(text);
    }
  }
  return { hasAssistantContent, hasReasoningContent };
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
  let completed = false;
  // EOF without `[DONE]` or a finish_reason chunk means the stream was cut mid-generation.
  let sawTerminalSignal = false;
  let terminalFinishReason: string | null = null;
  let sawAssistantContent = false;
  let sawReasoningContent = false;

  const throwIfReasoningOnlyLength = () => {
    if (
      terminalFinishReason === "length" &&
      sawReasoningContent &&
      !sawAssistantContent
    ) {
      throw new GenerationLengthError();
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        completed = true;
        if (!sawTerminalSignal) {
          throw new StreamInterruptedError();
        }
        throwIfReasoningOnlyLength();
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
          completed = true;
          sawTerminalSignal = true;
          throwIfReasoningOnlyLength();
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
        // Diffusion frame: a per-step canvas snapshot. Custom SSE payload (not an OpenAI chunk) with
        // no assistant text, surfaced as a transient marker for the in-place renderer, never the transcript.
        if ("type" in parsed && parsed.type === "diffusion_frame") {
          yield {
            _diffusionFrame: parsed,
          } as unknown as OpenAIChatChunk;
          separatorIndex = buffer.search(/\r?\n\r?\n/);
          continue;
        }
        // tool_start/end carry full input/output; tool_output streams
        // incremental stdout and tool_args streams the call arguments live.
        if (
          "type" in parsed &&
          (parsed.type === "tool_start" ||
            parsed.type === "tool_end" ||
            parsed.type === "tool_output" ||
            parsed.type === "tool_args")
        ) {
          yield { _toolEvent: parsed } as unknown as OpenAIChatChunk;
          separatorIndex = buffer.search(/\r?\n\r?\n/);
          continue;
        }
        // Relay server-side reasoning duration.
        if (
          parsed &&
          typeof parsed === "object" &&
          "type" in parsed &&
          parsed.type === "reasoning_summary"
        ) {
          yield {
            _reasoningDurationMs: (parsed as { duration_ms?: number })
              .duration_ms,
          } as unknown as OpenAIChatChunk;
          separatorIndex = buffer.search(/\r?\n\r?\n/);
          continue;
        }
        // finish_reason is a valid terminal signal for providers that close without a [DONE] sentinel.
        const parsedChoices = (
          parsed as {
            choices?: Array<{
              delta?: Record<string, unknown>;
              finish_reason?: string | null;
            }>;
          }
        ).choices;
        for (const choice of parsedChoices ?? []) {
          const delta = choice.delta;
          if (delta) {
            const contentState = classifyStructuredDeltaContent(delta.content);
            sawAssistantContent ||= contentState.hasAssistantContent;
            sawReasoningContent ||= contentState.hasReasoningContent;
            const reasoning =
              delta.reasoning_content ??
              delta.reasoning ??
              delta.reasoning_details;
            sawReasoningContent ||= hasNonWhitespaceText(reasoning);
          }
          if (choice.finish_reason) {
            terminalFinishReason = choice.finish_reason;
          }
        }
        const finishReason = parsedChoices?.[0]?.finish_reason;
        if (finishReason) {
          sawTerminalSignal = true;
        }
        yield parsed as OpenAIChatChunk;
        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }
    }
  } finally {
    // Only abort on an early/abnormal exit: after a natural [DONE] (or server EOF) the request is logically complete and the
    // backend finalizes its api-monitor entry, so cancelling here can mark a successful request as cancelled.
    if (!completed) {
      try {
        await reader.cancel();
      } catch {
        // already closed
      }
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
