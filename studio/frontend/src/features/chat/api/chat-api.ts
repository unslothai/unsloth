// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
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

function parseErrorText(status: number, body: unknown): string {
  if (
    body &&
    typeof body === "object" &&
    "detail" in body &&
    typeof body.detail === "string"
  ) {
    return body.detail;
  }
  if (
    body &&
    typeof body === "object" &&
    "message" in body &&
    typeof body.message === "string"
  ) {
    return body.message;
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

export async function listLoras(outputsDir?: string): Promise<ListLorasResponse> {
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

export interface CachedGgufRepo {
  repo_id: string;
  size_bytes: number;
  cache_path: string;
}

export async function getGgufDownloadProgress(
  repoId: string,
  variant: string,
  expectedBytes: number,
): Promise<{ downloaded_bytes: number; expected_bytes: number; progress: number }> {
  const params = new URLSearchParams({
    repo_id: repoId,
    variant,
    expected_bytes: String(expectedBytes),
  });
  const response = await authFetch(`/api/models/gguf-download-progress?${params}`);
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

export async function deleteCachedModel(repoId: string, variant?: string): Promise<void> {
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
        yield { _toolStatus: parsed.content ?? "" } as unknown as OpenAIChatChunk;
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }
      // Tool start/end events carry full input/output for the tool outputs panel
      if ("type" in parsed && (parsed.type === "tool_start" || parsed.type === "tool_end")) {
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
