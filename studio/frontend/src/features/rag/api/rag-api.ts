// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import { readSseJsonEvents } from "@/lib/sse-json-events";
import type {
  DocumentUploadResult,
  FolderSyncJob,
  FolderSyncJobEvent,
  IndexJob,
  JobEvent,
  KnowledgeBase,
  LinkedFolder,
  LinkedFolderScope,
  PreviewTarget,
  RagDocument,
  UploadedDocument,
} from "../types/rag";
import { noteRagAvailability, noteRagResponse } from "./rag-availability";

const RAG_BASE = "/api/rag";

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const { detail, message } = body as { detail?: unknown; message?: unknown };
    const formatted = formatFastApiDetail(detail);
    if (formatted) return formatted;
    if (typeof message === "string" && message) return message;
  }
  return `Request failed (${status})`;
}

async function ragRequest<T>(
  path: string,
  init?: { method?: string; body?: object },
): Promise<T> {
  const response = await authFetch(`${RAG_BASE}${path}`, {
    method: init?.method,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  if (response.status === 204) {
    noteRagResponse(204, null);
    return undefined as T;
  }
  const json = await response.json().catch(() => null);
  // Every RAG endpoint but the list gates on the extension loading, so its status is
  // also an availability answer. See api/rag-availability.
  noteRagResponse(response.status, json);
  if (!response.ok) throw new Error(parseErrorText(response.status, json));
  return json as T;
}

/** A desktop drop the webview can only name through a Rust-signed grant. */
export interface NativeUploadRef {
  nativePathLease: string;
}

export type UploadSource = File | NativeUploadRef;

async function ragUpload(
  path: string,
  source: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  const form = new FormData();
  if (source instanceof File) form.append("file", source);
  else form.append("nativePathLease", source.nativePathLease);
  // Per-upload overrides for the vision passes; omitted -> backend config default.
  if (ocr !== undefined) form.append("ocr", String(ocr));
  if (caption !== undefined) form.append("caption", String(caption));
  // No Content-Type: let the browser set the multipart boundary.
  const response = await authFetch(`${RAG_BASE}${path}`, {
    method: "POST",
    body: form,
  });
  const json = await response.json().catch(() => null);
  // Uploads bypass ragRequest, so they have to report availability themselves.
  noteRagResponse(response.status, json);
  if (!response.ok) throw new Error(parseErrorText(response.status, json));
  return json as DocumentUploadResult;
}

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const data = await ragRequest<{
    knowledgeBases: KnowledgeBase[];
    ragAvailable?: boolean;
    ragUnavailableReason?: string | null;
  }>("/knowledge-bases");
  // The one endpoint that degrades to 200 instead of 503, so an empty list here means
  // either an empty store or a host where RAG cannot run. The marker tells them apart.
  noteRagAvailability(data);
  return data.knowledgeBases ?? [];
}

export function createKnowledgeBase(payload: {
  name: string;
  description?: string;
}): Promise<{ id: string; name: string }> {
  return ragRequest("/knowledge-bases", {
    method: "POST",
    body: {
      name: payload.name,
      ...(payload.description ? { description: payload.description } : {}),
    },
  });
}

export function updateKnowledgeBase(
  kbId: string,
  payload: { name?: string; description?: string },
): Promise<{ ok: boolean }> {
  const body: Record<string, unknown> = {};
  if (payload.name !== undefined) body.name = payload.name;
  if (payload.description !== undefined) body.description = payload.description;
  return ragRequest(`/knowledge-bases/${encodeURIComponent(kbId)}`, {
    method: "PATCH",
    body,
  });
}

export function deleteKnowledgeBase(kbId: string): Promise<{ ok: boolean }> {
  return ragRequest(`/knowledge-bases/${encodeURIComponent(kbId)}`, {
    method: "DELETE",
  });
}

export async function listKnowledgeBaseDocuments(
  kbId: string,
): Promise<RagDocument[]> {
  const data = await ragRequest<{ documents: RagDocument[] }>(
    `/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
  );
  return data.documents ?? [];
}

export function uploadKnowledgeBaseDocument(
  kbId: string,
  file: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  return ragUpload(
    `/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
    file,
    ocr,
    caption,
  );
}

export async function listThreadDocuments(
  threadId: string,
): Promise<RagDocument[]> {
  const data = await ragRequest<{ documents: RagDocument[] }>(
    `/threads/${encodeURIComponent(threadId)}/documents`,
  );
  return data.documents ?? [];
}

export function uploadThreadDocument(
  threadId: string,
  file: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  return ragUpload(
    `/threads/${encodeURIComponent(threadId)}/documents`,
    file,
    ocr,
    caption,
  );
}

export async function listProjectDocuments(
  projectId: string,
): Promise<RagDocument[]> {
  const data = await ragRequest<{ documents: RagDocument[] }>(
    `/projects/${encodeURIComponent(projectId)}/documents`,
  );
  return data.documents ?? [];
}

export function uploadProjectDocument(
  projectId: string,
  file: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  return ragUpload(
    `/projects/${encodeURIComponent(projectId)}/documents`,
    file,
    ocr,
    caption,
  );
}

// Cached "does this project have indexed sources?" probe so the chat adapter can
// auto-scope project chats without a round trip per message. The sources panel
// invalidates on upload/delete.
const projectSourcesCache = new Map<string, { has: boolean; at: number }>();
const PROJECT_SOURCES_TTL_MS = 30_000;

export async function projectHasSources(projectId: string): Promise<boolean> {
  const cached = projectSourcesCache.get(projectId);
  if (cached && Date.now() - cached.at < PROJECT_SOURCES_TTL_MS) {
    return cached.has;
  }
  try {
    const docs = await listProjectDocuments(projectId);
    const has = docs.some((doc) => doc.status !== "failed");
    projectSourcesCache.set(projectId, { has, at: Date.now() });
    return has;
  } catch {
    // RAG unavailable or transient failure: don't cache, don't scope.
    return false;
  }
}

/** Fired on every project-source mutation. The composer's documents bar and the
 * project's Sources panel are separate hook instances over the same scope, so
 * without this a file added in one stays invisible to the other. */
export const PROJECT_SOURCES_CHANGED_EVENT = "unsloth-project-sources-changed";

export function invalidateProjectSources(projectId: string): void {
  publishProjectSourcesChanged(projectId);
  getProjectChannel()?.postMessage({ kind: "sources", projectId });
}

function publishProjectSourcesChanged(projectId: string): void {
  projectSourcesCache.delete(projectId);
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(PROJECT_SOURCES_CHANGED_EVENT, { detail: { projectId } }),
  );
}

/** A project's sources are shared by every tab on this origin, and a CustomEvent
 * reaches only the tab that fired it. Without this a second tab keeps listing
 * the sources it saw first, and its cached "no sources" answer stands for the
 * probe's whole TTL. Opened lazily so importing this module starts nothing. */
let projectChannel: BroadcastChannel | null | undefined;

function getProjectChannel(): BroadcastChannel | null {
  if (projectChannel !== undefined) {
    return projectChannel;
  }
  if (
    typeof window === "undefined" ||
    typeof BroadcastChannel === "undefined"
  ) {
    projectChannel = null;
    return null;
  }
  projectChannel = new BroadcastChannel(PROJECT_SOURCES_CHANGED_EVENT);
  projectChannel.onmessage = (event: MessageEvent) => {
    const message = event.data as {
      kind?: string;
      projectId?: string;
      delta?: number;
    } | null;
    if (!message?.projectId) {
      return;
    }
    if (message.kind === "work") {
      noteRemoteProjectWork(message.projectId, message.delta ?? 0);
      return;
    }
    publishProjectSourcesChanged(message.projectId);
  };
  return projectChannel;
}

/** Listeners only hear another tab once the channel is open. */
export function subscribeProjectSourcesBroadcast(): void {
  getProjectChannel();
}

/**
 * Work in flight against a project's sources, by project. A project's sources
 * are changed from the Sources tab as well as the composer, and the instance
 * that is not doing it holds no row for the work until it lands: an upload for
 * the length of its POST, a folder sync for the length of the job. Without this
 * the composer reports nothing indexing, stops polling, and lets a send go out
 * that the sources it is waiting on cannot reach.
 */
const projectWorkInFlight = new Map<string, number>();

export const PROJECT_WORK_CHANGED_EVENT = "unsloth-project-work-changed";

export function noteProjectWork(projectId: string, delta: number): void {
  const next = (projectWorkInFlight.get(projectId) ?? 0) + delta;
  if (next > 0) {
    projectWorkInFlight.set(projectId, next);
  } else {
    projectWorkInFlight.delete(projectId);
  }
  getProjectChannel()?.postMessage({ kind: "work", projectId, delta });
  syncWorkHeartbeat();
  publishProjectWorkChanged(projectId);
}

/**
 * Renew the deadline the other tabs put on this tab's work. An upload of a
 * large file outlives the deadline with no delta to send in between, and
 * without a renewal the other tab would stop counting it and let a send go out
 * mid-upload. A zero delta leaves the count alone and moves the deadline only.
 */
const WORK_HEARTBEAT_MS = 45_000;
let workHeartbeat: ReturnType<typeof setInterval> | null = null;

function syncWorkHeartbeat(): void {
  if (projectWorkInFlight.size === 0) {
    if (workHeartbeat !== null) {
      clearInterval(workHeartbeat);
      workHeartbeat = null;
    }
    return;
  }
  if (workHeartbeat !== null) {
    return;
  }
  workHeartbeat = setInterval(() => {
    const channel = getProjectChannel();
    if (!channel) return;
    for (const projectId of projectWorkInFlight.keys()) {
      channel.postMessage({ kind: "work", projectId, delta: 0 });
    }
  }, WORK_HEARTBEAT_MS);
}

function publishProjectWorkChanged(projectId: string): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent(PROJECT_WORK_CHANGED_EVENT, { detail: { projectId } }),
  );
}

/**
 * Work another tab reports, with a deadline. The tab that started it is the
 * only one that can report the end, and it may be closed or reloaded first, so
 * a count taken on its word alone could gate this project for the session. The
 * deadline caps that at a couple of minutes; a job outliving it falls back to
 * the rows the list refresh brings, which is where the gate started.
 */
const REMOTE_WORK_TTL_MS = 120_000;
const remoteProjectWork = new Map<string, { count: number; until: number }>();
const remoteWorkTimers = new Map<string, ReturnType<typeof setTimeout>>();

function noteRemoteProjectWork(projectId: string, delta: number): void {
  const entry = remoteProjectWork.get(projectId) ?? { count: 0, until: 0 };
  const count = Math.max(0, entry.count + delta);
  const timer = remoteWorkTimers.get(projectId);
  if (timer !== undefined) {
    clearTimeout(timer);
    remoteWorkTimers.delete(projectId);
  }
  if (count === 0) {
    remoteProjectWork.delete(projectId);
  } else {
    remoteProjectWork.set(projectId, {
      count,
      until: Date.now() + REMOTE_WORK_TTL_MS,
    });
    // Re-publish when it lapses, so a listener re-reads instead of holding a
    // count nothing will come back to clear. One timer per project: a chatty
    // tab renews the deadline, it does not stack another wake-up on it.
    remoteWorkTimers.set(
      projectId,
      setTimeout(() => {
        remoteWorkTimers.delete(projectId);
        publishProjectWorkChanged(projectId);
      }, REMOTE_WORK_TTL_MS),
    );
  }
  publishProjectWorkChanged(projectId);
}

export function projectWorkCount(projectId: string): number {
  const remote = remoteProjectWork.get(projectId);
  const remoteCount = remote && remote.until > Date.now() ? remote.count : 0;
  return (projectWorkInFlight.get(projectId) ?? 0) + remoteCount;
}

/** Reads in a row that fail before the watcher stops waiting on the job. */
const MAX_FOLDER_JOB_READ_FAILURES = 20;

const watchedFolderJobs = new Set<string>();

/** Count a folder sync as work on its project until the backend job ends.
 * Tied to the job rather than to whoever started it: leaving the Sources tab
 * aborts that component's event stream but not the job, and the composer has to
 * stay gated until the sources it is adding are actually in. Bounded, so a job
 * that never reports a terminal state cannot gate a project for the session. */
export function watchProjectFolderJob(projectId: string, jobId: string): void {
  if (watchedFolderJobs.has(jobId)) {
    return;
  }
  watchedFolderJobs.add(jobId);
  noteProjectWork(projectId, 1);
  void (async () => {
    // A read that fails is not a job that ended: a backend restart answers a
    // tick or two while the sync runs on. Give up only once the reads stop
    // coming back at all, or the project is released with sources still
    // indexing and the composer becomes sendable through them.
    let consecutiveFailures = 0;
    try {
      for (let attempt = 0; attempt < 600; attempt += 1) {
        try {
          const job = await getFolderSyncJob(jobId);
          consecutiveFailures = 0;
          if (job.status === "completed" || job.status === "failed") {
            break;
          }
        } catch {
          consecutiveFailures += 1;
          if (consecutiveFailures >= MAX_FOLDER_JOB_READ_FAILURES) {
            break;
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
    } finally {
      watchedFolderJobs.delete(jobId);
      noteProjectWork(projectId, -1);
    }
  })();
}

export async function listLinkedFolders(
  scope?: LinkedFolderScope,
): Promise<LinkedFolder[]> {
  const query = scope
    ? `?scope_type=${encodeURIComponent(scope.type)}&scope_id=${encodeURIComponent(scope.id)}`
    : "";
  const data = await ragRequest<{ linkedFolders: LinkedFolder[] }>(
    `/linked-folders${query}`,
  );
  return data.linkedFolders ?? [];
}

interface LinkedFolderMutationResult {
  linkedFolder: LinkedFolder;
  job: FolderSyncJob;
}

export function createLinkedFolder(
  scope: LinkedFolderScope,
  nativePathLease: string,
  displayName: string,
): Promise<LinkedFolderMutationResult> {
  const parent =
    scope.type === "knowledge_base" ? "knowledge-bases" : "projects";
  return ragRequest(
    `/${parent}/${encodeURIComponent(scope.id)}/linked-folders`,
    {
      method: "POST",
      body: { nativePathLease, displayName },
    },
  );
}

export function deleteLinkedFolder(
  linkedFolderId: string,
  removeIndex: boolean,
): Promise<{ ok: boolean }> {
  return ragRequest(
    `/linked-folders/${encodeURIComponent(linkedFolderId)}?remove_index=${removeIndex}`,
    { method: "DELETE" },
  );
}

function startLinkedFolderJob(
  linkedFolderId: string,
  action: "sync" | "rebuild",
): Promise<{ job: FolderSyncJob }> {
  return ragRequest(
    `/linked-folders/${encodeURIComponent(linkedFolderId)}/${action}`,
    { method: "POST" },
  );
}

export function syncLinkedFolder(
  linkedFolderId: string,
): Promise<{ job: FolderSyncJob }> {
  return startLinkedFolderJob(linkedFolderId, "sync");
}

export function rebuildLinkedFolder(
  linkedFolderId: string,
): Promise<{ job: FolderSyncJob }> {
  return startLinkedFolderJob(linkedFolderId, "rebuild");
}

export function getFolderSyncJob(jobId: string): Promise<FolderSyncJob> {
  return ragRequest(`/linked-folder-jobs/${encodeURIComponent(jobId)}`);
}

export async function listAllDocuments(): Promise<UploadedDocument[]> {
  const data = await ragRequest<{ documents: UploadedDocument[] }>(
    "/documents",
  );
  return data.documents ?? [];
}

export async function deleteDocument(
  documentId: string,
  projectId?: string | null,
): Promise<{ ok: boolean }> {
  const result = await ragRequest<{ ok: boolean }>(
    `/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
    },
  );
  if (projectId) invalidateProjectSources(projectId);
  return result;
}

export function getJob(jobId: string): Promise<IndexJob> {
  return ragRequest(`/jobs/${encodeURIComponent(jobId)}`);
}

/** Longest gap between frames before a stream is treated as buffered by a proxy. */
const SSE_STALL_MS = 12000;

async function openEventStream(
  url: string,
  signal: AbortSignal | undefined,
): Promise<ReadableStream<Uint8Array>> {
  const response = await authFetch(url, signal ? { signal } : undefined);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    // also gated on the extension, and also not routed through ragRequest
    noteRagResponse(response.status, body);
    throw new Error(parseErrorText(response.status, body));
  }
  if (!response.body) throw new Error("Stream response missing body");
  return response.body;
}

// sse; returns on [DONE]. transport errors propagate so callers can poll getJob
export async function* streamJobEvents(
  jobId: string,
  signal?: AbortSignal,
): AsyncGenerator<JobEvent> {
  // no stall bound: this consumer reads an early end as a finished job
  const body = await openEventStream(
    `${RAG_BASE}/jobs/${encodeURIComponent(jobId)}/events`,
    signal,
  );
  yield* readSseJsonEvents<JobEvent>(body);
}

export async function* streamFolderSyncJobEvents(
  jobId: string,
  signal?: AbortSignal,
): AsyncGenerator<FolderSyncJobEvent> {
  const body = await openEventStream(
    `${RAG_BASE}/linked-folder-jobs/${encodeURIComponent(jobId)}/events`,
    signal,
  );
  yield* readSseJsonEvents<FolderSyncJobEvent>(body, SSE_STALL_MS);
}

export function getPreviewTarget(
  documentId: string,
  chunkId?: string,
): Promise<PreviewTarget> {
  const qs = chunkId ? `?chunk_id=${encodeURIComponent(chunkId)}` : "";
  return ragRequest(
    `/documents/${encodeURIComponent(documentId)}/preview-target${qs}`,
  );
}

// Signed URL (no bearer) so pdf.js can issue Range requests. Absolute because
// consumers bypass authFetch, and a relative path under Tauri resolves against
// the webview origin.
export async function getDocumentFileUrl(documentId: string): Promise<string> {
  const data = await ragRequest<{ url: string }>(
    `/documents/${encodeURIComponent(documentId)}/file-url`,
  );
  return apiUrl(data.url);
}
