import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import { readSseJsonEvents } from "@/lib/sse-json-events";
import {
  type DocumentUploadResult,
  type FolderSyncJob,
  type FolderSyncJobEvent,
  type IndexJob,
  type JobEvent,
  type KnowledgeBase,
  type LinkedFolder,
  type LinkedFolderScope,
  type PreviewTarget,
  type RagDocument,
  type UploadedDocument,
  terminalJobStatus,
} from "../types/rag";
import {
  type KnowledgeBaseWriteInput,
  createDatasetKnowledgeBase,
  deleteDatasetKnowledgeBase,
  listAllKnowledgeBases,
  updateDatasetKnowledgeBase,
} from "./platform-dataset-adapter";
import { noteRagResponse } from "./rag-availability";

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

function ragError(status: number, body: unknown): Error & { status: number } {
  return Object.assign(new Error(parseErrorText(status, body)), { status });
}

export function isRagClientError(error: unknown): boolean {
  const status = (error as { status?: unknown } | null)?.status;
  return (
    typeof status === "number" &&
    status >= 400 &&
    status < 500 &&
    status !== 429
  );
}

async function ragRequest<T>(
  path: string,
  init?: { method?: string; body?: object; signal?: AbortSignal },
): Promise<T> {
  const response = await authFetch(`${RAG_BASE}${path}`, {
    method: init?.method,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
    signal: init?.signal,
  });
  if (response.status === 204) {
    noteRagResponse(204, null);
    return undefined as T;
  }
  const json = await response.json().catch(() => null);
  // Every RAG endpoint but the list gates on the extension loading, so its status is
  // also an availability answer. See api/rag-availability.
  noteRagResponse(response.status, json);
  if (!response.ok) throw ragError(response.status, json);
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
  if (!response.ok) throw ragError(response.status, json);
  return json as DocumentUploadResult;
}

export function listKnowledgeBases(
  signal?: AbortSignal,
): Promise<KnowledgeBase[]> {
  return listAllKnowledgeBases(signal);
}

export function createKnowledgeBase(
  payload: KnowledgeBaseWriteInput,
  signal?: AbortSignal,
): Promise<KnowledgeBase> {
  return createDatasetKnowledgeBase(payload, signal);
}

export function updateKnowledgeBase(
  kbId: string,
  payload: KnowledgeBaseWriteInput,
  signal?: AbortSignal,
): Promise<KnowledgeBase> {
  return updateDatasetKnowledgeBase(kbId, payload, signal);
}

export function deleteKnowledgeBase(
  kbId: string,
  signal?: AbortSignal,
): Promise<void> {
  return deleteDatasetKnowledgeBase(kbId, signal);
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

export const PROJECT_SOURCES_CHANGED_EVENT = "unsloth-project-sources-changed";
export const PROJECT_SOURCES_UPDATED_EVENT = PROJECT_SOURCES_CHANGED_EVENT;
export const PROJECT_WORK_CHANGED_EVENT = "unsloth-project-work-changed";

export function invalidateProjectSources(projectId: string): void {
  projectSourcesCache.delete(projectId);
}

function publishProjectSourcesChanged(projectId: string): void {
  invalidateProjectSources(projectId);
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(PROJECT_SOURCES_CHANGED_EVENT, { detail: { projectId } }),
  );
}

let projectChannel: BroadcastChannel | null | undefined;

function getProjectChannel(): BroadcastChannel | null {
  if (projectChannel !== undefined) return projectChannel;
  if (
    typeof window === "undefined" ||
    typeof BroadcastChannel === "undefined"
  ) {
    projectChannel = null;
    return null;
  }
  projectChannel = new BroadcastChannel(PROJECT_SOURCES_CHANGED_EVENT);
  (projectChannel as { unref?: () => void }).unref?.();
  projectChannel.onmessage = (event: MessageEvent) => {
    const message = event.data as {
      kind?: string;
      projectId?: string;
      delta?: number;
      count?: number;
      from?: string;
    } | null;
    if (!message) return;
    if (message.kind === "work-query") {
      answerWorkQuery();
      return;
    }
    if (!message.projectId) return;
    if (message.kind === "work-state" && message.from) {
      seedRemoteProjectWork(
        message.projectId,
        message.from,
        message.count ?? 0,
      );
      return;
    }
    if (message.kind === "work" && message.from) {
      noteRemoteProjectWork(
        message.projectId,
        message.from,
        message.delta ?? 0,
      );
      return;
    }
    if (message.kind !== "work" && message.kind !== "work-state") {
      publishProjectSourcesChanged(message.projectId);
    }
  };
  projectChannel.postMessage({ kind: "work-query" });
  return projectChannel;
}

const TAB_ID = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
const projectWorkInFlight = new Map<string, number>();

export function subscribeProjectSourcesBroadcast(): void {
  getProjectChannel();
}

export function announceProjectSourcesUpdated(projectId: string): void {
  publishProjectSourcesChanged(projectId);
  getProjectChannel()?.postMessage({ kind: "sources", projectId });
}

export function subscribeProjectSourcesUpdated(
  projectId: string,
  onUpdated: () => void,
): () => void {
  if (typeof window === "undefined") return () => undefined;
  const listener = (event: Event) => {
    const detail = (event as CustomEvent<{ projectId?: string }>).detail;
    if (detail?.projectId === projectId) onUpdated();
  };
  window.addEventListener(PROJECT_SOURCES_CHANGED_EVENT, listener);
  subscribeProjectSourcesBroadcast();
  return () =>
    window.removeEventListener(PROJECT_SOURCES_CHANGED_EVENT, listener);
}

function answerWorkQuery(): void {
  const channel = getProjectChannel();
  if (!channel) return;
  for (const [projectId, count] of projectWorkInFlight) {
    channel.postMessage({ kind: "work-state", projectId, count, from: TAB_ID });
  }
}

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
  if (workHeartbeat !== null) return;
  workHeartbeat = setInterval(answerWorkQuery, WORK_HEARTBEAT_MS);
  (workHeartbeat as { unref?: () => void }).unref?.();
}

function publishProjectWorkChanged(projectId: string): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(PROJECT_WORK_CHANGED_EVENT, { detail: { projectId } }),
  );
}

export function noteProjectWork(projectId: string, delta: number): void {
  const next = (projectWorkInFlight.get(projectId) ?? 0) + delta;
  if (next > 0) projectWorkInFlight.set(projectId, next);
  else projectWorkInFlight.delete(projectId);
  getProjectChannel()?.postMessage({
    kind: "work",
    projectId,
    delta,
    from: TAB_ID,
  });
  syncWorkHeartbeat();
  publishProjectWorkChanged(projectId);
}

const REMOTE_WORK_TTL_MS = 120_000;
const remoteProjectWork = new Map<
  string,
  Map<string, { count: number; until: number }>
>();
const remoteWorkTimers = new Map<string, ReturnType<typeof setTimeout>>();

function armRemoteWorkExpiry(projectId: string): void {
  const timer = remoteWorkTimers.get(projectId);
  if (timer !== undefined) clearTimeout(timer);
  remoteWorkTimers.delete(projectId);
  const bySender = remoteProjectWork.get(projectId);
  if (!bySender?.size) return;
  let earliest = Number.POSITIVE_INFINITY;
  for (const entry of bySender.values()) {
    earliest = Math.min(earliest, entry.until);
  }
  const expiry = setTimeout(
    () => {
      remoteWorkTimers.delete(projectId);
      const now = Date.now();
      const live = remoteProjectWork.get(projectId);
      if (live) {
        for (const [sender, entry] of live) {
          if (entry.until <= now) live.delete(sender);
        }
        if (live.size === 0) remoteProjectWork.delete(projectId);
      }
      publishProjectWorkChanged(projectId);
      armRemoteWorkExpiry(projectId);
    },
    Math.max(0, earliest - Date.now()),
  );
  (expiry as { unref?: () => void }).unref?.();
  remoteWorkTimers.set(projectId, expiry);
}

function setRemoteProjectWork(
  projectId: string,
  from: string,
  count: number,
): void {
  const bySender = remoteProjectWork.get(projectId) ?? new Map();
  if (count <= 0) bySender.delete(from);
  else bySender.set(from, { count, until: Date.now() + REMOTE_WORK_TTL_MS });
  if (bySender.size === 0) remoteProjectWork.delete(projectId);
  else {
    remoteProjectWork.set(projectId, bySender);
    armRemoteWorkExpiry(projectId);
  }
  publishProjectWorkChanged(projectId);
}

function remoteSenderCount(projectId: string, from: string): number {
  const entry = remoteProjectWork.get(projectId)?.get(from);
  return entry && entry.until > Date.now() ? entry.count : 0;
}

function noteRemoteProjectWork(
  projectId: string,
  from: string,
  delta: number,
): void {
  const current = remoteSenderCount(projectId, from);
  if (delta === 0 && current === 0) return;
  setRemoteProjectWork(projectId, from, Math.max(0, current + delta));
}

function seedRemoteProjectWork(
  projectId: string,
  from: string,
  count: number,
): void {
  if (count <= 0) return;
  setRemoteProjectWork(
    projectId,
    from,
    Math.max(remoteSenderCount(projectId, from), count),
  );
}

export function projectWorkCount(projectId: string): number {
  let remoteCount = 0;
  const now = Date.now();
  for (const entry of remoteProjectWork.get(projectId)?.values() ?? []) {
    if (entry.until > now) remoteCount += entry.count;
  }
  return (projectWorkInFlight.get(projectId) ?? 0) + remoteCount;
}

const MAX_FOLDER_JOB_READ_FAILURES = 20;
const watchedFolderJobs = new Set<string>();

export function watchProjectFolderJob(projectId: string, jobId: string): void {
  if (watchedFolderJobs.has(jobId)) return;
  watchedFolderJobs.add(jobId);
  noteProjectWork(projectId, 1);
  void (async () => {
    let consecutiveFailures = 0;
    try {
      for (let attempt = 0; attempt < 600; attempt += 1) {
        try {
          const job = await getFolderSyncJob(jobId);
          consecutiveFailures = 0;
          if (terminalJobStatus(job.status)) break;
        } catch (error) {
          if (isRagClientError(error)) break;
          consecutiveFailures += 1;
          if (consecutiveFailures >= MAX_FOLDER_JOB_READ_FAILURES) break;
        }
        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
    } finally {
      watchedFolderJobs.delete(jobId);
      announceProjectSourcesUpdated(projectId);
      noteProjectWork(projectId, -1);
    }
  })();
}

const folderReconcileNotBefore = new Map<string, number>();
const FOLDER_RECONCILE_MIN_GAP_MS = 5000;

export async function reconcileProjectFolderJobs(
  projectId: string,
): Promise<void> {
  const now = Date.now();
  if ((folderReconcileNotBefore.get(projectId) ?? 0) > now) return;
  folderReconcileNotBefore.set(projectId, now + FOLDER_RECONCILE_MIN_GAP_MS);
  noteProjectWork(projectId, 1);
  try {
    const folders = await listLinkedFolders({ type: "project", id: projectId });
    for (const folder of folders) {
      if (folder.activeJobId) {
        watchProjectFolderJob(projectId, folder.activeJobId);
      }
    }
  } catch {
    folderReconcileNotBefore.delete(projectId);
  } finally {
    noteProjectWork(projectId, -1);
  }
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

export function getJob(jobId: string): Promise<IndexJob> {
  return ragRequest(`/jobs/${encodeURIComponent(jobId)}`);
}

const SSE_STALL_MS = 12000;

async function openEventStream(
  url: string,
  signal: AbortSignal | undefined,
): Promise<ReadableStream<Uint8Array>> {
  const response = await authFetch(url, signal ? { signal } : undefined);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    noteRagResponse(response.status, body);
    throw new Error(parseErrorText(response.status, body));
  }
  if (!response.body) throw new Error("Stream response missing body");
  return response.body;
}

export async function* streamJobEvents(
  jobId: string,
  signal?: AbortSignal,
): AsyncGenerator<JobEvent> {
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

export function getPreviewTarget(
  documentId: string,
  chunkId?: string,
  signal?: AbortSignal,
): Promise<PreviewTarget> {
  const qs = chunkId ? `?chunk_id=${encodeURIComponent(chunkId)}` : "";
  return ragRequest(
    `/documents/${encodeURIComponent(documentId)}/preview-target${qs}`,
    { signal },
  );
}

// Signed URL (no bearer) so pdf.js can issue Range requests. Absolute because
// consumers bypass authFetch, and a relative path under Tauri resolves against
// the webview origin.
export async function getDocumentFileUrl(
  documentId: string,
  signal?: AbortSignal,
): Promise<string> {
  const data = await ragRequest<{ url: string }>(
    `/documents/${encodeURIComponent(documentId)}/file-url`,
    { signal },
  );
  return apiUrl(data.url);
}
