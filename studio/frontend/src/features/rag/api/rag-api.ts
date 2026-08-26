// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import { openStreamResponse } from "@/lib/open-stream-response";
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

/** The message a caller shows, plus the status one has to branch on: a 404 is an
 * answer, a network failure is not. */
export function ragError(status: number, body: unknown): Error & { status: number } {
  return Object.assign(new Error(parseErrorText(status, body)), { status });
}

/** True for a failure the server answered definitively, so retrying cannot help. */
export function isRagClientError(error: unknown): boolean {
  const status = (error as { status?: unknown } | null)?.status;
  return typeof status === "number" && status >= 400 && status < 500 && status !== 429;
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

/** Fired on every project-source mutation. The composer's bar and the Sources
 * panel are separate hooks over one scope, so each has to hear the other. */
export const PROJECT_SOURCES_CHANGED_EVENT = "unsloth-project-sources-changed";
/** Earlier name for the same event, so existing listeners keep working. */
export const PROJECT_SOURCES_UPDATED_EVENT = PROJECT_SOURCES_CHANGED_EVENT;

/** Drop the probe's cached answer, and nothing else: callers invalidate before
 * their own mutation too, where a refetch would resurrect a dropped row. */
export function invalidateProjectSources(projectId: string): void {
  projectSourcesCache.delete(projectId);
}

/** Invalidate, then tell every mounted list, in this tab and in the others a
 * CustomEvent never reaches. Call after a mutation. */
export function announceProjectSourcesUpdated(projectId: string): void {
  publishProjectSourcesChanged(projectId);
  getProjectChannel()?.postMessage({ kind: "sources", projectId });
}

/** Run `onUpdated` when this project's sources change elsewhere. Returns the
 * unsubscribe. */
export function subscribeProjectSourcesUpdated(
  projectId: string,
  onUpdated: () => void,
): () => void {
  if (typeof window === "undefined") return () => undefined;
  const listener = (event: Event) => {
    const detail = (event as CustomEvent<{ projectId?: string }>).detail;
    // Another project's save must not refetch this one's list.
    if (detail?.projectId === projectId) onUpdated();
  };
  window.addEventListener(PROJECT_SOURCES_CHANGED_EVENT, listener);
  subscribeProjectSourcesBroadcast();
  return () => {
    window.removeEventListener(PROJECT_SOURCES_CHANGED_EVENT, listener);
  };
}

function publishProjectSourcesChanged(projectId: string): void {
  projectSourcesCache.delete(projectId);
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(PROJECT_SOURCES_CHANGED_EVENT, { detail: { projectId } }),
  );
}

/** Every tab on this origin shares a project's sources, and a CustomEvent reaches
 * only its own tab, so a second tab would list what it saw first for the probe's
 * whole TTL. Opened lazily, so importing this module starts nothing. */
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
  // Node's BroadcastChannel holds the event loop open and hangs a test run.
  // Browsers have no unref and need none.
  (projectChannel as { unref?: () => void }).unref?.();
  projectChannel.onmessage = (event: MessageEvent) => {
    const message = event.data as {
      kind?: string;
      projectId?: string;
      delta?: number;
      count?: number;
      from?: string;
    } | null;
    if (!message) {
      return;
    }
    // A tab that opened mid-upload asking what is already running.
    if (message.kind === "work-query") {
      answerWorkQuery();
      return;
    }
    if (!message.projectId) {
      return;
    }
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
    if (message.kind === "work" || message.kind === "work-state") {
      return;
    }
    publishProjectSourcesChanged(message.projectId);
  };
  askForWorkInFlight();
  return projectChannel;
}

/** BroadcastChannel does not replay, so a tab opening mid-upload hears nothing
 * until that upload completes. Ask on the way in instead. */
function askForWorkInFlight(): void {
  projectChannel?.postMessage({ kind: "work-query" });
}

function answerWorkQuery(): void {
  const channel = getProjectChannel();
  if (!channel) return;
  for (const [projectId, count] of projectWorkInFlight) {
    channel.postMessage({ kind: "work-state", projectId, count, from: TAB_ID });
  }
}

/** This tab, so its work is counted apart from every other tab's: merged into one
 * project-wide count, the first upload to finish would release the second. */
const TAB_ID = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

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
  getProjectChannel()?.postMessage({
    kind: "work",
    projectId,
    delta,
    from: TAB_ID,
  });
  syncWorkHeartbeat();
  publishProjectWorkChanged(projectId);
}

/** Renew the deadline the other tabs put on this tab's work: a large upload
 * outlives it with no delta to send in between. The absolute count, not a zero
 * delta, since a delta cannot revive an entry the receiver has let lapse. */
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
  workHeartbeat = setInterval(answerWorkQuery, WORK_HEARTBEAT_MS);
  (workHeartbeat as { unref?: () => void }).unref?.();
}

function publishProjectWorkChanged(projectId: string): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent(PROJECT_WORK_CHANGED_EVENT, { detail: { projectId } }),
  );
}

/** Work another tab reports, with a deadline: only the tab that started it can
 * report the end, and it may be closed first. Past the deadline the gate falls
 * back to the rows the list refresh brings. */
const REMOTE_WORK_TTL_MS = 120_000;
/** Per project, per reporting tab: aggregated, the first upload to finish would
 * clear the count the second is still holding. */
const remoteProjectWork = new Map<
  string,
  Map<string, { count: number; until: number }>
>();
const remoteWorkTimers = new Map<string, ReturnType<typeof setTimeout>>();

function armRemoteWorkExpiry(projectId: string): void {
  const timer = remoteWorkTimers.get(projectId);
  if (timer !== undefined) {
    clearTimeout(timer);
  }
  remoteWorkTimers.delete(projectId);
  const bySender = remoteProjectWork.get(projectId);
  if (!bySender || bySender.size === 0) return;
  // The earliest deadline among the senders, not a fresh TTL: one timer covers
  // the project, so arming it for the tab that just reported would leave a tab
  // that has since closed counted until some later event happens to publish.
  let earliest = Number.POSITIVE_INFINITY;
  for (const entry of bySender.values()) {
    earliest = Math.min(earliest, entry.until);
  }
  const expiry = setTimeout(
    () => {
      remoteWorkTimers.delete(projectId);
      // Drop what has actually lapsed, tell the listeners, then arm for the next
      // deadline, or a listener holds a count nothing comes back to clear.
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
  // As with the channel: a pending timer would hold a Node test run open.
  (expiry as { unref?: () => void }).unref?.();
  remoteWorkTimers.set(projectId, expiry);
}

function setRemoteProjectWork(
  projectId: string,
  from: string,
  count: number,
): void {
  const bySender = remoteProjectWork.get(projectId) ?? new Map();
  if (count <= 0) {
    bySender.delete(from);
  } else {
    bySender.set(from, { count, until: Date.now() + REMOTE_WORK_TTL_MS });
  }
  if (bySender.size === 0) {
    remoteProjectWork.delete(projectId);
    const timer = remoteWorkTimers.get(projectId);
    if (timer !== undefined) {
      clearTimeout(timer);
      remoteWorkTimers.delete(projectId);
    }
  } else {
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
  // A zero delta is the heartbeat: it renews this sender's deadline only, and
  // says nothing about a sender with nothing running.
  if (delta === 0 && current === 0) return;
  setRemoteProjectWork(projectId, from, Math.max(0, current + delta));
}

/** An absolute count a tab reports, for work it started before this tab was
 * listening and on every heartbeat after. Recorded against that tab alone, since
 * the others answer separately. Always renews the deadline and never lowers a
 * count a delta raised, so a heartbeat also revives a lapsed entry. */
function seedRemoteProjectWork(
  projectId: string,
  from: string,
  count: number,
): void {
  if (count <= 0) return;
  setRemoteProjectWork(projectId, from, Math.max(remoteSenderCount(projectId, from), count));
}

export function projectWorkCount(projectId: string): number {
  let remoteCount = 0;
  const now = Date.now();
  for (const entry of remoteProjectWork.get(projectId)?.values() ?? []) {
    if (entry.until > now) remoteCount += entry.count;
  }
  return (projectWorkInFlight.get(projectId) ?? 0) + remoteCount;
}

/** Reads in a row that fail before the watcher stops waiting on the job. */
const MAX_FOLDER_JOB_READ_FAILURES = 20;

const watchedFolderJobs = new Set<string>();

/** Count a folder sync as work on its project until the backend job ends. Tied
 * to the job, not its starter: leaving the Sources tab aborts that component's
 * stream but not the sync. Bounded, so a job that never reports a terminal state
 * cannot gate a project for the session. */
export function watchProjectFolderJob(projectId: string, jobId: string): void {
  if (watchedFolderJobs.has(jobId)) {
    return;
  }
  watchedFolderJobs.add(jobId);
  noteProjectWork(projectId, 1);
  void (async () => {
    // A read that fails is not a job that ended: a backend restart misses a
    // tick or two while the sync runs on. Give up only once they stop coming.
    let consecutiveFailures = 0;
    try {
      for (let attempt = 0; attempt < 600; attempt += 1) {
        try {
          const job = await getFolderSyncJob(jobId);
          consecutiveFailures = 0;
          if (job.status === "completed" || job.status === "failed") {
            break;
          }
        } catch (error) {
          // An answered 4xx is the job being gone, not a read that failed:
          // unlinking deletes its job rows, and so does the history prune.
          if (isRagClientError(error)) break;
          consecutiveFailures += 1;
          if (consecutiveFailures >= MAX_FOLDER_JOB_READ_FAILURES) {
            break;
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
    } finally {
      watchedFolderJobs.delete(jobId);
      // The rows this job wrote are new sources, and this watcher is the only
      // observer left once the panel unmounts. Announce before the gate, or a
      // send released by it still reads the cached "no sources".
      announceProjectSourcesUpdated(projectId);
      noteProjectWork(projectId, -1);
    }
  })();
}

/** When a project may be looked at again: a bare per-call lookup doubles every
 * open (two bars mount at once), and a permanent one misses every job the
 * backend's timer starts later. */
const folderReconcileNotBefore = new Map<string, number>();

/** Shorter than the backend's own scan interval, so a periodic caller is never
 * the one skipped. */
const FOLDER_RECONCILE_MIN_GAP_MS = 5000;

/**
 * Pick up folder syncs already running on a project. Their watchers live in the
 * tab that started them, so a reload, or the tab closing, leaves a durable job
 * scanning with nothing counting it. The backend scans before it writes any
 * rows, so the composer's own list cannot see it either and the gate would open
 * on an empty list. Only the Sources panel lists linked folders, and a project
 * opens on Chats, so the composer has to ask.
 */
export async function reconcileProjectFolderJobs(
  projectId: string,
): Promise<void> {
  const now = Date.now();
  if ((folderReconcileNotBefore.get(projectId) ?? 0) > now) return;
  folderReconcileNotBefore.set(projectId, now + FOLDER_RECONCILE_MIN_GAP_MS);
  // Every look, not just the first: a scan writes no row until it is underway,
  // so the list the composer already has proves nothing.
  noteProjectWork(projectId, 1);
  try {
    const folders = await listLinkedFolders({ type: "project", id: projectId });
    for (const folder of folders) {
      if (folder.activeJobId) {
        watchProjectFolderJob(projectId, folder.activeJobId);
      }
    }
  } catch {
    // RAG unavailable or a transient failure. Allow another look rather than
    // recording a project as reconciled on an answer that never came.
    folderReconcileNotBefore.delete(projectId);
  } finally {
    // After the watchers above, which take their own leases, so the two overlap.
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
  if (projectId) announceProjectSourcesUpdated(projectId);
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
  const response = await openStreamResponse(authFetch, url, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    // also gated on the extension, and also not routed through ragRequest
    noteRagResponse(response.status, body);
    throw ragError(response.status, body);
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
