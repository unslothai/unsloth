// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthSessionEpoch, getAuthToken } from "@/features/auth";
import { type TranslationKey, translate } from "@/i18n";
import { apiUrl } from "@/lib/api-base";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { itemVersion, libraryFileName, libraryFileType } from "./file-name";
import { type DecodedNote, type NoteEncoding, decodeNote } from "./note-text";

export type LibrarySource = "uploaded" | "generated";

export interface LibraryItem {
  id: string;
  name: string;
  fileName?: string;
  source: LibrarySource;
  contentType: string;
  sizeBytes: number | null;
  storageBytes?: number;
  fingerprint?: string;
  createdAt: number;
  updatedAt: number;
  fileUrl: string;
  threadId: string | null;
  threadTitle: string | null;
  textOnly: boolean;
  favorite: boolean;
  folderId: string | null;
  openedAt: number | null;
  /** Set for fine-tuned models, which are directories: opened in chat, never downloaded. */
  model: LibraryModel | null;
  archived?: boolean;
}

export interface LibraryModel {
  path: string;
  origin: "training" | "exported";
  exportType: "lora" | "merged" | "gguf";
  baseModel: string | null;
}

export interface LibraryFolder {
  id: string;
  name: string;
  parentId: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface LibraryDisk {
  totalBytes: number;
  freeBytes: number;
  sources?: string[];
}

export interface LibrarySnapshot {
  items: LibraryItem[];
  folders: LibraryFolder[];
  disk?: LibraryDisk | null;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function ensureOk(response: Response): Promise<Response> {
  if (!response.ok) throw new Error(await readFastApiError(response));
  return response;
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function getLibrary(): Promise<LibrarySnapshot> {
  const response = await ensureOk(await authFetch("/api/library"));
  return response.json();
}

export async function getLibraryFavorites(): Promise<string[]> {
  const response = await ensureOk(await authFetch("/api/library/favorites"));
  return ((await response.json()) as { ids: string[] }).ids;
}

const writeQueues = new Map<string, Promise<void>>();

function sameSession(epoch: number, message: TranslationKey): () => void {
  return () => {
    if (getAuthSessionEpoch() !== epoch) throw new Error(translate(message));
  };
}

/** A write: a retry never goes out under an account other than the one that sent it. */
function sendWrite(input: string, init: RequestInit): Promise<Response> {
  return authFetch(input, init, {
    beforeRetry: sameSession(getAuthSessionEpoch(), "library.toast.signedOutBeforeSave"),
  });
}

/**
 * Sends one write for `target` after the ones before it, so a quick second change never lands
 * first. Per session too: another account can have an item or folder of this id, and must not
 * wait on a write of the account that left, which a sign-out does not cut short.
 */
function inOrder(target: string, send: (check: () => void) => Promise<Response>): Promise<void> {
  const epoch = getAuthSessionEpoch();
  const key = `${epoch}:${target}`;
  const request = (writeQueues.get(key) ?? Promise.resolve())
    .catch(() => {})
    .then(async () => {
      const check = sameSession(epoch, "library.toast.signedOutBeforeSave");
      check();
      await ensureOk(await send(check));
    });
  writeQueues.set(key, request);
  const settle = () => {
    if (writeQueues.get(key) === request) writeQueues.delete(key);
  };
  request.then(settle, settle);
  return request;
}

export function updateLibraryItem(
  id: string,
  patch: { name?: string; favorite?: boolean; folderId?: string | null },
): Promise<void> {
  return inOrder(`item:${id}`, (check) =>
    authFetch("/api/library/items", jsonInit("PATCH", { id, ...patch }), { beforeRetry: check }),
  );
}

export async function markLibraryItemOpened(id: string): Promise<void> {
  await ensureOk(await sendWrite("/api/library/items/opened", jsonInit("POST", { id })));
}

export async function addLibraryItemToProject(
  id: string,
  projectId: string,
): Promise<{ already: boolean }> {
  const response = await ensureOk(
    await sendWrite("/api/library/items/project", jsonInit("POST", { id, projectId })),
  );
  return response.json();
}

export async function revealLibraryItem(id: string): Promise<void> {
  await ensureOk(await sendWrite("/api/library/items/reveal", jsonInit("POST", { id })));
}

export interface LibraryLocation {
  key: "uploads" | "images" | "videos" | "audio" | "fineTunes" | "exports";
  path: string;
  movable?: boolean;
  custom?: boolean;
  available?: boolean;
  disk?: LibraryDisk | null;
  device?: string | null;
}

export async function getLibraryLocations(): Promise<LibraryLocation[]> {
  const response = await ensureOk(await authFetch("/api/library/locations"));
  return (await response.json()).locations;
}

export async function revealLibraryLocation(key: LibraryLocation["key"]): Promise<void> {
  await ensureOk(await sendWrite("/api/library/locations/reveal", jsonInit("POST", { key })));
}

export async function moveLibraryLocation(
  key: LibraryLocation["key"],
  path: string | null,
): Promise<{ locations: LibraryLocation[]; leftBehind: string | null }> {
  const response = await ensureOk(
    await sendWrite("/api/library/locations/move", jsonInit("POST", { key, path })),
  );
  const body = await response.json();
  return { locations: body.locations, leftBehind: body.leftBehind ?? null };
}

export async function deleteLibraryItem(id: string, fingerprint?: string): Promise<void> {
  await ensureOk(
    await sendWrite("/api/library/items/delete", jsonInit("POST", { id, fingerprint })),
  );
}

export interface LibraryUploadBatch {
  files?: File[];
  nativePathLeases?: string[];
  sessionEpoch?: number;
}

export const MAX_LIBRARY_UPLOAD_BYTES = 512 * 1024 * 1024;
// Starlette's form parser refuses more than 1000 files in one request (Request.form max_files).
const MAX_LIBRARY_UPLOAD_FILES = 1000;

function uploadGroups(files: File[]): File[][] {
  const groups: File[][] = [];
  let bytes = Infinity;
  for (const file of files) {
    if (
      bytes + file.size > MAX_LIBRARY_UPLOAD_BYTES ||
      (groups.at(-1)?.length ?? 0) >= MAX_LIBRARY_UPLOAD_FILES
    ) {
      groups.push([]);
      bytes = 0;
    }
    groups[groups.length - 1]!.push(file);
    bytes += file.size;
  }
  return groups;
}

export async function uploadLibraryFiles(
  batch: LibraryUploadBatch,
  folderId: string | null,
): Promise<string[]> {
  const files = batch.files ?? [];
  const tooLarge = files.find((file) => file.size > MAX_LIBRARY_UPLOAD_BYTES);
  if (tooLarge) throw new Error(translate("library.toast.uploadTooLarge", { name: tooLarge.name }));
  const leases = batch.nativePathLeases ?? [];
  const requests: FormData[] = uploadGroups(files).map((group) => {
    const form = new FormData();
    for (const file of group) form.append("files", file, file.name);
    return form;
  });
  if (requests.length === 0) requests.push(new FormData());
  for (const lease of leases) requests[0]!.append("nativePathLeases", lease);
  const check = sameSession(
    batch.sessionEpoch ?? getAuthSessionEpoch(),
    "library.toast.signedOutBeforeUpload",
  );
  const ids: string[] = [];
  for (const form of requests) {
    check();
    if (folderId) form.append("folderId", folderId);
    const response = await ensureOk(
      await authFetch("/api/library/uploads", { method: "POST", body: form }, { beforeRetry: check }),
    );
    ids.push(...((await response.json()) as { ids: string[] }).ids);
  }
  return ids;
}

export async function writeLibraryText(
  itemId: string,
  text: string,
  encoding: NoteEncoding,
): Promise<void> {
  const uploadId = itemId.replace(/^upload:/, "");
  await ensureOk(
    await sendWrite(
      `/api/library/uploads/${encodeURIComponent(uploadId)}/text`,
      jsonInit("PUT", { text, encoding }),
    ),
  );
}

export async function createLibraryFolder(
  name: string,
  parentId: string | null,
): Promise<LibraryFolder> {
  const response = await ensureOk(
    await sendWrite("/api/library/folders", jsonInit("POST", { name, parentId })),
  );
  return response.json();
}

export function updateLibraryFolder(
  id: string,
  patch: { name?: string; parentId?: string | null },
): Promise<void> {
  return inOrder(`folder:${id}`, (check) =>
    authFetch(`/api/library/folders/${encodeURIComponent(id)}`, jsonInit("PATCH", patch), {
      beforeRetry: check,
    }),
  );
}

export function deleteLibraryFolder(id: string): Promise<void> {
  return inOrder(`folder:${id}`, (check) =>
    authFetch(`/api/library/folders/${encodeURIComponent(id)}`, { method: "DELETE" }, {
      beforeRetry: check,
    }),
  );
}

export class LibraryFileTooLarge extends Error {}

export async function fetchLibraryBlob(
  item: LibraryItem,
  type: string,
  maxBytes = Infinity,
): Promise<Blob> {
  const response = await ensureOk(await authFetch(item.fileUrl));
  if (Number(response.headers.get("content-length")) > maxBytes) {
    void response.body?.cancel();
    throw new LibraryFileTooLarge(item.name);
  }
  const reader = maxBytes === Infinity ? undefined : response.body?.getReader();
  if (!reader) {
    const blob = await response.blob();
    if (blob.size > maxBytes) throw new LibraryFileTooLarge(item.name);
    return blob.type === type ? blob : new Blob([blob], { type });
  }
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let read = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    read += value.length;
    if (read > maxBytes) {
      void reader.cancel();
      throw new LibraryFileTooLarge(item.name);
    }
    chunks.push(value);
  }
  return new Blob(chunks, { type });
}

export async function fetchLibraryStreamUrl(item: LibraryItem): Promise<string> {
  const params = new URLSearchParams({ id: item.id });
  const response = await ensureOk(await authFetch(`/api/library/items/stream-url?${params}`));
  const { url } = (await response.json()) as { url?: string };
  if (!url) throw new Error(translate("library.toast.noMediaLink"));
  return apiUrl(url);
}

export async function fetchLibraryThumbnail(item: LibraryItem): Promise<Blob> {
  const params = new URLSearchParams({ id: item.id, v: itemVersion(item) });
  const response = await ensureOk(await authFetch(`/api/library/items/thumbnail?${params}`));
  return response.blob();
}

/** Up to `maxBytes` of the item, decoded as its BOM (or UTF-8) says; the rest is never read. */
export async function fetchLibraryText(
  item: LibraryItem,
  maxBytes: number,
): Promise<DecodedNote & { truncated: boolean }> {
  const response = await ensureOk(await authFetch(item.fileUrl));
  const reader = response.body?.getReader();
  if (!reader) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    const truncated = bytes.length > maxBytes;
    return { ...decodeNote(bytes.subarray(0, maxBytes), truncated), truncated };
  }
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let read = 0;
  let truncated = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (read + value.length > maxBytes) {
      chunks.push(value.subarray(0, maxBytes - read));
      truncated = true;
      void reader.cancel();
      break;
    }
    chunks.push(value);
    read += value.length;
  }
  const bytes = new Uint8Array(await new Blob(chunks).arrayBuffer());
  return { ...decodeNote(bytes, truncated), truncated };
}

export async function libraryDownloadUrl(item: LibraryItem): Promise<string> {
  const path = `/api/library/items/download?${new URLSearchParams({ id: item.id })}`;
  await ensureOk(await authFetch(path, { method: "HEAD" }));
  const token = getAuthToken();
  return apiUrl(token ? `${path}&token=${encodeURIComponent(token)}` : path);
}

export async function libraryItemFile(item: LibraryItem): Promise<File> {
  const name = libraryFileName(item);
  const type = libraryFileType(name, item.textOnly ? "text/plain" : item.contentType);
  return new File([await fetchLibraryBlob(item, type)], name, { type });
}
