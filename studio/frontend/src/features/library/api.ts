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
  /** `<source>:<ref>`: upload, attachment, image, audio, model or sandbox. */
  id: string;
  /** Shown, and changed by Rename. */
  name: string;
  /** The file's own name, which a rename leaves alone: its type comes from this. */
  fileName?: string;
  source: LibrarySource;
  contentType: string;
  sizeBytes: number | null;
  /** What it takes on disk with the files kept beside it (a clip's recipe), when that is more. */
  storageBytes?: number;
  /** The file a path-derived item (sandbox) was listed as; a delete only takes that file. */
  fingerprint?: string;
  createdAt: number;
  updatedAt: number;
  /** Served by the item's own source route; always fetched with auth. */
  fileUrl: string;
  threadId: string | null;
  threadTitle: string | null;
  /** Chat uploads of documents keep only their extracted text. */
  textOnly: boolean;
  favorite: boolean;
  folderId: string | null;
  /** Last time it was opened in the Library, if ever. */
  openedAt: number | null;
  /** Set for fine-tuned models, which are directories: opened in chat, never downloaded. */
  model: LibraryModel | null;
  /** Generated media off its gallery page's active shelf. */
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
  /** Item sources on this disk (id prefix, `model:<origin>` for models); others may be elsewhere. */
  sources?: string[];
}

export interface LibrarySnapshot {
  items: LibraryItem[];
  folders: LibraryFolder[];
  /** The disk holding the Library's own files; null when it could not be read. */
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

// Writes to one item or folder, in the order they were made.
const writeQueues = new Map<string, Promise<void>>();

/** Throws once the session that `epoch` came from has ended. */
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
      // Queued behind a write that outlived a sign-out: it belongs to the account that left.
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

/** Copies the item's file into a project's folder; the Library keeps its item. */
export async function addLibraryItemToProject(
  id: string,
  projectId: string,
): Promise<{ already: boolean }> {
  const response = await ensureOk(
    await sendWrite("/api/library/items/project", jsonInit("POST", { id, projectId })),
  );
  return response.json();
}

/** Shows the item's file in the OS file manager on the machine running Studio. */
export async function revealLibraryItem(id: string): Promise<void> {
  await ensureOk(await sendWrite("/api/library/items/reveal", jsonInit("POST", { id })));
}

export interface LibraryLocation {
  key: "uploads" | "images" | "videos" | "audio" | "fineTunes" | "exports";
  path: string;
  /** Can be moved to another folder (installation owner only). */
  movable?: boolean;
  /** Already moved away from the default. */
  custom?: boolean;
  /** False while a chosen folder's drive is not connected. */
  available?: boolean;
  /** Space on the disk holding the folder; null while it cannot be read. */
  disk?: LibraryDisk | null;
  /** Tells folders on one disk from folders on another. */
  device?: string | null;
}

export async function getLibraryLocations(): Promise<LibraryLocation[]> {
  const response = await ensureOk(await authFetch("/api/library/locations"));
  return (await response.json()).locations;
}

export async function revealLibraryLocation(key: LibraryLocation["key"]): Promise<void> {
  await ensureOk(await sendWrite("/api/library/locations/reveal", jsonInit("POST", { key })));
}

/** Move one kind of file, files and all (`path` null: back to the default), untimed, as that can
 *  take a while across drives. `leftBehind`: a folder Reset let go of on an unplugged drive. */
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

/** Browser Files, or desktop drops as signed path grants the backend reads itself. */
export interface LibraryUploadBatch {
  files?: File[];
  nativePathLeases?: string[];
  /** The session the batch was gathered in, when that came before the upload (a drop's grants). */
  sessionEpoch?: number;
}

// The backend's cap per file and per request; a larger batch goes as several requests.
export const MAX_LIBRARY_UPLOAD_BYTES = 512 * 1024 * 1024;

/** Batches of files that each fit one request, in order. */
function uploadGroups(files: File[]): File[][] {
  const groups: File[][] = [];
  let bytes = Infinity;
  for (const file of files) {
    if (bytes + file.size > MAX_LIBRARY_UPLOAD_BYTES) {
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
  // Refused here with its name, before anything is sent, rather than as a bare 413.
  const tooLarge = files.find((file) => file.size > MAX_LIBRARY_UPLOAD_BYTES);
  if (tooLarge) throw new Error(translate("library.toast.uploadTooLarge", { name: tooLarge.name }));
  const leases = batch.nativePathLeases ?? [];
  const requests: FormData[] = uploadGroups(files).map((group) => {
    const form = new FormData();
    for (const file of group) form.append("files", file, file.name);
    return form;
  });
  // Desktop drops are grants, not bytes: they ride with the first request.
  if (requests.length === 0) requests.push(new FormData());
  for (const lease of leases) requests[0]!.append("nativePathLeases", lease);
  // A sign-out mid-batch ends it, before the next request or a retry: the token would be another
  // account's.
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

/** Only Library-owned uploads (`upload:<id>`) are writable. */
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

/** A file past the most the caller would hold, found from what the server sends. */
export class LibraryFileTooLarge extends Error {}

/** The item's bytes as `type`: sources serve most files as opaque downloads, so the caller says
 *  what they are (see file-name.ts for the rules). Past `maxBytes` it stops reading and throws
 *  LibraryFileTooLarge: the listed size can be out of date, or unknown. */
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

/**
 * A short-lived signed link an <audio> or <video> element streams the item from, with range
 * requests, instead of buffering the whole file. Bearer-gated to mint, HMAC to use, and good for
 * this one item only, so no long-lived token ends up in a URL.
 */
export async function fetchLibraryStreamUrl(item: LibraryItem): Promise<string> {
  const params = new URLSearchParams({ id: item.id });
  const response = await ensureOk(await authFetch(`/api/library/items/stream-url?${params}`));
  const { url } = (await response.json()) as { url?: string };
  if (!url) throw new Error(translate("library.toast.noMediaLink"));
  // Absolute, since the element fetches it without authFetch, and under Tauri a relative path
  // resolves against the webview.
  return apiUrl(url);
}

/** A video item's first frame, drawn by the backend. The version keeps a stale frame out of caches. */
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

/**
 * An absolute URL for the item's bytes that carries its own token, for the desktop app's native
 * save, which sends no header. The HEAD goes through authFetch first, which refreshes an expired
 * token; the URL alone cannot.
 */
export async function libraryDownloadUrl(item: LibraryItem): Promise<string> {
  const path = `/api/library/items/download?${new URLSearchParams({ id: item.id })}`;
  await ensureOk(await authFetch(path, { method: "HEAD" }));
  const token = getAuthToken();
  return apiUrl(token ? `${path}&token=${encodeURIComponent(token)}` : path);
}

/** What Download and "Chat about this" hand over: a name safe on any OS, typed for the composer. */
export async function libraryItemFile(item: LibraryItem): Promise<File> {
  const name = libraryFileName(item);
  const type = libraryFileType(name, item.textOnly ? "text/plain" : item.contentType);
  return new File([await fetchLibraryBlob(item, type)], name, { type });
}
