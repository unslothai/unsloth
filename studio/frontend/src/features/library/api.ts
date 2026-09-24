// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type LibrarySource = "uploaded" | "generated";

export interface LibraryItem {
  /** `<source>:<ref>`: upload, attachment, image, audio, model or sandbox. */
  id: string;
  name: string;
  source: LibrarySource;
  contentType: string;
  sizeBytes: number | null;
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
  /** Set for fine-tuned models, which are directories: opened in chat, never downloaded. */
  model: LibraryModel | null;
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

export interface LibrarySnapshot {
  items: LibraryItem[];
  folders: LibraryFolder[];
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

// Edits to one item go out in order, so a quick second toggle never lands before the first.
const itemQueues = new Map<string, Promise<void>>();

export function updateLibraryItem(
  id: string,
  patch: { name?: string; favorite?: boolean; folderId?: string | null },
): Promise<void> {
  const request = (itemQueues.get(id) ?? Promise.resolve())
    .catch(() => {})
    .then(async () => {
      await ensureOk(
        await authFetch("/api/library/items", jsonInit("PATCH", { id, ...patch })),
      );
    });
  itemQueues.set(id, request);
  const settle = () => {
    if (itemQueues.get(id) === request) itemQueues.delete(id);
  };
  request.then(settle, settle);
  return request;
}

/** Copies the item's file into a project's folder; the Library keeps its item. */
export async function addLibraryItemToProject(
  id: string,
  projectId: string,
): Promise<{ already: boolean }> {
  const response = await ensureOk(
    await authFetch("/api/library/items/project", jsonInit("POST", { id, projectId })),
  );
  return response.json();
}

/** Shows the item's file in the OS file manager on the machine running Studio. */
export async function revealLibraryItem(id: string): Promise<void> {
  await ensureOk(await authFetch("/api/library/items/reveal", jsonInit("POST", { id })));
}

export interface LibraryLocation {
  key: "uploads" | "images" | "videos" | "audio" | "fineTunes" | "exports";
  path: string;
}

export async function getLibraryLocations(): Promise<LibraryLocation[]> {
  const response = await ensureOk(await authFetch("/api/library/locations"));
  return (await response.json()).locations;
}

export async function revealLibraryLocation(key: LibraryLocation["key"]): Promise<void> {
  await ensureOk(await authFetch("/api/library/locations/reveal", jsonInit("POST", { key })));
}

export async function deleteLibraryItem(id: string): Promise<void> {
  await ensureOk(
    await authFetch("/api/library/items/delete", jsonInit("POST", { id })),
  );
}

/** Browser Files, or desktop drops as signed path grants the backend reads itself. */
export interface LibraryUploadBatch {
  files?: File[];
  nativePathLeases?: string[];
}

export async function uploadLibraryFiles(
  batch: LibraryUploadBatch,
  folderId: string | null,
): Promise<string[]> {
  const form = new FormData();
  for (const file of batch.files ?? []) form.append("files", file, file.name);
  for (const lease of batch.nativePathLeases ?? []) form.append("nativePathLeases", lease);
  if (folderId) form.append("folderId", folderId);
  const response = await ensureOk(
    await authFetch("/api/library/uploads", { method: "POST", body: form }),
  );
  return ((await response.json()) as { ids: string[] }).ids;
}

/** Only Library-owned uploads (`upload:<id>`) are writable. */
export async function writeLibraryText(
  itemId: string,
  text: string,
): Promise<void> {
  const uploadId = itemId.replace(/^upload:/, "");
  await ensureOk(
    await authFetch(
      `/api/library/uploads/${encodeURIComponent(uploadId)}/text`,
      jsonInit("PUT", { text }),
    ),
  );
}

export async function createLibraryFolder(
  name: string,
  parentId: string | null,
): Promise<LibraryFolder> {
  const response = await ensureOk(
    await authFetch("/api/library/folders", jsonInit("POST", { name, parentId })),
  );
  return response.json();
}

export async function updateLibraryFolder(
  id: string,
  patch: { name?: string; parentId?: string | null },
): Promise<void> {
  await ensureOk(
    await authFetch(
      `/api/library/folders/${encodeURIComponent(id)}`,
      jsonInit("PATCH", patch),
    ),
  );
}

export async function deleteLibraryFolder(id: string): Promise<void> {
  await ensureOk(
    await authFetch(`/api/library/folders/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  );
}

/** The item's bytes, typed as what they are: sources serve most files as opaque downloads. */
export async function fetchLibraryBlob(item: LibraryItem): Promise<Blob> {
  const response = await ensureOk(await authFetch(item.fileUrl));
  const blob = await response.blob();
  const type = item.textOnly ? "text/plain" : item.contentType;
  return blob.type === type ? blob : new Blob([blob], { type });
}

/** A video item's first frame, drawn by the backend. The version keeps a stale frame out of caches. */
export async function fetchLibraryThumbnail(item: LibraryItem): Promise<Blob> {
  const params = new URLSearchParams({ id: item.id, v: String(item.updatedAt) });
  const response = await ensureOk(await authFetch(`/api/library/items/thumbnail?${params}`));
  return response.blob();
}

/** Up to `maxBytes` of the item decoded as text; the rest of the body is never read. */
export async function fetchLibraryTextPrefix(
  item: LibraryItem,
  maxBytes: number,
): Promise<{ text: string; truncated: boolean }> {
  const response = await ensureOk(await authFetch(item.fileUrl));
  const reader = response.body?.getReader();
  if (!reader) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    return {
      text: new TextDecoder().decode(bytes.subarray(0, maxBytes)),
      truncated: bytes.length > maxBytes,
    };
  }
  const decoder = new TextDecoder();
  let text = "";
  let read = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return { text: text + decoder.decode(), truncated: false };
    if (read + value.length > maxBytes) {
      text += decoder.decode(value.subarray(0, maxBytes - read), { stream: true });
      void reader.cancel();
      return { text: text + decoder.decode(), truncated: true };
    }
    read += value.length;
    text += decoder.decode(value, { stream: true });
  }
}

/** What Download and "Chat about this" hand over. Text-only chat uploads say so in the name. */
export async function libraryItemFile(item: LibraryItem): Promise<File> {
  const blob = await fetchLibraryBlob(item);
  const name = item.textOnly ? `${item.name}.txt` : item.name;
  return new File([blob], name, { type: blob.type });
}
