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
  /** Last time it was opened in the Library, if ever. */
  openedAt: number | null;
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

export async function updateLibraryItem(
  id: string,
  patch: { name?: string; favorite?: boolean; folderId?: string | null },
): Promise<void> {
  await ensureOk(
    await authFetch("/api/library/items", jsonInit("PATCH", { id, ...patch })),
  );
}

export async function markLibraryItemOpened(id: string): Promise<void> {
  await ensureOk(
    await authFetch("/api/library/items/opened", jsonInit("POST", { id })),
  );
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

/** What Download and "Chat about this" hand over. Text-only chat uploads say so in the name. */
export async function libraryItemFile(item: LibraryItem): Promise<File> {
  const blob = await fetchLibraryBlob(item);
  const name = item.textOnly ? `${item.name}.txt` : item.name;
  return new File([blob], name, { type: blob.type });
}
