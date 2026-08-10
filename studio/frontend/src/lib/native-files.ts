// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { decodeDataUri, isDataUri } from "@/lib/data-uri";

const NATIVE_FILE_NAME_HEADER = "x-unsloth-default-name";
export class DownloadCancelledError extends Error {
  constructor() {
    super("Save cancelled.");
    this.name = "DownloadCancelledError";
  }
}

export function isDownloadCancelled(error: unknown): boolean {
  return error instanceof DownloadCancelledError;
}

function encodeNativeFilename(filename: string): string {
  const bytes = new TextEncoder().encode(filename);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

export interface NativeImportedTextFile {
  name: string;
  content: string;
}

export type NativeChatImport = NativeImportedTextFile;

function browserDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function browserUrlDownload(url: string, filename: string): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

async function fetchDownload(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed with status ${response.status}.`);
  }
  return response;
}

/** Save through a native chooser in Tauri and retain normal downloads on web. */
export async function downloadFile(
  content: string | Blob | Uint8Array,
  filename: string,
  mimeType = "application/octet-stream",
): Promise<void> {
  if (isTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    const bytes =
      typeof content === "string"
        ? new TextEncoder().encode(content)
        : content instanceof Blob
          ? new Uint8Array(await content.arrayBuffer())
          : content;
    const savedPath = await invoke<string | null>("save_native_file", bytes, {
      headers: {
        [NATIVE_FILE_NAME_HEADER]: encodeNativeFilename(filename),
      },
    });
    if (savedPath === null) {
      throw new DownloadCancelledError();
    }
    return;
  }

  const browserContent =
    content instanceof Uint8Array ? Uint8Array.from(content).buffer : content;
  const blob =
    browserContent instanceof Blob
      ? browserContent
      : new Blob([browserContent], { type: mimeType });

  browserDownload(blob, filename);
  return;
}

export async function urlToBlob(url: string): Promise<Blob> {
  if (isDataUri(url)) {
    const { bytes, mimeType } = decodeDataUri(url);
    return new Blob([Uint8Array.from(bytes).buffer], { type: mimeType });
  }
  return (await fetchDownload(url)).blob();
}

/**
 * Save a local backend URL without holding it in memory. `downloadUrl` buffers the body
 * to cross the IPC boundary, which is the wrong shape for a gallery clip: here the
 * chooser opens first and Rust streams to the chosen path. The browser keeps its anchor.
 */
export async function downloadUrlStreaming(
  url: string,
  filename: string,
): Promise<void> {
  if (!isTauri) {
    browserUrlDownload(url, filename);
    return;
  }
  const { invoke } = await import("@tauri-apps/api/core");
  const savedPath = await invoke<string | null>("save_native_file_from_url", {
    url,
    fileName: filename,
  });
  if (savedPath === null) {
    throw new DownloadCancelledError();
  }
}

/** Resolve media before crossing the native save boundary. */
export async function downloadUrl(
  url: string,
  filename: string,
): Promise<void> {
  if (isDataUri(url)) {
    const { bytes, mimeType } = decodeDataUri(url);
    await downloadFile(bytes, filename, mimeType);
    return;
  }
  if (!isTauri) {
    browserUrlDownload(url, filename);
    return;
  }
  const response = await fetchDownload(url);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await downloadFile(
    bytes,
    filename,
    response.headers.get("content-type") || "application/octet-stream",
  );
}

/** Open the bounded native chat-import picker. Cancellation returns null. */
export async function pickNativeChatImport(): Promise<NativeChatImport | null> {
  if (!isTauri) {
    return null;
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<NativeChatImport | null>("pick_native_chat_import");
}

export async function pickNativeTrainingConfig(): Promise<NativeImportedTextFile | null> {
  if (!isTauri) {
    return null;
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<NativeImportedTextFile | null>("pick_native_training_config");
}
