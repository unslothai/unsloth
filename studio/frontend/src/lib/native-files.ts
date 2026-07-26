// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

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

export interface NativeChatImport {
  name: string;
  content: string;
}

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

/** Open the bounded native chat-import picker. Cancellation returns null. */
export async function pickNativeChatImport(): Promise<NativeChatImport | null> {
  if (!isTauri) {
    return null;
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<NativeChatImport | null>("pick_native_chat_import");
}
