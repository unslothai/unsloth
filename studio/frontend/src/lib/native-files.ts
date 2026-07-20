// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

const NATIVE_FILE_NAME_HEADER = "x-unsloth-default-name";

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
  content: string | Blob,
  filename: string,
  mimeType = "application/octet-stream",
): Promise<boolean> {
  const blob =
    content instanceof Blob ? content : new Blob([content], { type: mimeType });

  if (isTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    const bytes =
      typeof content === "string"
        ? new TextEncoder().encode(content)
        : new Uint8Array(await content.arrayBuffer());
    const savedPath = await invoke<string | null>("save_native_file", bytes, {
      headers: {
        [NATIVE_FILE_NAME_HEADER]: encodeNativeFilename(filename),
      },
    });
    return savedPath !== null;
  }

  browserDownload(blob, filename);
  return true;
}

/** Open the bounded native chat-import picker. Cancellation returns null. */
export async function pickNativeChatImport(): Promise<NativeChatImport | null> {
  if (!isTauri) {
    return null;
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<NativeChatImport | null>("pick_native_chat_import");
}
