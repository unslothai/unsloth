// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useNavigate } from "@tanstack/react-router";
import { getAuthSessionEpoch } from "@/features/auth";
import { clearNewChatDraft, listLoras, useChatRuntimeStore } from "@/features/chat";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  requestModelConfigHandoff,
} from "@/features/model-picker";
import { MAX_AUDIO_SIZE } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { downloadFile, downloadUrlStreaming, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { MAX_VIDEO_SIZE } from "@/lib/video-utils";
import { type LibraryItem, libraryDownloadUrl, libraryItemFile } from "./api";
import { fileKind } from "./file-kind";
import { libraryFileName } from "./file-name";
import {
  type LibraryChatHandoff,
  useLibraryChatHandoffStore,
} from "./chat-handoff-store";

type Navigate = ReturnType<typeof useNavigate>;

// More than this and the composer turns into a wall of chips; the rest can be added by hand.
export const MAX_CHAT_FILES = 10;

// The composer's image and text limit.
const MAX_IMAGE_OR_TEXT_BYTES = 20 * 1024 * 1024;
// Its PDF, DOCX and OpenDocument limit, and the most any other adapter takes, so the ceiling for the rest.
const MAX_DOCUMENT_BYTES = 50 * 1024 * 1024;

/** The most the composer would accept for this item. Checked before any download. */
function chatSizeLimit(item: LibraryItem): number {
  const kind = fileKind(item);
  if (kind === "audio") return MAX_AUDIO_SIZE;
  if (kind === "video") return MAX_VIDEO_SIZE;
  if (kind === "image" || kind === "code" || kind === "web" || item.textOnly) {
    return MAX_IMAGE_OR_TEXT_BYTES;
  }
  return item.contentType.startsWith("text/") ? MAX_IMAGE_OR_TEXT_BYTES : MAX_DOCUMENT_BYTES;
}

function fitsInChat(item: LibraryItem): boolean {
  return item.sizeBytes === null || item.sizeBytes <= chatSizeLimit(item);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// Items with a file of their own, which the Library can serve by id. Chat attachments live inside
// messages and stay small.
const STREAMABLE = /^(upload|image|video|audio|sandbox):/;

export async function downloadLibraryItem(item: LibraryItem): Promise<void> {
  try {
    // The desktop app streams to the chosen path: a Blob plus its IPC copy would hold the file
    // in memory twice.
    if (isTauri && !item.textOnly && STREAMABLE.test(item.id)) {
      await downloadUrlStreaming(await libraryDownloadUrl(item), libraryFileName(item));
      return;
    }
    const file = await libraryItemFile(item);
    await downloadFile(file, file.name, file.type);
  } catch (error) {
    if (isDownloadCancelled(error)) return;
    toast.error(`Could not download ${item.name}`, {
      description: errorMessage(error),
    });
  }
}

// crypto.randomUUID only exists in secure contexts, and Studio is also served over plain http to
// the LAN.
function createNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function resetToNewChat(): void {
  clearNewChatDraft();
  const runtime = useChatRuntimeStore.getState();
  runtime.setActiveThreadId(null);
  runtime.setActiveProjectId(null);
  runtime.setIncognito(false);
}

/** Open a fresh chat with these files attached in the composer. */
function startLibraryChat(
  navigate: Navigate,
  handoff: LibraryChatHandoff,
): void {
  const nonce = createNonce();
  resetToNewChat();
  useLibraryChatHandoffStore.getState().offer(`single:${nonce}`, handoff);
  void navigate({ to: "/chat", search: { new: nonce } });
}

export async function chatAboutItems(
  navigate: Navigate,
  items: LibraryItem[],
): Promise<void> {
  if (items.length === 0) {
    toast("Nothing to chat about yet", {
      description: "This folder has no files.",
    });
    return;
  }
  const fitting = items.filter(fitsInChat);
  const tooLarge = items.length - fitting.length;
  if (fitting.length === 0) {
    toast.error(
      items.length === 1 ? `${items[0].name} is too large to attach` : "These files are too large to attach",
    );
    return;
  }
  const chosen = fitting.slice(0, MAX_CHAT_FILES);
  // A sign-out while the files download would hand them to the next account's chat.
  const epoch = getAuthSessionEpoch();
  try {
    // One at a time, so a folder never holds ten downloads in flight at once.
    const files: File[] = [];
    for (const item of chosen) {
      files.push(await libraryItemFile(item));
      if (getAuthSessionEpoch() !== epoch) return;
    }
    const leftOut = fitting.length - chosen.length;
    if (tooLarge > 0 || leftOut > 0) {
      toast(`Attached ${chosen.length} ${chosen.length === 1 ? "file" : "files"}`, {
        description: [
          tooLarge > 0 && `${tooLarge} too large to attach.`,
          leftOut > 0 && `${leftOut} more past the ${MAX_CHAT_FILES} file limit.`,
        ]
          .filter(Boolean)
          .join(" "),
      });
    }
    startLibraryChat(navigate, { files });
  } catch (error) {
    toast.error("Could not open the files", { description: errorMessage(error) });
  }
}

/** Open a fresh chat with this fine-tuned model's run settings up, as the model picker does. */
export async function chatWithModel(
  navigate: Navigate,
  item: LibraryItem,
): Promise<void> {
  const model = item.model;
  if (!model) return;
  // Only the picker's scan reads a checkpoint's tokenizer; a speech model belongs on the Audio page,
  // since chat cannot serve it.
  const epoch = getAuthSessionEpoch();
  const scanned = await listLoras()
    .then(({ loras }) => loras.find((lora) => lora.adapter_path === model.path))
    .catch(() => undefined);
  if (getAuthSessionEpoch() !== epoch) return;
  if (scanned?.audio_type) {
    toast(`${item.name} is a speech model`, {
      description: "Pick it from the model menu on the Audio page.",
    });
    void navigate({ to: "/audio" });
    return;
  }
  const requestId = createModelConfigHandoffRequestId();
  resetToNewChat();
  requestModelConfigHandoff({
    requestId,
    id: model.path,
    displayName: item.name,
    meta: {
      source: model.origin === "exported" ? "exported" : "lora",
      isLora: model.exportType === "lora",
      isDownloaded: true,
      // An exported GGUF is listed by its first weights file, which loads directly.
      isGguf: model.exportType === "gguf",
    },
  });
  void navigate({ to: "/chat", search: { new: requestId } }).catch(() =>
    clearModelConfigHandoff(requestId),
  );
}
