// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useNavigate } from "@tanstack/react-router";
import { clearNewChatDraft, listLoras, useChatRuntimeStore } from "@/features/chat";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  requestModelConfigHandoff,
} from "@/features/model-picker";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { type LibraryItem, libraryItemFile } from "./api";
import {
  type LibraryChatHandoff,
  useLibraryChatHandoffStore,
} from "./chat-handoff-store";

type Navigate = ReturnType<typeof useNavigate>;

// More than this and the composer turns into a wall of chips; the rest can be added by hand.
export const MAX_CHAT_FILES = 10;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function downloadLibraryItem(item: LibraryItem): Promise<void> {
  try {
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
  const chosen = items.slice(0, MAX_CHAT_FILES);
  try {
    const files = await Promise.all(chosen.map(libraryItemFile));
    if (items.length > chosen.length) {
      toast(`Attached the ${chosen.length} most recent files`, {
        description: `${items.length - chosen.length} more were left out.`,
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
  const scanned = await listLoras()
    .then(({ loras }) => loras.find((lora) => lora.adapter_path === model.path))
    .catch(() => undefined);
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
