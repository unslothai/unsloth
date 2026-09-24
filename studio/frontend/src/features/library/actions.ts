// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useNavigate } from "@tanstack/react-router";
import { listLoras } from "@/features/chat";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  requestModelConfigHandoff,
} from "@/features/model-picker";
import { MAX_AUDIO_SIZE } from "@/lib/audio-utils";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { MAX_VIDEO_SIZE } from "@/lib/video-utils";
import { type LibraryItem, libraryItemFile } from "./api";
import { fileKind } from "./file-kind";
import { resetToNewChat, startLibraryChat } from "./start-chat";

type Navigate = ReturnType<typeof useNavigate>;

// More than this and the composer turns into a wall of chips; the rest can be added by hand.
export const MAX_CHAT_FILES = 10;

// The composer's image and text limit.
const MAX_IMAGE_OR_TEXT_BYTES = 20 * 1024 * 1024;

/** What the composer would accept, or null where it sets no limit. Checked before any download. */
function chatSizeLimit(item: LibraryItem): number | null {
  const kind = fileKind(item);
  if (kind === "audio") return MAX_AUDIO_SIZE;
  if (kind === "video") return MAX_VIDEO_SIZE;
  if (kind === "image" || kind === "code" || kind === "web" || item.textOnly) {
    return MAX_IMAGE_OR_TEXT_BYTES;
  }
  return item.contentType.startsWith("text/") ? MAX_IMAGE_OR_TEXT_BYTES : null;
}

function fitsInChat(item: LibraryItem): boolean {
  const limit = chatSizeLimit(item);
  return limit === null || item.sizeBytes === null || item.sizeBytes <= limit;
}

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
  try {
    // One at a time, so a folder never holds ten downloads in flight at once.
    const files: File[] = [];
    for (const item of chosen) files.push(await libraryItemFile(item));
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
