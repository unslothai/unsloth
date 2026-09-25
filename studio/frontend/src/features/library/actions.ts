// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useNavigate } from "@tanstack/react-router";
import { zipSync } from "fflate";
import { getAuthSessionEpoch } from "@/features/auth";
import { listLoras } from "@/features/chat";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  requestModelConfigHandoff,
} from "@/features/model-picker";
import { translate } from "@/i18n";
import { MAX_AUDIO_SIZE } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { downloadFile, downloadUrlStreaming, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { MAX_VIDEO_SIZE } from "@/lib/video-utils";
import { type LibraryItem, errorMessage, libraryDownloadUrl, libraryItemFile } from "./api";
import { fileKind } from "./file-kind";
import { hasOwnFile, libraryFileName, uniqueFileNames } from "./file-name";
import { MAX_IMAGE_OR_TEXT_BYTES, resetToNewChat, startLibraryChat } from "./start-chat";

type Navigate = ReturnType<typeof useNavigate>;

export const MAX_CHAT_FILES = 10;

const MAX_DOCUMENT_BYTES = 50 * 1024 * 1024;

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

export async function downloadLibraryItem(
  item: LibraryItem,
  epoch = getAuthSessionEpoch(),
): Promise<void> {
  if (getAuthSessionEpoch() !== epoch) return;
  try {
    if (isTauri && !item.textOnly && hasOwnFile(item.id)) {
      const url = await libraryDownloadUrl(item);
      if (getAuthSessionEpoch() !== epoch) return;
      await downloadUrlStreaming(url, libraryFileName(item));
      return;
    }
    const file = await libraryItemFile(item);
    if (getAuthSessionEpoch() !== epoch) return;
    await downloadFile(file, file.name, file.type);
  } catch (error) {
    if (isDownloadCancelled(error)) return;
    toast.error(translate("library.toast.downloadFailed", { name: item.name }), {
      description: errorMessage(error),
    });
  }
}

const MAX_ZIP_BYTES = 256 * 1024 * 1024;

export async function downloadLibraryItems(items: LibraryItem[]): Promise<void> {
  const epoch = getAuthSessionEpoch();
  if (items.length <= 1 || isTauri) {
    for (const item of items) await downloadLibraryItem(item, epoch);
    return;
  }
  // A file of unknown size could be any size, so it never goes into a zip held in memory.
  const bytes = items.reduce((sum, item) => sum + (item.sizeBytes ?? Infinity), 0);
  if (bytes > MAX_ZIP_BYTES) {
    toast(translate("library.toast.downloadingMany", { count: items.length }), {
      description: translate("library.toast.downloadingManyDescription"),
    });
    for (const item of items) await downloadLibraryItem(item, epoch);
    return;
  }
  const progress = toast.loading(translate("library.toast.preparingMany", { count: items.length }));
  try {
    const files: File[] = [];
    for (const item of items) {
      files.push(await libraryItemFile(item));
      if (getAuthSessionEpoch() !== epoch) return;
    }
    const names = uniqueFileNames(["__proto__", ...files.map((file) => file.name)]).slice(1);
    const entries: Record<string, Uint8Array> = {};
    for (const [index, file] of files.entries()) {
      entries[names[index]!] = new Uint8Array(await file.arrayBuffer());
    }
    const archive = zipSync(entries, { level: 0 });
    if (getAuthSessionEpoch() !== epoch) return;
    await downloadFile(
      new Blob([archive], { type: "application/zip" }),
      `${translate("library.toast.zipFileName")}.zip`,
    );
  } catch (error) {
    toast.error(translate("library.toast.downloadManyFailed"), { description: errorMessage(error) });
  } finally {
    toast.dismiss(progress);
  }
}

export async function chatAboutItems(
  navigate: Navigate,
  items: LibraryItem[],
): Promise<void> {
  if (items.length === 0) {
    toast(translate("library.toast.nothingToChat"), {
      description: translate("library.toast.emptyFolder"),
    });
    return;
  }
  const fitting = items.filter(fitsInChat);
  const tooLarge = items.length - fitting.length;
  if (fitting.length === 0) {
    toast.error(
      items.length === 1
        ? translate("library.toast.tooLargeOne", { name: items[0].name })
        : translate("library.toast.tooLargeMany"),
    );
    return;
  }
  const chosen = fitting.slice(0, MAX_CHAT_FILES);
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
      const attached =
        chosen.length === 1 ? "library.toast.attachedOne" : "library.toast.attachedMany";
      toast(translate(attached, { count: chosen.length }), {
        description: [
          tooLarge > 0 && translate("library.toast.skippedTooLarge", { count: tooLarge }),
          leftOut > 0 &&
            translate("library.toast.skippedOverLimit", { count: leftOut, limit: MAX_CHAT_FILES }),
        ]
          .filter(Boolean)
          .join(" "),
      });
    }
    startLibraryChat(navigate, { files });
  } catch (error) {
    toast.error(translate("library.toast.openFilesFailed"), { description: errorMessage(error) });
  }
}

export async function chatWithModel(
  navigate: Navigate,
  item: LibraryItem,
): Promise<void> {
  const model = item.model;
  if (!model) return;
  const epoch = getAuthSessionEpoch();
  const scanned = await listLoras()
    .then(({ loras }) => loras.find((lora) => lora.adapter_path === model.path))
    .catch(() => undefined);
  if (getAuthSessionEpoch() !== epoch) return;
  if (scanned?.audio_type) {
    toast(translate("library.toast.speechModel", { name: item.name }), {
      description: translate("library.toast.speechModelDescription"),
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
      isGguf: model.exportType === "gguf",
    },
  });
  void navigate({ to: "/chat", search: { new: requestId } }).catch(() =>
    clearModelConfigHandoff(requestId),
  );
}
