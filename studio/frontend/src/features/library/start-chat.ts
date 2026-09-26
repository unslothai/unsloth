// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useNavigate } from "@tanstack/react-router";
import { getAuthSessionEpoch } from "@/features/auth";
// eslint-disable-next-line no-restricted-imports -- the chat barrel imports the Library back
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
// eslint-disable-next-line no-restricted-imports -- the chat barrel imports the Library back
import { clearNewChatDraft } from "@/features/chat/utils/composer-draft";
import { createModelConfigHandoffRequestId } from "@/features/model-picker";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { MAX_VIDEO_SIZE } from "@/lib/video-utils";
import {
  type LibraryChatHandoff,
  useLibraryChatHandoffStore,
} from "./chat-handoff-store";
import { mediaFileName } from "@/lib/prompt-text";

type Navigate = ReturnType<typeof useNavigate>;

export function resetToNewChat(): void {
  clearNewChatDraft();
  const runtime = useChatRuntimeStore.getState();
  runtime.setActiveThreadId(null);
  runtime.setActiveProjectId(null);
  runtime.setIncognito(false);
}

export function startLibraryChat(
  navigate: Navigate,
  handoff: LibraryChatHandoff,
): void {
  const nonce = createModelConfigHandoffRequestId();
  resetToNewChat();
  // Offered once the new chat is on screen: from inside a chat, the composer that is open until
  // then is the old thread's, and an offer drained early would attach the files there.
  void navigate({ to: "/chat", search: { new: nonce } }).then(() => {
    requestAnimationFrame(() =>
      useLibraryChatHandoffStore.getState().offer(`single:${nonce}`, handoff),
    );
  });
}

export const MAX_IMAGE_OR_TEXT_BYTES = 20 * 1024 * 1024;

const MEDIA = {
  image: {
    bytes: MAX_IMAGE_OR_TEXT_BYTES,
    extension: "png",
    type: "image/png",
    tooLarge: "library.toast.imageTooLarge",
    attaching: "library.toast.attachingImage",
    readFailed: "library.toast.readImageFailed",
  },
  video: {
    bytes: MAX_VIDEO_SIZE,
    extension: "mp4",
    type: "video/mp4",
    tooLarge: "library.toast.videoTooLarge",
    attaching: "library.toast.attachingVideo",
    readFailed: "library.toast.readVideoFailed",
  },
} as const;

const LOADING_TOAST_DELAY_MS = 400;

// One hand-off at a time: a second click while a large clip downloads would open a second chat.
let handoffInFlight = false;

export async function chatAboutMedia(
  navigate: Navigate,
  load: () => Promise<Response>,
  prompt: string,
  kind: keyof typeof MEDIA,
): Promise<void> {
  if (handoffInFlight) return;
  handoffInFlight = true;
  const media = MEDIA[kind];
  const tooLarge = () =>
    toast.error(translate(media.tooLarge), {
      description: translate("library.toast.attachmentLimit", {
        size: Math.round(media.bytes / (1024 * 1024)),
      }),
    });
  let loadingToast: string | number | null = null;
  const loadingTimer = setTimeout(() => {
    loadingToast = toast.loading(translate(media.attaching));
  }, LOADING_TOAST_DELAY_MS);
  // A sign-out while the file downloads would hand it to the next account's chat.
  const epoch = getAuthSessionEpoch();
  // The media pages stay mounted off-route: a user who moved on must not be pulled into a new chat.
  const startedAt = typeof window === "undefined" ? "" : window.location.pathname;
  try {
    const response = await load();
    if (!response.ok) throw new Error(translate(media.readFailed, { status: response.status }));
    // Before reading the body, so an oversized file is never buffered.
    if (Number(response.headers.get("content-length")) > media.bytes) {
      void response.body?.cancel();
      tooLarge();
      return;
    }
    const blob = await response.blob();
    if (getAuthSessionEpoch() !== epoch) return;
    if (typeof window !== "undefined" && window.location.pathname !== startedAt) return;
    if (blob.size > media.bytes) {
      tooLarge();
      return;
    }
    const type = blob.type.startsWith(`${kind}/`) ? blob.type : media.type;
    startLibraryChat(navigate, {
      files: [new File([blob], mediaFileName(prompt, media.extension), { type })],
    });
  } catch (error) {
    toast.error(translate("library.toast.openFileFailed"), {
      description: error instanceof Error ? error.message : String(error),
    });
  } finally {
    clearTimeout(loadingTimer);
    if (loadingToast !== null) toast.dismiss(loadingToast);
    handoffInFlight = false;
  }
}
