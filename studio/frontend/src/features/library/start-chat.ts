// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept apart from actions.ts so pages outside the Library can start a chat without loading it.
import type { useNavigate } from "@tanstack/react-router";
import { getAuthSessionEpoch } from "@/features/auth";
// eslint-disable-next-line no-restricted-imports -- the chat barrel imports the Library back
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
// eslint-disable-next-line no-restricted-imports -- the chat barrel imports the Library back
import { clearNewChatDraft } from "@/features/chat/utils/composer-draft";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { MAX_VIDEO_SIZE } from "@/lib/video-utils";
import {
  type LibraryChatHandoff,
  useLibraryChatHandoffStore,
} from "./chat-handoff-store";
import { mediaFileName } from "./media-file-name";

type Navigate = ReturnType<typeof useNavigate>;

// crypto.randomUUID only exists in secure contexts, and Studio is also served over plain http to
// the LAN.
function createNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function resetToNewChat(): void {
  clearNewChatDraft();
  const runtime = useChatRuntimeStore.getState();
  runtime.setActiveThreadId(null);
  runtime.setActiveProjectId(null);
  runtime.setIncognito(false);
}

/** Open a fresh chat with these files attached in the composer. */
export function startLibraryChat(
  navigate: Navigate,
  handoff: LibraryChatHandoff,
): void {
  const nonce = createNonce();
  resetToNewChat();
  useLibraryChatHandoffStore.getState().offer(`single:${nonce}`, handoff);
  void navigate({ to: "/chat", search: { new: nonce } });
}

// The composer's image and text limit.
export const MAX_IMAGE_OR_TEXT_BYTES = 20 * 1024 * 1024;

// The composer's own limits, checked first so a clip it would refuse never opens an empty chat.
const MEDIA_LIMITS = {
  image: { bytes: MAX_IMAGE_OR_TEXT_BYTES, extension: "png", type: "image/png" },
  video: { bytes: MAX_VIDEO_SIZE, extension: "mp4", type: "video/mp4" },
} as const;

// Long enough that a small image never flashes a toast.
const LOADING_TOAST_DELAY_MS = 400;

// One hand-off at a time: a second click while a large clip downloads would open a second chat.
let handoffInFlight = false;

/**
 * Open a fresh chat with a generated image or clip attached, named after its prompt. `source` is
 * a URL to fetch, or a loader for a response a URL cannot give (WebKit will not refetch an object
 * URL it is displaying, and a signed link can die with a server restart).
 */
export async function chatAboutMedia(
  navigate: Navigate,
  source: string | (() => Promise<Response>),
  prompt: string,
  kind: keyof typeof MEDIA_LIMITS,
): Promise<void> {
  if (handoffInFlight) return;
  handoffInFlight = true;
  const limit = MEDIA_LIMITS[kind];
  const tooLarge = () =>
    toast.error(translate(kind === "image" ? "library.toast.imageTooLarge" : "library.toast.videoTooLarge"), {
      description: translate("library.toast.attachmentLimit", {
        size: Math.round(limit.bytes / (1024 * 1024)),
      }),
    });
  let loadingToast: string | number | null = null;
  const loadingTimer = setTimeout(() => {
    loadingToast = toast.loading(
      translate(kind === "image" ? "library.toast.attachingImage" : "library.toast.attachingVideo"),
    );
  }, LOADING_TOAST_DELAY_MS);
  // A sign-out while the file downloads would hand it to the next account's chat.
  const epoch = getAuthSessionEpoch();
  try {
    const response = typeof source === "string" ? await fetch(source) : await source();
    if (!response.ok) {
      throw new Error(
        translate(kind === "image" ? "library.toast.readImageFailed" : "library.toast.readVideoFailed", {
          status: response.status,
        }),
      );
    }
    // Before reading the body, so an oversized file is never buffered.
    if (Number(response.headers.get("content-length")) > limit.bytes) {
      void response.body?.cancel();
      tooLarge();
      return;
    }
    const blob = await response.blob();
    if (getAuthSessionEpoch() !== epoch) return;
    if (blob.size > limit.bytes) {
      tooLarge();
      return;
    }
    // A missing or generic type would not be taken as an image or a clip by the composer.
    const type = blob.type.startsWith(`${kind}/`) ? blob.type : limit.type;
    startLibraryChat(navigate, {
      files: [new File([blob], mediaFileName(prompt, limit.extension), { type })],
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
