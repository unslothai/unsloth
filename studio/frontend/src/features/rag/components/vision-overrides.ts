// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CHAT_RAG_CAPTION_KEY,
  CHAT_RAG_OCR_KEY,
  useChatRuntimeStore,
} from "@/features/chat";

function hasLocal(key: string): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(key) !== null;
  } catch {
    // Storage can be blocked outright (sandboxed context). These overrides are
    // optional, so fall back to the backend defaults rather than failing the
    // upload that asked for them.
    return false;
  }
}

/** Ingest-time vision-pass overrides, sent only once the user has set them;
 * otherwise backend env defaults own the policy. Shared by every upload path. */
export async function resolveVisionOverrides(): Promise<{
  ocr: boolean | undefined;
  caption: boolean | undefined;
}> {
  // These live in the mirrored chat settings, so a fresh browser has to wait for
  // them: an ingest cannot be undone once its vision passes have run.
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  const state = useChatRuntimeStore.getState();
  return {
    ocr: hasLocal(CHAT_RAG_OCR_KEY) ? state.ragOcrScanned : undefined,
    caption: hasLocal(CHAT_RAG_CAPTION_KEY)
      ? state.ragCaptionFigures
      : undefined,
  };
}
