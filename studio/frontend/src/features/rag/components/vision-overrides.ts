// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CHAT_RAG_CAPTION_KEY,
  CHAT_RAG_OCR_KEY,
  useChatRuntimeStore,
} from "@/features/chat";

// authFetch sets no AbortSignal, so bound hydration and fall back to the local values.
const HYDRATION_WAIT_MS = 8_000;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function hasLocal(key: string): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(key) !== null;
  } catch {
    // Storage can be blocked (sandboxed context); fall back to backend defaults.
    return false;
  }
}

export async function resolveVisionOverrides(): Promise<{
  ocr: boolean | undefined;
  caption: boolean | undefined;
}> {
  // Wait for mirrored settings: an ingest cannot be undone once its vision passes have run.
  await Promise.race([
    useChatRuntimeStore.getState().hydratePersistedSettings(),
    wait(HYDRATION_WAIT_MS),
  ]);
  const state = useChatRuntimeStore.getState();
  return {
    ocr: hasLocal(CHAT_RAG_OCR_KEY) ? state.ragOcrScanned : undefined,
    caption: hasLocal(CHAT_RAG_CAPTION_KEY)
      ? state.ragCaptionFigures
      : undefined,
  };
}
