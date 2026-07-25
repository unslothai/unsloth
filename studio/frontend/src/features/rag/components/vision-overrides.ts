// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CHAT_RAG_CAPTION_KEY,
  CHAT_RAG_OCR_KEY,
  useChatRuntimeStore,
} from "@/features/chat";

function hasLocal(key: string): boolean {
  return (
    typeof window !== "undefined" && window.localStorage.getItem(key) !== null
  );
}

/** Ingest-time vision-pass overrides, sent only once the user has set them;
 * otherwise backend env defaults own the policy. Shared by every upload path. */
export function resolveVisionOverrides(): {
  ocr: boolean | undefined;
  caption: boolean | undefined;
} {
  const state = useChatRuntimeStore.getState();
  return {
    ocr: hasLocal(CHAT_RAG_OCR_KEY) ? state.ragOcrScanned : undefined,
    caption: hasLocal(CHAT_RAG_CAPTION_KEY)
      ? state.ragCaptionFigures
      : undefined,
  };
}
