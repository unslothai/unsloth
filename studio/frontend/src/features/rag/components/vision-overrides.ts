


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
