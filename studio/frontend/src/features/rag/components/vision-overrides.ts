


import {
  CHAT_RAG_CAPTION_KEY,
  CHAT_RAG_OCR_KEY,
  useChatRuntimeStore,
} from "@/features/chat";

// Hydration is a network round trip with no deadline of its own: authFetch sets
// no AbortSignal, so a wedged or unreachable server leaves the promise pending
// for as long as the socket does. Uploads used to read these overrides
// synchronously, so waiting forever would be a new way for a drop to hang with
// nothing on screen. Bound it and take the local values, the way a blocked
// localStorage already falls through to the backend defaults.
const HYDRATION_WAIT_MS = 8_000;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

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
