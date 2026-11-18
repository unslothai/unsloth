// SPDX-License-Identifier: AGPL-3.0-only
import { getAuthSessionEpoch, getConfirmedLoginMode } from "@/features/auth";
import type { DiffusionLoadRequest } from "./api";

export const CHAT_IMAGE_SWAP_ENABLED = "unsloth:chat:swap-image-models";
export const CHAT_IMAGE_SWAP_CHANGED = "unsloth:chat-image-swap-changed";
const storageKey = "unsloth:chat:last-text-to-image-load";
let cached: { epoch: number; image: DiffusionLoadRequest } | undefined;

export function rememberImageForChat(
  request: DiffusionLoadRequest,
  epoch: number,
) {
  if (getAuthSessionEpoch() !== epoch || getConfirmedLoginMode() !== "single")
    return;
  const image = { ...request };
  delete image.hf_token;
  delete image.load_token;
  cached = { epoch, image };
  try {
    localStorage.setItem(storageKey, JSON.stringify(image));
  } catch {
    // The current tab can still use the model with storage disabled.
  }
}

export function chatImageSwapSelection() {
  if (getConfirmedLoginMode() !== "single") return null;
  const epoch = getAuthSessionEpoch();
  try {
    if (localStorage.getItem(CHAT_IMAGE_SWAP_ENABLED) !== "true") return null;
    if (cached?.epoch === epoch) return cached;
    const image = JSON.parse(localStorage.getItem(storageKey) ?? "null");
    if (typeof image?.model_path !== "string" || !image.model_path) return null;
    cached = { epoch, image };
    return cached;
  } catch {
    return null;
  }
}
