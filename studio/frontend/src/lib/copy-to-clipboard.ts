// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

// Native IPC, so it works outside a user gesture (e.g. after an await).
async function copyWithTauriClipboard(text: string): Promise<boolean> {
  try {
    const { writeText } = await import("@tauri-apps/plugin-clipboard-manager");
    await writeText(text);
    return true;
  } catch (error) {
    console.warn("Tauri clipboard-manager writeText failed", error);
    return false;
  }
}

/**
 * Synchronous textarea + execCommand copy so it runs in the same user gesture
 * as the click (required by Safari's clipboard security).
 */
function copyWithExecCommand(text: string): boolean {
  if (typeof document === "undefined" || !document.body) return false;

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.readOnly = true;
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "0";
  textarea.style.fontSize = "12pt";
  textarea.style.opacity = "0";
  textarea.setAttribute("aria-hidden", "true");
  document.body.appendChild(textarea);
  textarea.focus({ preventScroll: true });
  textarea.select();

  try {
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    document.body.removeChild(textarea);
    return false;
  }
}

export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof text !== "string" || text.length === 0) {
    return false;
  }

  // Exactly one writer runs per call. Pre-arming the web write would keep the click's
  // activation for a native failure, but it also leaves a second write in flight that a
  // rapid second copy can lose a race to, and a silently stale clipboard is worse than a
  // visible failure. Gated synchronously, so a browser still writes inside the gesture.
  if (isTauri && (await copyWithTauriClipboard(text))) {
    return true;
  }

  // Primary: async Clipboard API
  if (typeof navigator?.clipboard?.writeText === "function") {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      console.warn("Async clipboard API failed, falling back to execCommand", error);
      // Rejected (NotAllowedError, insecure context, etc.); fall through.
    }
  }

  // Fallback: execCommand (works in Safari when called during user gesture)
  return copyWithExecCommand(text);
}

/**
 * Copy the result of a read that has to happen first, without losing the
 * gesture that asked for it. Safari drops transient activation across an
 * await, so both writers below fail when the text only arrives after one; the
 * write starts synchronously with a promised payload instead, where the
 * browser has it. A payload that rejects, which is how a caller says it found
 * nothing to copy, leaves the clipboard untouched.
 */
export async function copyToClipboardFrom(
  load: () => Promise<string>,
): Promise<boolean> {
  const payload = load();
  // The write paths below each decide what to do with a rejection; this keeps
  // it from being reported as unhandled in the meantime.
  payload.catch(() => undefined);

  if (
    !isTauri &&
    typeof ClipboardItem === "function" &&
    typeof navigator?.clipboard?.write === "function"
  ) {
    try {
      await navigator.clipboard.write([
        new ClipboardItem({
          "text/plain": payload.then((text) => {
            if (!text) throw new Error("nothing to copy");
            return new Blob([text], { type: "text/plain" });
          }),
        }),
      ]);
      return true;
    } catch (error) {
      console.warn("Promised clipboard write failed, falling back", error);
    }
  }

  try {
    return await copyToClipboard(await payload);
  } catch {
    return false;
  }
}
