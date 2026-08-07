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

/**
 * Start the async Clipboard API write without awaiting it. The spec reads transient
 * activation when writeText is *called*, so arming it here keeps the result usable
 * after a later await. Null when the API is missing (insecure context).
 */
function startWebClipboardWrite(text: string): Promise<boolean> | null {
  if (typeof navigator?.clipboard?.writeText !== "function") return null;
  return navigator.clipboard.writeText(text).then(
    () => true,
    (error) => {
      console.warn("Async clipboard API failed, falling back to execCommand", error);
      return false;
    },
  );
}

export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof text !== "string" || text.length === 0) {
    return false;
  }

  // Armed before any await, so it still carries the click's activation even when the
  // Tauri attempt below yields first and then fails.
  const webWrite = startWebClipboardWrite(text);

  // Native IPC has no activation requirement, which is the whole point: callers that
  // await something before copying have already lost the gesture.
  if (isTauri && (await copyWithTauriClipboard(text))) {
    return true;
  }

  if (webWrite && (await webWrite)) {
    return true;
  }

  // Fallback: execCommand (works in Safari when called during user gesture)
  return copyWithExecCommand(text);
}
