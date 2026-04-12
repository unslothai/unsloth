// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Copy text to clipboard in a way that works on Mac/Safari.
 * Uses a synchronous textarea + execCommand fallback so the copy runs in the
 * same user gesture as the click (required by Safari's clipboard security).
 */
function copyWithExecCommand(text: string): boolean {
  if (typeof document === "undefined" || !document.body) return false;

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "0";
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

export function copyToClipboard(text: string): boolean {
  if (typeof text !== "string" || text.length === 0) {
    return false;
  }

  if (document.queryCommandSupported?.("copy") !== false) {
    return copyWithExecCommand(text);
  }

  // Async fallback for environments where execCommand is entirely unsupported
  // but the Clipboard API is available (rare; kept for original contract parity).
  if (typeof navigator?.clipboard?.writeText === "function") {
    navigator.clipboard.writeText(text).then(
      () => {},
      () => {},
    );
    return true;
  }

  return false;
}

export async function copyToClipboardAsync(text: string): Promise<boolean> {
  if (typeof text !== "string" || text.length === 0) {
    return false;
  }

  // Prefer the async Clipboard API: avoids focus disruption in Radix
  // focus-trapped dialogs where execCommand always fails.
  if (typeof navigator?.clipboard?.writeText === "function") {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // After an await rejection the synchronous user-gesture frame is gone,
      // so execCommand will also fail. Return false rather than attempting it.
      return false;
    }
  }

  // No Clipboard API (older browser / non-secure context): still in the
  // original user-gesture frame, so execCommand can work.
  return copyWithExecCommand(text);
}
