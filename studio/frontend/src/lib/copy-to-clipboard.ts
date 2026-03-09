// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

/**
 * Copy text to clipboard in a way that works on Mac/Safari.
 * Uses a synchronous textarea + execCommand fallback so the copy runs in the
 * same user gesture as the click (required by Safari's clipboard security).
 */
export function copyToClipboard(text: string): boolean {
  if (typeof text !== "string" || text.length === 0) {
    return false;
  }

  // Synchronous fallback: works in Safari/Mac when clipboard API fails
  // because it runs entirely within the user gesture (click) stack.
  if (document.queryCommandSupported?.("copy") !== false) {
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

  // Modern API only when fallback not available (e.g. non-browser)
  if (typeof navigator?.clipboard?.writeText === "function") {
    navigator.clipboard.writeText(text).then(
      () => {},
      () => {}
    );
    return true;
  }

  return false;
}
