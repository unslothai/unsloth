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

/** Synchronous so it runs inside the click gesture, as Safari requires. */
function copyWithExecCommand(text: string): boolean {
  if (typeof document === "undefined" || !document.body) return false;

  // A modal focus trap empties the selection; the copy event writes the text anyway.
  let written = false;
  const onCopy = (event: ClipboardEvent) => {
    if (!event.clipboardData) return;
    event.clipboardData.setData("text/plain", text);
    event.preventDefault();
    event.stopImmediatePropagation();
    written = true;
  };

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

  document.addEventListener("copy", onCopy, true);
  try {
    return document.execCommand("copy") && written;
  } catch {
    return false;
  } finally {
    document.removeEventListener("copy", onCopy, true);
    document.body.removeChild(textarea);
  }
}

export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof text !== "string" || text.length === 0) {
    return false;
  }

  // Exactly one writer per call: a pre-armed second write could lose a race and leave a stale
  // clipboard. Gated synchronously so browsers still write inside the gesture.
  if (isTauri && (await copyWithTauriClipboard(text))) {
    return true;
  }

  if (typeof navigator?.clipboard?.writeText === "function") {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      console.warn("Async clipboard API failed, falling back to execCommand", error);
    }
  }

  // execCommand works in Safari during a user gesture.
  return copyWithExecCommand(text);
}

/** Safari drops activation across an await, so the write starts synchronously with a promised
 * payload. A rejecting payload leaves the clipboard untouched. */
export async function copyToClipboardFrom(
  load: () => Promise<string>,
): Promise<boolean> {
  const payload = load();
  // Keep the rejection from being reported as unhandled meanwhile.
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
