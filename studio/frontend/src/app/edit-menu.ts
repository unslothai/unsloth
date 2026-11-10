// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Edit > Undo / Redo on the macOS desktop menu (src-tauri/src/app_menu.rs `EDIT_HISTORY`). */
export type EditMenuAction = "edit-undo" | "edit-redo";

type Focused = {
  tagName?: string;
  type?: string;
  readOnly?: boolean;
  isContentEditable?: boolean;
} | null;

const TEXT_INPUTS = new Set([
  "",
  "text",
  "search",
  "url",
  "tel",
  "email",
  "password",
  "number",
]);

/** Whether the native text undo applies, as it did under the stock Edit menu. */
export function editsText(el: Focused): boolean {
  if (!el) return false;
  if (el.isContentEditable) return true;
  const tag = (el.tagName ?? "").toUpperCase();
  if (tag === "TEXTAREA") return !el.readOnly;
  if (tag === "INPUT")
    return !el.readOnly && TEXT_INPUTS.has((el.type ?? "").toLowerCase());
  return false;
}

/** The keydown the page would have seen had the menu not taken it. */
export function editMenuKey(action: EditMenuAction): KeyboardEventInit {
  return {
    key: "z",
    code: "KeyZ",
    metaKey: true,
    shiftKey: action === "edit-redo",
    bubbles: true,
    cancelable: true,
  };
}

export function runEditMenuAction(
  action: EditMenuAction,
  doc: Document,
  makeKey: (init: KeyboardEventInit) => Event = (init) =>
    new KeyboardEvent("keydown", init),
): void {
  const focused = doc.activeElement as HTMLElement | null;
  if (editsText(focused)) {
    doc.execCommand(action === "edit-redo" ? "redo" : "undo");
    return;
  }
  (focused ?? doc.body).dispatchEvent(makeKey(editMenuKey(action)));
}

/** Listens for the menu's Undo / Redo; returns the cleanup. */
export function listenEditMenu(): () => void {
  let disposed = false;
  let unlisten: (() => void) | undefined;
  void import("@tauri-apps/api/event")
    .then(({ listen }) =>
      listen<string>("app-menu-action", ({ payload }) => {
        if (payload === "edit-undo" || payload === "edit-redo")
          runEditMenuAction(payload, document);
      }),
    )
    .then((cleanup) => {
      if (disposed) cleanup();
      else unlisten = cleanup;
    })
    .catch(() => undefined);
  return () => {
    disposed = true;
    unlisten?.();
  };
}
