// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { useSettingsDialogStore } from "@/features/settings";

/**
 * Type-to-activate: when the user just starts typing (a printable key, no
 * shortcuts) while nothing editable is focused, route the keystroke to the
 * screen's primary input -- the settings search when the settings dialog is
 * open, otherwise the first visible input tagged `data-type-to-activate` (the
 * chat composer, or the Images/Video/Audio prompt). Focusing during keydown
 * lets the browser's default action insert the character.
 */
export function useTypeToActivate(): void {
  const settingsOpen = useSettingsDialogStore((s) => s.open);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      // AltGr (ctrl+alt, e.g. Polish ą) and macOS Option (Option+G → ©) both
      // produce printable characters and are typing intent, not shortcuts.
      const altGraph =
        typeof e.getModifierState === "function" &&
        e.getModifierState("AltGraph");
      // Cmd/Ctrl/Alt combos are shortcuts, not typing intent.
      if (e.metaKey) return;
      if (e.ctrlKey && !altGraph) return;
      if (
        e.altKey &&
        !altGraph &&
        !(IS_MAC && (codePointCount(e.key) === 1 || e.key === "Dead"))
      ) {
        return;
      }
      // IME initiation (keyCode 229 / "Process") and dead keys ("Dead", used
      // to compose accented characters) must still activate the target so
      // composition can begin there. Other keys need a single printable
      // character -- counted by Unicode code point so non-BMP chars (e.g.
      // Osage) are accepted -- and a lone space is scroll/button activation.
      const compositionStart =
        e.keyCode === 229 || e.isComposing || e.key === "Dead";
      if (
        !compositionStart &&
        (codePointCount(e.key) !== 1 || e.key === " ")
      ) {
        return;
      }

      const active = document.activeElement;
      if (isEditable(active)) return;
      if (hasOpenOverlay(settingsOpen)) return;

      // The settings search is hidden on small screens, so pick it through the
      // visibility filter too: a display:none input cannot take focus and the
      // initiating keystroke would be dropped. When settings are closed, its
      // search must be excluded: Radix keeps the dialog mounted during the
      // close animation, so it would otherwise win `firstVisible` and swallow
      // the keystroke.
      const target = firstVisible(
        settingsOpen
          ? '[data-type-to-activate="settings-search"]'
          : '[data-type-to-activate]:not([data-type-to-activate="settings-search"])',
      );
      if (!target) return;
      target.focus();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [settingsOpen]);
}

/**
 * macOS reports Option-based text entry (Option+G → ©) with altKey but no
 * AltGraph; detect it from the browser platform, not the server-reported
 * device_type, which can differ for remote/tunneled sessions.
 */
const IS_MAC =
  typeof navigator !== "undefined" &&
  (navigator.platform?.toLowerCase().includes("mac") ||
    navigator.userAgent.toLowerCase().includes("mac"));

/** Native input types that accept keyboard text entry. */
const TEXT_INPUT_TYPES = new Set([
  "date",
  "datetime-local",
  "email",
  "month",
  "number",
  "password",
  "search",
  "tel",
  "text",
  "time",
  "url",
  "week",
]);

/** Number of Unicode code points, so a surrogate pair counts as one char. */
function codePointCount(value: string): number {
  return Array.from(value).length;
}

function isEditable(el: Element | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  // A focused read-only field cannot consume the printable key either, so it
  // must not block type-to-activate.
  if (el.tagName === "TEXTAREA" && el instanceof HTMLTextAreaElement) {
    return !el.readOnly;
  }
  if (el.tagName === "SELECT") return true;
  if (el.isContentEditable) return true;
  if (
    el.closest('[contenteditable="true"], [contenteditable="plaintext-only"]')
  ) {
    return true;
  }
  if (el.tagName === "INPUT" && el instanceof HTMLInputElement) {
    // Only text-capable inputs consume printable characters; controls like
    // range/checkbox/color, and read-only fields, must not block activation.
    return TEXT_INPUT_TYPES.has(el.type) && !el.readOnly;
  }
  return false;
}

function hasOpenOverlay(settingsOpen: boolean): boolean {
  // An open overlay owns the keystroke: popovers (role-less, tagged
  // `data-slot="popover-content"`), menus, selects, the image lightbox
  // (`data-slot="image-zoom-overlay"`), and any dialog or alertdialog --
  // Radix (data-state) or custom (aria-modal, e.g. the fullscreen Canvas) --
  // other than the settings one (data-settings-dialog).
  const settingsExclusion = settingsOpen ? ":not([data-settings-dialog])" : "";
  const overlaySelector = [
    '[data-slot="popover-content"][data-state="open"]',
    '[data-slot="combobox-content"][data-open]',
    '[data-slot="closing-screen"]',
    '[data-slot="image-zoom-overlay"]',
    '[role="menu"][data-state="open"]',
    '[role="listbox"][data-state="open"]',
    `[role="dialog"][data-state="open"]${settingsExclusion}`,
    `[role="dialog"][aria-modal="true"]:not([data-state="closed"])${settingsExclusion}`,
    `[role="alertdialog"][data-state="open"]${settingsExclusion}`,
    `[role="alertdialog"][aria-modal="true"]:not([data-state="closed"])${settingsExclusion}`,
  ].join(", ");
  return Boolean(document.querySelector(overlaySelector));
}

function firstVisible(selector: string): HTMLElement | null {
  for (const el of document.querySelectorAll<HTMLElement>(selector)) {
    if (isVisible(el)) return el;
  }
  return null;
}

function isVisible(el: HTMLElement): boolean {
  return el.getClientRects().length > 0 && !el.closest("[inert]");
}
