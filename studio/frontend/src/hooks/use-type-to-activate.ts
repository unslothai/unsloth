// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { useSettingsDialogStore } from "@/features/settings";

/** Focusing during keydown lets the default action insert the character there. */
export function useTypeToActivate(): void {
  const settingsOpen = useSettingsDialogStore((s) => s.open);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      // AltGr (Ctrl+Alt) and macOS Option type characters, so they are not shortcuts.
      const altGraph =
        typeof e.getModifierState === "function" &&
        e.getModifierState("AltGraph");
      if (e.metaKey) return;
      if (e.ctrlKey && !altGraph) return;
      if (
        e.altKey &&
        !altGraph &&
        !(IS_MAC && (codePointCount(e.key) === 1 || e.key === "Dead"))
      ) {
        return;
      }
      // IME start and dead keys must reach the target so composition begins there.
      const compositionStart =
        e.keyCode === 229 || e.isComposing || e.key === "Dead";
      if (
        !compositionStart &&
        (codePointCount(e.key) !== 1 || e.key === " ")
      ) {
        return;
      }

      if (isEditable(document.activeElement)) return;
      if (hasOpenOverlay(settingsOpen)) return;

      // Radix keeps the settings dialog mounted while closing, so exclude its search then.
      const target = firstVisible(
        settingsOpen
          ? '[data-type-to-activate="settings-search"]'
          : '[data-type-to-activate]:not([data-type-to-activate="settings-search"])',
      );
      target?.focus();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [settingsOpen]);
}

// From the browser, not the server's device_type: remote sessions can differ.
const IS_MAC =
  typeof navigator !== "undefined" &&
  (navigator.platform?.toLowerCase().includes("mac") ||
    navigator.userAgent.toLowerCase().includes("mac"));

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

/** Surrogate pairs count as one character. */
function codePointCount(value: string): number {
  return Array.from(value).length;
}

function isEditable(el: Element | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el instanceof HTMLTextAreaElement) return !el.readOnly;
  if (el instanceof HTMLSelectElement) return true;
  if (el.isContentEditable) return true;
  if (el instanceof HTMLInputElement) {
    return TEXT_INPUT_TYPES.has(el.type) && !el.readOnly;
  }
  return false;
}

/** Any dialog except settings owns the keystroke. */
function hasOpenOverlay(settingsOpen: boolean): boolean {
  const notSettings = settingsOpen ? ":not([data-settings-dialog])" : "";
  const overlaySelector = [
    '[data-slot="popover-content"][data-state="open"]',
    '[data-slot="combobox-content"][data-open]',
    '[data-slot="image-zoom-overlay"]',
    "[data-blocking-screen]",
    '[role="menu"][data-state="open"]',
    '[role="listbox"][data-state="open"]',
    `[role="dialog"][data-state="open"]${notSettings}`,
    `[role="dialog"][aria-modal="true"]:not([data-state="closed"])${notSettings}`,
    `[role="alertdialog"][data-state="open"]${notSettings}`,
    `[role="alertdialog"][aria-modal="true"]:not([data-state="closed"])${notSettings}`,
  ].join(", ");
  return document.querySelector(overlaySelector) !== null;
}

function firstVisible(selector: string): HTMLElement | null {
  for (const el of document.querySelectorAll<HTMLElement>(selector)) {
    if (el.matches(":disabled") || el.closest("[inert]")) continue;
    if (el.getClientRects().length > 0) return el;
  }
  return null;
}
