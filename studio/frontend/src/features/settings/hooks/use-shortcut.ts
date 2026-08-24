// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useRef } from "react";
import {
  type ShortcutId,
  formatBindingLabel,
  matchesBinding,
  parseBinding,
} from "../lib/keyboard-shortcuts";
import {
  resolveBinding,
  shortcutOwningBinding,
  useKeyboardShortcutsStore,
} from "../stores/keyboard-shortcuts-store";

function isTextEntryFocused(): boolean {
  if (typeof document === "undefined") return false;
  const el = document.activeElement as HTMLElement | null;
  const tag = el?.tagName;
  return (
    tag === "INPUT" || tag === "TEXTAREA" || Boolean(el?.isContentEditable)
  );
}

export interface UseShortcutOptions {
  /** Skip registration entirely (route gating, dialogs). Defaults to true. */
  enabled?: boolean;
  /**
   * Ignore the chord while a text field has focus. For chords that would
   * otherwise steal a keystroke the composer wants.
   */
  skipInTextFields?: boolean;
}

/**
 * Run `handler` when the user presses whatever chord `id` is currently bound
 * to. The binding is read from the shortcuts store, so an edit in Settings
 * takes effect without a reload; an unassigned action registers nothing.
 */
export function useShortcut(
  id: ShortcutId,
  handler: (event: KeyboardEvent) => void,
  options: UseShortcutOptions = {},
): void {
  const { enabled = true, skipInTextFields = false } = options;
  const value = useKeyboardShortcutsStore((s) =>
    resolveBinding(s.overrides, id),
  );
  // A chord claimed by two actions is consumed by whichever listener runs
  // first, so only its owner registers. Otherwise the winner would follow mount
  // order and change from route to route.
  const owned = useKeyboardShortcutsStore(
    (s) =>
      shortcutOwningBinding(s.overrides, resolveBinding(s.overrides, id)) ===
      id,
  );
  const binding = useMemo(() => parseBinding(value), [value]);
  // The handler usually closes over fresh props, so keep it in a ref rather
  // than tearing down and re-adding the listener on every render.
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    if (!binding || !enabled || !owned) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (!matchesBinding(binding, event)) return;
      if (skipInTextFields && isTextEntryFocused()) return;
      event.preventDefault();
      handlerRef.current(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [binding, enabled, owned, skipInTextFields]);
}

/**
 * Label for the chord `id` is bound to right now, in the platform's own
 * notation, or null when the action is unassigned. Hints that render the
 * shipped default instead would tell the user to press a chord that stopped
 * working the moment they rebound or cleared the action in Settings.
 */
export function useShortcutLabel(id: ShortcutId): string | null {
  const value = useKeyboardShortcutsStore((s) =>
    resolveBinding(s.overrides, id),
  );
  return useMemo(() => {
    const binding = parseBinding(value);
    return binding ? formatBindingLabel(binding) : null;
  }, [value]);
}
