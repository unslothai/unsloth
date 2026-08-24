// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useRef } from "react";
import {
  SHORTCUT_SLOTS,
  activationBelongsToFocus,
  type ShortcutBinding,
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

/** The chat composer's textarea, which some chords are allowed to fire from. */
export const COMPOSER_INPUT_SELECTOR = ".aui-composer-input";

/** Exported for the test: the gate is easier to pin here than through React. */
export function isTextEntryFocused(exceptFor?: string): boolean {
  if (typeof document === "undefined") return false;
  const el = document.activeElement as HTMLElement | null;
  const tag = el?.tagName;
  const typing =
    tag === "INPUT" || tag === "TEXTAREA" || Boolean(el?.isContentEditable);
  if (!typing) return false;
  // A named field does not type this chord, so it does not shield it either.
  return !(exceptFor && el?.matches(exceptFor) === true);
}

export interface UseShortcutOptions {
  /** Skip registration entirely (route gating, dialogs). Defaults to true. */
  enabled?: boolean;
  /**
   * Ignore the chord while a text field has focus. For chords that would
   * otherwise steal a keystroke the composer wants.
   */
  skipInTextFields?: boolean;
  /**
   * A selector for text fields the gate above does not apply to, for a chord
   * that types nothing in them. Escape in the composer is the case: it leaves
   * the text alone, so a prompt still focused from the message that opened a
   * tool request must not be what stops the request being declined.
   */
  textFieldException?: string;
  /**
   * Run again on each auto-repeat while the chord is held. For walking a list,
   * where holding is the gesture. Everything else is one-shot: a toggle held
   * past the repeat delay would land wherever the user let go.
   */
  repeats?: boolean;
}

/**
 * The chords `id` answers to right now, as a stable joined key so the effect
 * below re-runs only when one of them actually changes.
 */
function useBindingValues(id: ShortcutId): string {
  return useKeyboardShortcutsStore((s) =>
    SHORTCUT_SLOTS.map((slot) => resolveBinding(s.overrides, id, slot) ?? "")
      .join("\0"),
  );
}

/**
 * Run `handler` when the user presses either chord `id` is currently bound to.
 * Bindings are read from the shortcuts store, so an edit in Settings takes
 * effect without a reload; an unassigned action registers nothing.
 */
export function useShortcut(
  id: ShortcutId,
  handler: (event: KeyboardEvent) => void,
  options: UseShortcutOptions = {},
): void {
  const {
    enabled = true,
    skipInTextFields = false,
    textFieldException,
    repeats = false,
  } = options;
  const values = useBindingValues(id);
  // A chord claimed by two actions is consumed by whichever listener runs
  // first, so a slot only registers when this action owns it. Otherwise the
  // winner would follow mount order and change from route to route.
  const ownedFlags = useKeyboardShortcutsStore((s) =>
    SHORTCUT_SLOTS.map((slot) => {
      const value = resolveBinding(s.overrides, id, slot);
      return value && shortcutOwningBinding(s.overrides, value) === id
        ? "1"
        : "0";
    }).join(""),
  );
  const bindings = useMemo(() => {
    const out: ShortcutBinding[] = [];
    values.split("\0").forEach((value, index) => {
      if (ownedFlags[index] !== "1") return;
      const parsed = parseBinding(value);
      if (parsed) out.push(parsed);
    });
    return out;
  }, [values, ownedFlags]);
  // The handler usually closes over fresh props, so keep it in a ref rather
  // than tearing down and re-adding the listener on every render.
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    if (bindings.length === 0 || !enabled) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      const hit = bindings.find((binding) => matchesBinding(binding, event));
      if (!hit) return;
      if (skipInTextFields && isTextEntryFocused(textFieldException)) return;
      // The focused control keeps its own Enter or Space.
      if (
        typeof document !== "undefined" &&
        activationBelongsToFocus(hit, document.activeElement)
      ) {
        return;
      }
      event.preventDefault();
      // Held past the OS repeat delay, the chord arrives again and again. The
      // chord is still ours, so it stays consumed, but only an action that
      // asked for repeats runs on them.
      if (event.repeat && !repeats) return;
      handlerRef.current(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [bindings, enabled, skipInTextFields, textFieldException, repeats]);
}

/**
 * Label for the primary chord `id` is bound to right now, in the platform's own
 * notation, or null when the slot is unassigned. Hints that render the shipped
 * default instead would tell the user to press a chord that stopped working the
 * moment they rebound or cleared the action in Settings.
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

/** Both chords, for surfaces that want to show the alternate too. */
export function useShortcutLabels(id: ShortcutId): string[] {
  const values = useBindingValues(id);
  return useMemo(
    () =>
      values
        .split("\0")
        .map((value) => parseBinding(value))
        .filter((binding): binding is ShortcutBinding => binding !== null)
        .map((binding) => formatBindingLabel(binding)),
    [values],
  );
}
