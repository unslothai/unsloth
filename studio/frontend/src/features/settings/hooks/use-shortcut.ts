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

/**
 * A keydown the IME is still composing with. Escape cancels a candidate and
 * Enter commits one, so without this, declining a tool call takes the Escape a
 * CJK user aimed at the candidate window. Both signals: isComposing on WebKit,
 * the legacy 229 on Chromium.
 */
export function isImeComposing(event: KeyboardEvent): boolean {
  return event.isComposing || event.keyCode === 229;
}

/**
 * Whether `selector`'s element is the foreground rather than under a modal. A
 * dialog leaves the route mounted, so a route-gated chord still fires behind
 * it, and `enabled` is read at render, which a dialog opening need not trigger.
 * So anything irreversible asks here, at press time.
 */
export function isSurfaceInForeground(selector: string): boolean {
  if (typeof document === "undefined") return false;
  // Every match, not the first: Compare keeps the base view mounted and inert
  // behind the panes, so the first composer found is the hidden one. Radix
  // marks the rest of the page aria-hidden (older React: inert) for a modal's
  // life, which is the general signal, not a per-dialog store.
  return [...document.querySelectorAll(selector)].some(
    (el) => !el.closest('[aria-hidden="true"], [inert]'),
  );
}

/**
 * Whether every `selector` match sits under a modal. Not the complement of the
 * check above: an unrendered surface is not backgrounded, and reading it that
 * way would kill a chord on a layout that never renders the element. The mobile
 * sidebar is the case, unmounted while its drawer is closed.
 */
export function isSurfaceBackgrounded(selector: string): boolean {
  if (typeof document === "undefined") return false;
  const found = [...document.querySelectorAll(selector)];
  return (
    found.length > 0 &&
    found.every((el) => el.closest('[aria-hidden="true"], [inert]') !== null)
  );
}

/**
 * Whether pressing `binding` in a focused text field would put something in it.
 * Narrow on purpose: only Escape, the function keys, and anything held with a
 * modifier other than Shift. A caret key inserts nothing but still has an edit
 * to stand aside for, so it counts as typing.
 */
export function typesInTextField(binding: ShortcutBinding): boolean {
  if (binding.mod || binding.ctrl || binding.alt) return false;
  if (binding.code === "Escape") return false;
  return !/^F([1-9]|1[0-9]|2[0-4])$/.test(binding.code);
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

/** The chords `id` answers to now, joined so the effect re-runs only on a real change. */
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
  // winner follows mount order and changes from route to route.
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
      if (isImeComposing(event)) return;
      const hit = bindings.find((binding) => matchesBinding(binding, event));
      if (!hit) return;
      // The exception is for a chord that types nothing there. Decline ships
      // on Escape; rebound to Enter or a letter, the same pass would deny the
      // request as the user edits the prompt.
      const exception = typesInTextField(hit) ? undefined : textFieldException;
      if (skipInTextFields && isTextEntryFocused(exception)) return;
      // The focused control keeps its own Enter or Space.
      if (
        typeof document !== "undefined" &&
        activationBelongsToFocus(hit, document.activeElement)
      ) {
        return;
      }
      event.preventDefault();
      // Held past the OS repeat delay the chord arrives again and again. It
      // stays consumed either way, but only an action that asked for repeats
      // runs on them.
      if (event.repeat && !repeats) return;
      handlerRef.current(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [bindings, enabled, skipInTextFields, textFieldException, repeats]);
}

/**
 * Label for the primary chord `id` is bound to now, in the platform's notation,
 * or null when the slot is unassigned. Rendering the shipped default instead
 * would name a chord that stopped working the moment the user rebound it.
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
