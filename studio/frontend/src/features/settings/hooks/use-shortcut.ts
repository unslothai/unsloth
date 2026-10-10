// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import {
  SHORTCUT_SLOTS,
  type ShortcutBinding,
  type ShortcutId,
  activationBelongsToFocus,
  formatBindingLabel,
  matchesBinding,
  parseBinding,
} from "../lib/keyboard-shortcuts";
import {
  resolveBinding,
  shortcutOwningBinding,
  useKeyboardShortcutsStore,
} from "../stores/keyboard-shortcuts-store";

export const COMPOSER_INPUT_SELECTOR = ".aui-composer-input";

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

/** Escape and Enter belong to the IME while composing; isComposing on WebKit, 229 on Chromium. */
export function isImeComposing(event: KeyboardEvent): boolean {
  return event.isComposing || event.keyCode === 229;
}

/** Checked at press time: a dialog leaves the route mounted and need not re-render `enabled`. */
export function isSurfaceInForeground(selector: string): boolean {
  if (typeof document === "undefined") return false;
  // Every match: Compare keeps an inert base composer mounted; Radix aria-hides the page under a modal.
  return [...document.querySelectorAll(selector)].some(
    (el) => !el.closest('[aria-hidden="true"], [inert]'),
  );
}

/** Not the complement of the above: an unrendered surface (closed mobile sidebar) is not
 * backgrounded. */
export function isSurfaceBackgrounded(selector: string): boolean {
  if (typeof document === "undefined") return false;
  const found = [...document.querySelectorAll(selector)];
  return (
    found.length > 0 &&
    found.every((el) => el.closest('[aria-hidden="true"], [inert]') !== null)
  );
}

/** Only Escape, function keys, and non-Shift modifier chords; caret keys count as typing. */
export function typesInTextField(binding: ShortcutBinding): boolean {
  if (binding.mod || binding.ctrl || binding.alt) return false;
  if (binding.code === "Escape") return false;
  return !/^F([1-9]|1[0-9]|2[0-4])$/.test(binding.code);
}

export interface UseShortcutOptions {
  enabled?: boolean;
  skipInTextFields?: boolean;
  /** Fields exempt from skipInTextFields for a chord that types nothing there (Escape in the
   * composer must still decline a tool request). */
  textFieldException?: string;
  /** Run on auto-repeat; one-shot otherwise, since a held toggle lands wherever released. */
  repeats?: boolean;
  /** Asked at press time before the event is prevented; false leaves the key to the browser. */
  claims?: () => boolean;
}

export interface ShortcutTrigger {
  claims: () => boolean;
  run: () => void;
}

const triggers = new Map<ShortcutId, ShortcutTrigger[]>();
const triggerListeners = new Set<() => void>();
const notifyTriggerListeners = () => triggerListeners.forEach((listener) => listener());

/** Exported for the test. Returns the unregister. */
export function registerShortcutTrigger(
  id: ShortcutId,
  trigger: ShortcutTrigger,
): () => void {
  triggers.set(id, [...(triggers.get(id) ?? []), trigger]);
  notifyTriggerListeners();
  return () => {
    triggers.set(id, (triggers.get(id) ?? []).filter((t) => t !== trigger));
    notifyTriggerListeners();
  };
}

/** False when nothing took it. */
export function triggerShortcut(id: ShortcutId): boolean {
  const trigger = [...(triggers.get(id) ?? [])].reverse().find((t) => t.claims());
  trigger?.run();
  return trigger !== undefined;
}

// Modals show as aria-hidden, inert, or only a body-portaled backdrop when an aria-live region exists.
let modalObserver: MutationObserver | null = null;
function subscribeTriggers(listener: () => void, watchModals: boolean): () => void {
  triggerListeners.add(listener);
  if (watchModals && !modalObserver && typeof MutationObserver !== "undefined") {
    modalObserver = new MutationObserver(notifyTriggerListeners);
    modalObserver.observe(document.documentElement, {
      subtree: true,
      attributes: true,
      attributeFilter: ["aria-hidden", "inert"],
    });
    modalObserver.observe(document.body, { childList: true });
  }
  return () => {
    triggerListeners.delete(listener);
    if (triggerListeners.size === 0) {
      modalObserver?.disconnect();
      modalObserver = null;
    }
  };
}

/** `watchModals` also re-asks when a modal opens or closes (desktop menu only). */
export function useShortcutAvailable(id: ShortcutId, watchModals: boolean): boolean {
  const subscribe = useCallback(
    (listener: () => void) => subscribeTriggers(listener, watchModals),
    [watchModals],
  );
  return useSyncExternalStore(subscribe, () =>
    (triggers.get(id) ?? []).some((t) => t.claims()),
  );
}

/** Joined so the effect re-runs only on a real change. */
function useBindingValues(id: ShortcutId): string {
  return useKeyboardShortcutsStore((s) =>
    SHORTCUT_SLOTS.map(
      (slot) => resolveBinding(s.overrides, id, slot) ?? "",
    ).join("\0"),
  );
}

/** Bindings come from the store, so Settings edits apply without reload. */
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
    claims,
  } = options;
  const values = useBindingValues(id);
  // A chord claimed by two actions registers only for its owner, not by mount order.
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
  const latestRef = useRef({ handler, claims });
  latestRef.current = { handler, claims };

  // Registered even with no chord bound: the menu still reaches the action.
  useEffect(() => {
    if (!enabled) return;
    return registerShortcutTrigger(id, {
      claims: () => latestRef.current.claims?.() !== false,
      run: () => latestRef.current.handler(new KeyboardEvent("keydown")),
    });
  }, [id, enabled]);

  useEffect(() => {
    if (bindings.length === 0 || !enabled) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (isImeComposing(event)) return;
      const hit = bindings.find((binding) => matchesBinding(binding, event));
      if (!hit) return;
      // Only for a chord that types nothing there; rebound to Enter it would deny while editing.
      const exception = typesInTextField(hit) ? undefined : textFieldException;
      if (skipInTextFields && isTextEntryFocused(exception)) return;
      if (
        typeof document !== "undefined" &&
        activationBelongsToFocus(hit, document.activeElement)
      ) {
        return;
      }
      // Before preventDefault: an unclaimed chord must reach the browser untouched.
      if (latestRef.current.claims?.() === false) return;
      event.preventDefault();
      if (event.repeat && !repeats) return;
      latestRef.current.handler(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [bindings, enabled, skipInTextFields, textFieldException, repeats]);
}

/** Null when unassigned; never the shipped default, which may have been rebound. */
export function useShortcutLabel(id: ShortcutId): string | null {
  const value = useKeyboardShortcutsStore((s) =>
    resolveBinding(s.overrides, id),
  );
  return useMemo(() => {
    const binding = parseBinding(value);
    return binding ? formatBindingLabel(binding) : null;
  }, [value]);
}

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
