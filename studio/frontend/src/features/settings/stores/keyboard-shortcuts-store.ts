// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  SHORTCUT_DEFS,
  SHORTCUT_DEF_BY_ID,
  type ShortcutId,
  isShortcutId,
  // Explicit extension: this module is imported directly by the node test
  // runner, which does not do bundler-style resolution.
} from "../lib/keyboard-shortcuts.ts";

/**
 * Exported so "Reset all local preferences" in General can clear it by
 * reference rather than by a second copy of the literal.
 */
export const KEYBOARD_SHORTCUTS_STORAGE_KEY = "unsloth_keyboard_shortcuts";
const STORAGE_KEY = KEYBOARD_SHORTCUTS_STORAGE_KEY;

/**
 * Deltas only, so a changed default still reaches rows the user never touched.
 * `null` means "unassigned"; an absent key means "use the default".
 */
export type ShortcutOverrides = Partial<Record<ShortcutId, string | null>>;

function loadOverrides(): ShortcutOverrides {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return {};
    const out: ShortcutOverrides = {};
    for (const [id, value] of Object.entries(
      parsed as Record<string, unknown>,
    )) {
      // Drop ids from an older build so a removed action cannot resurrect.
      if (!isShortcutId(id)) continue;
      if (value === null || typeof value === "string") out[id] = value;
    }
    return out;
  } catch {
    return {};
  }
}

function persist(overrides: ShortcutOverrides): void {
  if (typeof window === "undefined") return;
  try {
    if (Object.keys(overrides).length === 0) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
  } catch {
    // Private mode / quota: customization is a preference, not worth failing on.
  }
}

/** The binding actually in force for `id`, given a set of overrides. */
export function resolveBinding(
  overrides: ShortcutOverrides,
  id: ShortcutId,
): string | null {
  return Object.hasOwn(overrides, id)
    ? (overrides[id] ?? null)
    : SHORTCUT_DEF_BY_ID[id].defaultBinding;
}

export function resolveAllBindings(
  overrides: ShortcutOverrides,
): Record<ShortcutId, string | null> {
  return Object.fromEntries(
    SHORTCUT_DEFS.map((def) => [def.id, resolveBinding(overrides, def.id)]),
  ) as Record<ShortcutId, string | null>;
}

/**
 * Which action a chord runs when more than one claims it. The first window
 * listener consumes the event, so without a rule the winner would be whichever
 * component mounted first, and that varies by route. Registry order decides.
 */
export function shortcutOwningBinding(
  overrides: ShortcutOverrides,
  value: string | null,
): ShortcutId | null {
  if (!value) return null;
  for (const def of SHORTCUT_DEFS) {
    if (resolveBinding(overrides, def.id) === value) return def.id;
  }
  return null;
}

/**
 * Ids sharing a binding with at least one other action. Only the owner above
 * runs, so the tab flags the clash rather than silently refusing the edit.
 */
export function findConflicts(overrides: ShortcutOverrides): Set<ShortcutId> {
  const byValue = new Map<string, ShortcutId[]>();
  for (const def of SHORTCUT_DEFS) {
    const value = resolveBinding(overrides, def.id);
    if (!value) continue;
    const ids = byValue.get(value);
    if (ids) ids.push(def.id);
    else byValue.set(value, [def.id]);
  }
  const out = new Set<ShortcutId>();
  for (const ids of byValue.values()) {
    if (ids.length > 1) for (const id of ids) out.add(id);
  }
  return out;
}

interface KeyboardShortcutsState {
  overrides: ShortcutOverrides;
  /** Assign a serialized binding, e.g. "Mod+Shift+KeyO". */
  setBinding: (id: ShortcutId, value: string) => void;
  /** Leave the action with no binding at all. */
  clearBinding: (id: ShortcutId) => void;
  /** Drop the override so the shipped default applies again. */
  resetBinding: (id: ShortcutId) => void;
  resetAll: () => void;
}

export const useKeyboardShortcutsStore = create<KeyboardShortcutsState>(
  (set) => ({
    overrides: loadOverrides(),
    setBinding: (id, value) =>
      set((state) => {
        const overrides = { ...state.overrides, [id]: value };
        persist(overrides);
        return { overrides };
      }),
    clearBinding: (id) =>
      set((state) => {
        const overrides = { ...state.overrides, [id]: null };
        persist(overrides);
        return { overrides };
      }),
    resetBinding: (id) =>
      set((state) => {
        if (!Object.hasOwn(state.overrides, id)) return state;
        const overrides = { ...state.overrides };
        delete overrides[id];
        persist(overrides);
        return { overrides };
      }),
    resetAll: () =>
      set(() => {
        persist({});
        return { overrides: {} };
      }),
  }),
);

/** Binding in force right now, for non-React callers (keydown handlers). */
export function currentBinding(id: ShortcutId): string | null {
  return resolveBinding(useKeyboardShortcutsStore.getState().overrides, id);
}
