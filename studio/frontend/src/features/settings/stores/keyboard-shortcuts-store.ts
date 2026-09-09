// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  SHORTCUT_DEFS,
  SHORTCUT_DEF_BY_ID,
  SHORTCUT_SLOTS,
  type ShortcutId,
  type ShortcutSlot,
  defaultBindingFor,
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

/** Per-slot delta. An absent slot means "use the shipped default". */
export type ShortcutOverrideEntry = Partial<
  Record<ShortcutSlot, string | null>
>;

/**
 * Deltas only, so a changed default still reaches rows the user never touched.
 * `null` means "unassigned"; an absent key means "use the default".
 */
export type ShortcutOverrides = Partial<
  Record<ShortcutId, ShortcutOverrideEntry>
>;

/** Builds before alternates stored `id -> string | null`. Read that as the
 *  primary slot, or every existing customization reverts to defaults. */
function normalizeEntry(value: unknown): ShortcutOverrideEntry | null {
  // A null back then cleared the action, which had one chord, so it has to
  // clear both now. Left to the primary alone it would pick up whatever
  // alternate has shipped since and start answering again, which is the
  // opposite of what the user asked for.
  if (value === null) return { primary: null, alternate: null };
  // A rebind is different: the user chose a primary and never saw an
  // alternate, so that slot is untouched and takes the shipped default, the
  // same as it does for someone who never opened the tab.
  if (typeof value === "string") return { primary: value };
  if (!value || typeof value !== "object") return null;
  const entry: ShortcutOverrideEntry = {};
  for (const slot of SHORTCUT_SLOTS) {
    const slotValue = (value as Record<string, unknown>)[slot];
    if (slotValue === null || typeof slotValue === "string") {
      entry[slot] = slotValue;
    }
  }
  return Object.keys(entry).length > 0 ? entry : null;
}

/** Parsed JSON to overrides, tolerating both the current and the older shape. */
export function migrateStoredOverrides(parsed: unknown): ShortcutOverrides {
  if (!parsed || typeof parsed !== "object") return {};
  const out: ShortcutOverrides = {};
  for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
    // Drop ids from an older build so a removed action cannot resurrect.
    if (!isShortcutId(id)) continue;
    const entry = normalizeEntry(value);
    if (entry) out[id] = entry;
  }
  return out;
}

function loadOverrides(): ShortcutOverrides {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    return migrateStoredOverrides(JSON.parse(raw) as unknown);
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

/** Write one slot, dropping the whole entry once nothing is overridden. */
function withSlot(
  overrides: ShortcutOverrides,
  id: ShortcutId,
  slot: ShortcutSlot,
  value: string | null | undefined,
): ShortcutOverrides {
  const entry: ShortcutOverrideEntry = { ...overrides[id] };
  if (value === undefined) delete entry[slot];
  else entry[slot] = value;
  const next = { ...overrides };
  if (Object.keys(entry).length === 0) delete next[id];
  else next[id] = entry;
  return next;
}

/** The binding actually in force for one slot of `id`. */
export function resolveBinding(
  overrides: ShortcutOverrides,
  id: ShortcutId,
  slot: ShortcutSlot = "primary",
): string | null {
  const entry = overrides[id];
  if (entry && Object.hasOwn(entry, slot)) return entry[slot] ?? null;
  return defaultBindingFor(SHORTCUT_DEF_BY_ID[id], slot);
}

/** Both chords in force for `id`, in slot order. */
export function resolveBindings(
  overrides: ShortcutOverrides,
  id: ShortcutId,
): Record<ShortcutSlot, string | null> {
  return {
    primary: resolveBinding(overrides, id, "primary"),
    alternate: resolveBinding(overrides, id, "alternate"),
  };
}

/** True when this slot carries a user edit rather than the shipped default. */
export function isSlotOverridden(
  overrides: ShortcutOverrides,
  id: ShortcutId,
  slot: ShortcutSlot,
): boolean {
  const entry = overrides[id];
  return Boolean(entry && Object.hasOwn(entry, slot));
}

export function resolveAllBindings(
  overrides: ShortcutOverrides,
): Record<ShortcutId, Record<ShortcutSlot, string | null>> {
  return Object.fromEntries(
    SHORTCUT_DEFS.map((d) => [d.id, resolveBindings(overrides, d.id)]),
  ) as Record<ShortcutId, Record<ShortcutSlot, string | null>>;
}

/**
 * Which action a chord runs when more than one claims it. The first window
 * listener consumes the event, so without a rule the winner would be whichever
 * component mounted first, and that varies by route. Registry order decides,
 * and an action's own two slots count as one claim.
 */
export function shortcutOwningBinding(
  overrides: ShortcutOverrides,
  value: string | null,
): ShortcutId | null {
  if (!value) return null;
  for (const d of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      if (resolveBinding(overrides, d.id, slot) === value) return d.id;
    }
  }
  return null;
}

/**
 * Ids sharing a chord with at least one other action, counting either slot on
 * either side. Only the owner above runs, so the tab flags the clash rather
 * than silently refusing the edit.
 */
export function findConflicts(overrides: ShortcutOverrides): Set<ShortcutId> {
  const byValue = new Map<string, Set<ShortcutId>>();
  for (const d of SHORTCUT_DEFS) {
    for (const slot of SHORTCUT_SLOTS) {
      const value = resolveBinding(overrides, d.id, slot);
      if (!value) continue;
      const ids = byValue.get(value);
      if (ids) ids.add(d.id);
      else byValue.set(value, new Set([d.id]));
    }
  }
  const out = new Set<ShortcutId>();
  for (const ids of byValue.values()) {
    if (ids.size > 1) for (const id of ids) out.add(id);
  }
  return out;
}

interface KeyboardShortcutsState {
  overrides: ShortcutOverrides;
  /** Assign a serialized binding, e.g. "Mod+Shift+KeyO". */
  setBinding: (id: ShortcutId, slot: ShortcutSlot, value: string) => void;
  /** Leave the slot with no binding at all. */
  clearBinding: (id: ShortcutId, slot: ShortcutSlot) => void;
  /** Drop the override so the shipped default applies again. */
  resetBinding: (id: ShortcutId, slot: ShortcutSlot) => void;
  /** Drop both of an action's slots. */
  resetAction: (id: ShortcutId) => void;
  resetAll: () => void;
}

export const useKeyboardShortcutsStore = create<KeyboardShortcutsState>(
  (set) => ({
    overrides: loadOverrides(),
    setBinding: (id, slot, value) =>
      set((state) => {
        const overrides = withSlot(state.overrides, id, slot, value);
        persist(overrides);
        return { overrides };
      }),
    clearBinding: (id, slot) =>
      set((state) => {
        const overrides = withSlot(state.overrides, id, slot, null);
        persist(overrides);
        return { overrides };
      }),
    resetBinding: (id, slot) =>
      set((state) => {
        if (!isSlotOverridden(state.overrides, id, slot)) return state;
        const overrides = withSlot(state.overrides, id, slot, undefined);
        persist(overrides);
        return { overrides };
      }),
    resetAction: (id) =>
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
export function currentBinding(
  id: ShortcutId,
  slot: ShortcutSlot = "primary",
): string | null {
  return resolveBinding(useKeyboardShortcutsStore.getState().overrides, id, slot);
}
