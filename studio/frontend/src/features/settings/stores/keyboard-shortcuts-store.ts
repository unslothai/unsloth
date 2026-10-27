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
  isMacPlatform,
  isShortcutId,
  matchesBinding,
  parseBinding,
  // Explicit extension: imported directly by the node test runner.
} from "../lib/keyboard-shortcuts.ts";

export const KEYBOARD_SHORTCUTS_STORAGE_KEY = "unsloth_keyboard_shortcuts";
const STORAGE_KEY = KEYBOARD_SHORTCUTS_STORAGE_KEY;

/** Per-slot delta. An absent slot means "use the shipped default"; `disabled` keeps the chords but fires none. */
export type ShortcutOverrideEntry = Partial<
  Record<ShortcutSlot, string | null>
> & { disabled?: true };

/** Deltas only, so changed defaults reach untouched rows. `null` = unassigned. */
export type ShortcutOverrides = Partial<
  Record<ShortcutId, ShortcutOverrideEntry>
>;

/** Older builds stored `id -> string | null`; read that as the primary slot. */
function normalizeEntry(value: unknown): ShortcutOverrideEntry | null {
  // An old null cleared the single chord, so clear both, or a newly shipped alternate would fire.
  if (value === null) return { primary: null, alternate: null };
  // An old rebind never saw an alternate, so that slot takes the shipped default.
  if (typeof value === "string") return { primary: value };
  if (!value || typeof value !== "object") return null;
  const entry: ShortcutOverrideEntry = {};
  for (const slot of SHORTCUT_SLOTS) {
    const slotValue = (value as Record<string, unknown>)[slot];
    if (slotValue === null || typeof slotValue === "string") {
      entry[slot] = slotValue;
    }
  }
  if ((value as Record<string, unknown>).disabled === true) {
    entry.disabled = true;
  }
  return Object.keys(entry).length > 0 ? entry : null;
}

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

function withSlot(
  overrides: ShortcutOverrides,
  id: ShortcutId,
  slot: ShortcutSlot,
  value: string | null | undefined,
): ShortcutOverrides {
  const entry: ShortcutOverrideEntry = { ...overrides[id] };
  if (value === undefined) delete entry[slot];
  else entry[slot] = value;
  return withEntry(overrides, id, entry);
}

function withEntry(
  overrides: ShortcutOverrides,
  id: ShortcutId,
  entry: ShortcutOverrideEntry,
): ShortcutOverrides {
  const next = { ...overrides };
  if (Object.keys(entry).length === 0) delete next[id];
  else next[id] = entry;
  return next;
}

export function assignedBinding(
  overrides: ShortcutOverrides,
  id: ShortcutId,
  slot: ShortcutSlot = "primary",
): string | null {
  const entry = overrides[id];
  if (entry && Object.hasOwn(entry, slot)) return entry[slot] ?? null;
  return defaultBindingFor(SHORTCUT_DEF_BY_ID[id], slot);
}

export function isShortcutEnabled(
  overrides: ShortcutOverrides,
  id: ShortcutId,
): boolean {
  return overrides[id]?.disabled !== true;
}

/** The binding actually in force for one slot of `id`: none while the action is switched off. */
export function resolveBinding(
  overrides: ShortcutOverrides,
  id: ShortcutId,
  slot: ShortcutSlot = "primary",
): string | null {
  if (!isShortcutEnabled(overrides, id)) return null;
  return assignedBinding(overrides, id, slot);
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

/** Registry order decides a contested chord, not mount order; an action's two slots are one claim. */
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

/** The tab flags clashes rather than refusing the edit. */
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

/** For handlers running before the window listener (the composer's Enter), using the same
 * ownership rule as useShortcut. */
export function shortcutMatchingEvent<Id extends ShortcutId>(
  overrides: ShortcutOverrides,
  ids: readonly Id[],
  event: Parameters<typeof matchesBinding>[1],
  mac = isMacPlatform(),
): Id | null {
  for (const id of ids) {
    for (const slot of SHORTCUT_SLOTS) {
      const value = resolveBinding(overrides, id, slot);
      if (!value || shortcutOwningBinding(overrides, value) !== id) continue;
      const binding = parseBinding(value);
      if (binding && matchesBinding(binding, event, mac)) return id;
    }
  }
  return null;
}

interface KeyboardShortcutsState {
  overrides: ShortcutOverrides;
  /** Assign a serialized binding, e.g. "Mod+Shift+KeyO"; also switches the action back on. */
  setBinding: (id: ShortcutId, slot: ShortcutSlot, value: string) => void;
  setEnabled: (id: ShortcutId, enabled: boolean) => void;
  /** Leave the slot with no binding at all. */
  clearBinding: (id: ShortcutId, slot: ShortcutSlot) => void;
  resetBinding: (id: ShortcutId, slot: ShortcutSlot) => void;
  resetAction: (id: ShortcutId) => void;
  resetAll: () => void;
}

export const useKeyboardShortcutsStore = create<KeyboardShortcutsState>(
  (set) => ({
    overrides: loadOverrides(),
    setBinding: (id, slot, value) =>
      set((state) => {
        const { disabled: _, ...entry } = state.overrides[id] ?? {};
        const overrides = withEntry(state.overrides, id, {
          ...entry,
          [slot]: value,
        });
        persist(overrides);
        return { overrides };
      }),
    setEnabled: (id, enabled) =>
      set((state) => {
        if (isShortcutEnabled(state.overrides, id) === enabled) return state;
        const { disabled: _, ...entry } = state.overrides[id] ?? {};
        const overrides = withEntry(
          state.overrides,
          id,
          enabled ? entry : { ...entry, disabled: true },
        );
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

// Else this tab's next edit writes its stale copy over another tab's. A null key is localStorage.clear().
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === STORAGE_KEY || event.key === null) {
      useKeyboardShortcutsStore.setState({ overrides: loadOverrides() });
    }
  });
}

/** Binding in force right now, for non-React callers (keydown handlers). */
export function currentBinding(
  id: ShortcutId,
  slot: ShortcutSlot = "primary",
): string | null {
  return resolveBinding(useKeyboardShortcutsStore.getState().overrides, id, slot);
}
