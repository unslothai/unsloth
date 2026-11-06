// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports
import { useMaterialStore } from "@/features/chat-background/material";
// eslint-disable-next-line no-restricted-imports
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
// eslint-disable-next-line no-restricted-imports
import { usePlusMenuPrefsStore } from "@/features/chat/stores/plus-menu-prefs-store";
import {
  getSpellCheck,
  setSpellCheck,
  subscribeSpellCheck,
} from "@/lib/spellcheck";
import {
  getTimeFormat,
  isTimeFormat,
  setTimeFormat,
  subscribeTimeFormat,
} from "@/lib/time-format";
import type { ClientPreferencesUpdate } from "../api/device-sync";
import type { PendingSection } from "./client-prefs-pending";

type Prefs = Record<string, unknown>;

/** A preference group kept in the account's client-preferences record. */
export type SyncedSection = {
  pending: PendingSection;
  field: keyof ClientPreferencesUpdate;
  read: () => unknown;
  isDefault: () => boolean;
  apply: (remote: unknown) => void;
  subscribe: (onChange: () => void) => () => void;
};

type PersistedStore<S> = {
  getState: () => S;
  getInitialState: () => S;
  setState: (state: S) => void;
  subscribe: (listener: (state: S, prev: S) => void) => () => void;
  persist: {
    getOptions: () => {
      partialize?: (state: S) => unknown;
      merge?: (persisted: unknown, current: S) => S;
    };
  };
};

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

function persistedFields<S>(store: PersistedStore<S>, state: S, local: string[]): Prefs {
  const { partialize } = store.persist.getOptions();
  const saved = (partialize ? partialize(state) : state) as Prefs;
  return Object.fromEntries(
    Object.entries(saved).filter(
      ([key, value]) => typeof value !== "function" && !local.includes(key),
    ),
  );
}

/** A zustand persist store, applied through its own merge so stored values are normalised the
 *  same way a localStorage record is. `local` names fields that stay on this device. */
function persistedSection<S>(
  pending: PendingSection,
  field: keyof ClientPreferencesUpdate,
  store: PersistedStore<S>,
  local: string[] = [],
): SyncedSection {
  const read = () => persistedFields(store, store.getState(), local);
  const defaults = persistedFields(store, store.getInitialState(), local);
  return {
    pending,
    field,
    read,
    isDefault: () => same(read(), defaults),
    apply: (remote) => {
      if (!remote || typeof remote !== "object" || Array.isArray(remote)) return;
      const remoteFields = Object.fromEntries(
        Object.entries(remote as Prefs).filter(([key]) => !local.includes(key)),
      );
      const current = store.getState();
      const merge = store.persist.getOptions().merge;
      // The current persisted fields fill in anything the record lacks (fields added later).
      const incoming = { ...persistedFields(store, current, []), ...remoteFields };
      store.setState(
        merge ? merge(incoming, current) : ({ ...current, ...incoming } as S),
      );
    },
    subscribe: (onChange) => {
      let last = read();
      return store.subscribe(() => {
        const next = read();
        if (same(next, last)) return;
        last = next;
        onChange();
      });
    },
  };
}

export const SYNCED_SECTIONS: SyncedSection[] = [
  // Whether the files panel is open is layout state for this window, not a preference.
  persistedSection("chat", "chat", useChatPreferencesStore, ["filesPanelCollapsed"]),
  persistedSection("plusMenu", "plusMenu", usePlusMenuPrefsStore),
  persistedSection("material", "material", useMaterialStore),
  {
    pending: "timeFormat",
    field: "timeFormat",
    read: getTimeFormat,
    isDefault: () => getTimeFormat() === "system",
    apply: (remote) => {
      if (isTimeFormat(remote) && remote !== getTimeFormat()) setTimeFormat(remote);
    },
    subscribe: subscribeTimeFormat,
  },
  {
    pending: "spellcheck",
    field: "spellcheck",
    read: getSpellCheck,
    isDefault: () => getSpellCheck(),
    apply: (remote) => {
      if (typeof remote === "boolean" && remote !== getSpellCheck()) setSpellCheck(remote);
    },
    subscribe: subscribeSpellCheck,
  },
];
