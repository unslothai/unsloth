// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports
import { useBrowserPrefsStore } from "@/features/browser/prefs-store";
// eslint-disable-next-line no-restricted-imports
import { useMaterialStore } from "@/features/chat-background/material";
// eslint-disable-next-line no-restricted-imports
import { startChatBackgroundSync } from "@/features/chat-background/sync";
// eslint-disable-next-line no-restricted-imports
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
// eslint-disable-next-line no-restricted-imports
import { SIDEBAR_SYNC_BASE_KEY } from "@/features/chat/stores/sidebar-organization-keys";
// eslint-disable-next-line no-restricted-imports
import { useSidebarOrganizationStore } from "@/features/chat/stores/sidebar-organization-store";
// eslint-disable-next-line no-restricted-imports
import { usePlusMenuPrefsStore } from "@/features/chat/stores/plus-menu-prefs-store";
import {
  accountTransitionPending,
  BROWSER_ACCOUNT_KEY,
  sameBrowserAccount,
} from "@/lib/account-transition";
import { getSpellCheck, setSpellCheck, subscribeSpellCheck } from "@/lib/spellcheck";
import {
  getTimeFormat,
  setTimeFormat,
  subscribeTimeFormat,
  type TimeFormat,
} from "@/lib/time-format";
import { useEffect } from "react";
import {
  type ClientPreferencesChanges,
  loadClientPreferences,
  saveClientPreferences,
} from "../api/device-sync";
import {
  INTERFACE_SCALE_RANGE,
  useInterfaceScaleStore,
} from "../stores/interface-scale-store";
import {
  markPending,
  type Pending,
  PENDING_SECTIONS,
  readPending,
} from "../lib/client-prefs-pending";

const PUSH_DEBOUNCE_MS = 800;

type Prefs = Record<string, unknown>;

function prefsOf(state: object, skip?: ReadonlySet<string>): Prefs {
  return Object.fromEntries(
    Object.entries(state).filter(
      ([key, value]) => typeof value !== "function" && !skip?.has(key),
    ),
  );
}

export function prefsFromRemote(remote: Prefs, defaults: Prefs): Prefs {
  return Object.fromEntries(
    Object.entries(remote).filter(([key, value]) => {
      const fallback = defaults[key];
      return (
        key in defaults &&
        value !== null &&
        Array.isArray(value) === Array.isArray(fallback) &&
        typeof value === typeof fallback
      );
    }),
  );
}

// The notification needs this browser's permission; the files panel is layout state.
const LOCAL_CHAT_PREFS = new Set(["notifyWhenReplyFinishes", "filesPanelCollapsed"]);
// Whether native vibrancy worked and whether this route is in scope belong to this window.
const LOCAL_MATERIAL = new Set(["nativeFailed", "here"]);

type Section = {
  field: Exclude<keyof ClientPreferencesChanges, "sidebarBase">;
  fallback: unknown;
  read: () => unknown;
  apply: (remote: unknown) => void;
  subscribe: (listener: () => void) => () => void;
};

type PersistedStore<S> = {
  getState: () => S;
  getInitialState: () => S;
  setState: (state: Partial<S>) => void;
  subscribe: (listener: () => void) => () => void;
  persist: {
    getOptions: () => { merge?: (persisted: unknown, current: S) => S };
  };
};

function storeSection<S extends object>(
  field: Section["field"],
  store: PersistedStore<S>,
  skip?: ReadonlySet<string>,
): Section {
  const fallback = prefsOf(store.getInitialState(), skip);
  return {
    field,
    fallback,
    read: () => prefsOf(store.getState(), skip),
    apply: (remote) => {
      // merge() resets absent fields to defaults; a field the record lacks keeps this browser's value.
      const saved = {
        ...prefsOf(store.getState(), skip),
        ...prefsFromRemote(remote as Prefs, fallback),
      };
      const next =
        store.persist.getOptions().merge?.(saved, store.getState()) ?? saved;
      store.setState(prefsOf(next, skip) as Partial<S>);
    },
    subscribe: store.subscribe,
  };
}

function sections(): Record<keyof Pending, Section> {
  return {
    scale: {
      field: "interfaceScale",
      fallback: INTERFACE_SCALE_RANGE.default,
      read: () => useInterfaceScaleStore.getState().scale,
      apply: (remote) =>
        useInterfaceScaleStore.getState().setScale(remote as number),
      subscribe: useInterfaceScaleStore.subscribe,
    },
    browser: storeSection("browser", useBrowserPrefsStore),
    chat: storeSection("chat", useChatPreferencesStore, LOCAL_CHAT_PREFS),
    sidebar: storeSection("sidebar", useSidebarOrganizationStore, new Set(["pendingNewChatSection"])),
    plusMenu: storeSection("plusMenu", usePlusMenuPrefsStore),
    material: storeSection("material", useMaterialStore, LOCAL_MATERIAL),
    timeFormat: {
      field: "timeFormat",
      fallback: "system",
      read: getTimeFormat,
      apply: (remote) => setTimeFormat(remote as TimeFormat),
      subscribe: subscribeTimeFormat,
    },
    spellcheck: {
      field: "spellcheck",
      fallback: true,
      read: getSpellCheck,
      apply: (remote) => setSpellCheck(remote as boolean),
      subscribe: subscribeSpellCheck,
    },
  };
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

export function startClientPreferencesSync(): () => void {
  const all = sections();
  let loaded = false;
  let applying = false;
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const touched = readPending();
  let edits = 0;
  // Persist the common ancestor for offline edits. Account transitions clear this key.
  let sidebarBase: Prefs = {};
  try {
    const saved = JSON.parse(localStorage.getItem(SIDEBAR_SYNC_BASE_KEY) ?? "null");
    if (saved && typeof saved === "object" && !Array.isArray(saved)) sidebarBase = saved;
  } catch { /* No usable base: migrate this browser's local sections on its first sync. */ }
  const rememberSidebar = (base: Prefs) => {
    sidebarBase = base;
    try { localStorage.setItem(SIDEBAR_SYNC_BASE_KEY, JSON.stringify(base)); } catch { /* storage full */ }
  };
  const touch = (section: keyof Pending) => {
    touched[section] = true;
    edits++;
    markPending([section], true);
  };

  const account = localStorage.getItem(BROWSER_ACCOUNT_KEY);
  // Another tab's switch clears these keys and a queued request outlives this one's: neither is this account's.
  const switched = () => accountTransitionPending() || !sameBrowserAccount(account);

  let saving: Promise<void> = Promise.resolve();
  const pushAll = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      saving = saving
        .then(async () => {
          const sent = edits;
          const edited = PENDING_SECTIONS.filter((section) => touched[section]);
          if (edited.length === 0 || switched()) return;
          const changes = Object.fromEntries(
            edited.map((section) => [all[section].field, all[section].read()]),
          );
          if (changes.sidebar) changes.sidebarBase = sidebarBase;
          const response = await saveClientPreferences(changes);
          if (changes.sidebar && !cancelled && !switched() && response.sidebar) {
            if (same(all.sidebar.read(), changes.sidebar)) {
              applying = true;
              try { all.sidebar.apply(response.sidebar); } finally { applying = false; }
              last.sidebar = all.sidebar.read();
              rememberSidebar(last.sidebar as Prefs);
            } else {
              // Another local edit is queued. Its delta starts at the state we just sent.
              rememberSidebar(changes.sidebar as Prefs);
            }
          }
          if (edits === sent && !switched()) {
            for (const section of edited) touched[section] = false;
            markPending(edited, false);
          }
        })
        .catch(() => undefined);
    }, PUSH_DEBOUNCE_MS);
  };

  const last = {} as Record<keyof Pending, unknown>;
  const stops = PENDING_SECTIONS.map((section) => {
    last[section] = all[section].read();
    return all[section].subscribe(() => {
      const next = all[section].read();
      if (applying || same(next, last[section])) return;
      if (switched()) return;
      last[section] = next;
      touch(section);
      if (loaded) pushAll();
    });
  });

  void loadClientPreferences()
    .then((remote) => {
      if (cancelled || switched()) return;
      loaded = true;
      // A second browser may already have its own pre-sync sections. Merge those
      // from an empty base instead of replacing them with the first browser's copy.
      if (remote.sidebar && !Object.keys(sidebarBase).length && !same(all.sidebar.read(), all.sidebar.fallback)) {
        touch("sidebar");
      }
      applying = true;
      for (const section of PENDING_SECTIONS) {
        const value = remote[all[section].field];
        if (value != null && !touched[section]) all[section].apply(value);
        last[section] = all[section].read();
      }
      applying = false;
      if (remote.sidebar && !touched.sidebar) rememberSidebar(last.sidebar as Prefs);
      for (const section of PENDING_SECTIONS) {
        if (
          remote[all[section].field] == null &&
          !same(last[section], all[section].fallback)
        )
          touch(section);
      }
      if (PENDING_SECTIONS.some((section) => touched[section])) pushAll();
    })
    .catch(() => {
      loaded = true;
    });

  return () => {
    cancelled = true;
    clearTimeout(timer);
    for (const stop of stops) stop();
  };
}

export function useDeviceSettingsSync(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    const stopBackground = startChatBackgroundSync();
    const stopPreferences = startClientPreferencesSync();
    return () => {
      stopPreferences();
      stopBackground();
    };
  }, [enabled]);
}
