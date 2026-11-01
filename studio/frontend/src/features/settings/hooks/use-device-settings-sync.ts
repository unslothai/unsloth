// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBrowserPrefsStore } from "@/features/browser/prefs-store";
import { startChatBackgroundSync } from "@/features/chat-background/sync";
import { useEffect } from "react";
import {
  type ClientPreferences,
  loadClientPreferences,
  saveClientPreferences,
} from "../api/device-sync";
import {
  INTERFACE_SCALE_RANGE,
  useInterfaceScaleStore,
} from "../stores/interface-scale-store";

const PUSH_DEBOUNCE_MS = 800;

type BrowserPrefs = Record<string, unknown>;

function prefsOf(state: object): BrowserPrefs {
  return Object.fromEntries(
    Object.entries(state).filter(([, value]) => typeof value !== "function"),
  );
}

const browserPrefs = () => prefsOf(useBrowserPrefsStore.getState());
// The store's own defaults, from before it read this browser's saved copy.
const DEFAULT_BROWSER_PREFS = prefsOf(useBrowserPrefsStore.getInitialState());

/** Keeps only known fields whose type matches the default, so a stale record cannot break the store. */
export function browserPrefsFromRemote(remote: BrowserPrefs): BrowserPrefs {
  return Object.fromEntries(
    Object.entries(remote).filter(([key, value]) => {
      const fallback = DEFAULT_BROWSER_PREFS[key];
      return (
        key in DEFAULT_BROWSER_PREFS &&
        value !== null &&
        Array.isArray(value) === Array.isArray(fallback) &&
        typeof value === typeof fallback
      );
    }),
  );
}

function sameBrowserPrefs(a: BrowserPrefs, b: BrowserPrefs): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Interface scale, Browser tab preferences and the chat background follow the account, not the origin. */
export function useDeviceSettingsSync(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    const stopBackground = startChatBackgroundSync();
    let loaded = false;
    let applying = false;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const touched = { scale: false, browser: false };

    const push = (
      data: Partial<Pick<ClientPreferences, "interfaceScale" | "browser">>,
    ) => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        void saveClientPreferences(data).catch(() => undefined);
      }, PUSH_DEBOUNCE_MS);
    };
    const pushAll = () =>
      push({
        interfaceScale: useInterfaceScaleStore.getState().scale,
        browser: browserPrefs(),
      });

    const stopScale = useInterfaceScaleStore.subscribe((state, prev) => {
      if (applying || state.scale === prev.scale) return;
      touched.scale = true;
      if (loaded) pushAll();
    });
    let lastBrowser = browserPrefs();
    const stopBrowser = useBrowserPrefsStore.subscribe(() => {
      const next = browserPrefs();
      if (applying || sameBrowserPrefs(next, lastBrowser)) return;
      lastBrowser = next;
      touched.browser = true;
      if (loaded) pushAll();
    });

    void loadClientPreferences()
      .then((remote) => {
        if (cancelled) return;
        loaded = true;
        applying = true;
        // A change made while the request was in flight is newer than the account's copy.
        if (remote.interfaceScale !== null && !touched.scale)
          useInterfaceScaleStore.getState().setScale(remote.interfaceScale);
        if (remote.browser && !touched.browser)
          useBrowserPrefsStore.setState(browserPrefsFromRemote(remote.browser));
        lastBrowser = browserPrefs();
        applying = false;
        const scale = useInterfaceScaleStore.getState().scale;
        const missingScale =
          remote.interfaceScale === null &&
          scale !== INTERFACE_SCALE_RANGE.default;
        const missingBrowser =
          !remote.browser && !sameBrowserPrefs(lastBrowser, DEFAULT_BROWSER_PREFS);
        if (touched.scale || touched.browser || missingScale || missingBrowser)
          pushAll();
      })
      .catch(() => {
        loaded = true;
      });

    return () => {
      cancelled = true;
      clearTimeout(timer);
      stopScale();
      stopBrowser();
      stopBackground();
    };
  }, [enabled]);
}
