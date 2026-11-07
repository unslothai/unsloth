// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBrowserPrefsStore } from "@/features/browser/prefs-store";
import { startChatBackgroundSync } from "@/features/chat-background/sync";
import { useEffect } from "react";
import { loadClientPreferences, saveClientPreferences } from "../api/device-sync";
import {
  INTERFACE_SCALE_RANGE,
  useInterfaceScaleStore,
} from "../stores/interface-scale-store";
import {
  type Pending,
  anyPending,
  readPending,
  writePending,
} from "../lib/client-prefs-pending";
import { SYNCED_SECTIONS } from "../lib/synced-preferences";

const PUSH_DEBOUNCE_MS = 800;

type BrowserPrefs = Record<string, unknown>;

function prefsOf(state: object): BrowserPrefs {
  return Object.fromEntries(
    Object.entries(state).filter(([, value]) => typeof value !== "function"),
  );
}

const browserPrefs = () => prefsOf(useBrowserPrefsStore.getState());
const DEFAULT_BROWSER_PREFS = prefsOf(useBrowserPrefsStore.getInitialState());

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

export function useDeviceSettingsSync(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    const stopBackground = startChatBackgroundSync();
    let loaded = false;
    let applying = false;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const touched = readPending();
    let edits = 0;
    const touch = (section: keyof Pending) => {
      touched[section] = true;
      edits++;
      writePending(touched);
    };

    let saving: Promise<void> = Promise.resolve();
    const pushAll = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        saving = saving
          .then(async () => {
            const sent = edits;
            if (!anyPending(touched)) return;
            const sections = Object.fromEntries(
              SYNCED_SECTIONS.filter((section) => touched[section.pending]).map(
                (section) => [section.field, section.read()],
              ),
            );
            await saveClientPreferences({
              interfaceScale: touched.scale
                ? useInterfaceScaleStore.getState().scale
                : undefined,
              browser: touched.browser ? browserPrefs() : undefined,
              ...sections,
            });
            if (edits === sent) {
              for (const section of Object.keys(touched) as (keyof Pending)[])
                touched[section] = false;
              writePending(touched);
            }
          })
          .catch(() => undefined);
      }, PUSH_DEBOUNCE_MS);
    };

    const stopScale = useInterfaceScaleStore.subscribe((state, prev) => {
      if (applying || state.scale === prev.scale) return;
      touch("scale");
      if (loaded) pushAll();
    });
    let lastBrowser = browserPrefs();
    const stopBrowser = useBrowserPrefsStore.subscribe(() => {
      const next = browserPrefs();
      if (applying || sameBrowserPrefs(next, lastBrowser)) return;
      lastBrowser = next;
      touch("browser");
      if (loaded) pushAll();
    });
    const stopSections = SYNCED_SECTIONS.map((section) =>
      section.subscribe(() => {
        if (applying) return;
        touch(section.pending);
        if (loaded) pushAll();
      }),
    );

    void loadClientPreferences()
      .then((remote) => {
        if (cancelled) return;
        loaded = true;
        applying = true;
        if (remote.interfaceScale !== null && !touched.scale)
          useInterfaceScaleStore.getState().setScale(remote.interfaceScale);
        if (remote.browser && !touched.browser)
          useBrowserPrefsStore.setState(browserPrefsFromRemote(remote.browser));
        lastBrowser = browserPrefs();
        for (const section of SYNCED_SECTIONS) {
          const value = remote[section.field];
          if (value === null || value === undefined || touched[section.pending]) continue;
          try {
            section.apply(value);
          } catch {
            // A record this build cannot read leaves the local value in place.
          }
        }
        applying = false;
        const scale = useInterfaceScaleStore.getState().scale;
        const missingScale =
          remote.interfaceScale === null &&
          scale !== INTERFACE_SCALE_RANGE.default;
        const missingBrowser =
          !remote.browser && !sameBrowserPrefs(lastBrowser, DEFAULT_BROWSER_PREFS);
        if (missingScale) touch("scale");
        if (missingBrowser) touch("browser");
        // A value set here before the account had one is uploaded, never a default over a saved one.
        for (const section of SYNCED_SECTIONS) {
          const value = remote[section.field];
          if ((value === null || value === undefined) && !section.isDefault())
            touch(section.pending);
        }
        if (anyPending(touched)) pushAll();
      })
      .catch(() => {
        loaded = true;
      });

    return () => {
      cancelled = true;
      clearTimeout(timer);
      stopScale();
      stopBrowser();
      for (const stop of stopSections) stop();
      stopBackground();
    };
  }, [enabled]);
}
