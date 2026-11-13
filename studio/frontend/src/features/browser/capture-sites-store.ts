// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "ask" is the system's own prompt and "block" refuses: the app never grants a page a device, so there is no "allow".

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { accountDatabaseName } from "@/lib/account-transition";
import { syncAcrossWindows } from "@/lib/sync-across-windows";

export type CaptureRule = "ask" | "block";
export type CaptureKind = "camera" | "microphone";
export type CaptureSiteRule = Record<CaptureKind, CaptureRule>;
export type CaptureRules = CaptureSiteRule & { sites: Record<string, CaptureSiteRule> };

// Both mirror browser_webview.rs, which refuses the whole set past either.
export const MAX_CAPTURE_SITES = 500;
const MAX_ORIGIN_LENGTH = 300;

/** An origin, not a host pattern: the rule is for one scheme, host and port. */
export function normalizeCaptureOrigin(input: string): string | null {
  const value = input.trim();
  if (!value || /\s/.test(value)) return null;
  try {
    const url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(value) ? value : `https://${value}`);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    const host = url.hostname.replace(/\.$/, "");
    if (!host) return null;
    url.hostname = host;
    return url.origin.length <= MAX_ORIGIN_LENGTH ? url.origin : null;
  } catch {
    return null;
  }
}

interface CaptureSitesState extends CaptureRules {
  setDefault: (kind: CaptureKind, rule: CaptureRule) => void;
  setSite: (origin: string, rule: CaptureSiteRule | null) => boolean;
}

export const useCaptureSitesStore = create<CaptureSitesState>()(
  persist(
    (set, get) => ({
      camera: "ask",
      microphone: "ask",
      sites: {},
      setDefault: (kind, rule) => set(kind === "camera" ? { camera: rule } : { microphone: rule }),
      setSite: (origin, rule) => {
        const sites = { ...get().sites };
        if (!rule) delete sites[origin];
        else if (origin in sites || Object.keys(sites).length < MAX_CAPTURE_SITES) sites[origin] = rule;
        else return false;
        set({ sites });
        return true;
      },
    }),
    {
      name: accountDatabaseName("unsloth_browser_capture_sites"),
      version: 1,
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
syncAcrossWindows(useCaptureSitesStore);

const ruleOf = (value: unknown): CaptureRule => (value === "block" ? "block" : "ask");

/** Cleaned, not trusted: one bad stored entry would make the app refuse the whole set. */
export function captureRules(state: CaptureRules): CaptureRules {
  const sites: Record<string, CaptureSiteRule> = {};
  for (const [site, rule] of Object.entries(state.sites ?? {}).slice(0, MAX_CAPTURE_SITES)) {
    const origin = normalizeCaptureOrigin(site);
    if (origin) sites[origin] = { camera: ruleOf(rule?.camera), microphone: ruleOf(rule?.microphone) };
  }
  return { camera: ruleOf(state.camera), microphone: ruleOf(state.microphone), sites };
}

export function watchCaptureRules(send: (rules: CaptureRules) => Promise<unknown>): () => void {
  let sent = "";
  const push = () => {
    const rules = captureRules(useCaptureSitesStore.getState());
    const key = JSON.stringify(rules);
    if (key === sent) return;
    sent = key;
    // Forgotten on failure, so the next change sends again.
    void send(rules).catch(() => {
      sent = "";
    });
  };
  push();
  return useCaptureSitesStore.subscribe(push);
}
