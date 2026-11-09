// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

export const TIME_FORMAT_STORAGE_KEY = "unsloth_time_format";

export type TimeFormat = "system" | "12h" | "24h";

const listeners = new Set<() => void>();

function isTimeFormat(value: unknown): value is TimeFormat {
  return value === "system" || value === "12h" || value === "24h";
}

export function getTimeFormat(): TimeFormat {
  try {
    const stored = localStorage.getItem(TIME_FORMAT_STORAGE_KEY);
    return isTimeFormat(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

export function setTimeFormat(next: TimeFormat): void {
  try {
    if (next === "system") {
      localStorage.removeItem(TIME_FORMAT_STORAGE_KEY);
    } else {
      localStorage.setItem(TIME_FORMAT_STORAGE_KEY, next);
    }
  } catch {
    // storage unavailable
  }
  notify();
}

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function subscribeTimeFormat(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTimeFormat(): TimeFormat {
  return useSyncExternalStore(subscribeTimeFormat, getTimeFormat, getTimeFormat);
}

export function watchTimeFormat(win: Window): void {
  win.addEventListener("storage", (event) => {
    if (event.key === TIME_FORMAT_STORAGE_KEY || event.key === null) {
      notify();
    }
  });
}

type HourCycle = "h12" | "h23";

const language = (tag: string) => tag.split("-")[0].toLowerCase();

// The UI locale has no region (en-GB is "en"), so a shared language takes the browser's clock.
function systemHourCycle(locale: string | undefined): HourCycle | undefined {
  if (!locale) {
    return undefined;
  }
  try {
    const browser = new Intl.DateTimeFormat(undefined, {
      hour: "numeric",
    }).resolvedOptions();
    if (language(browser.locale) !== language(locale)) {
      return undefined;
    }
    return browser.hourCycle === "h11" || browser.hourCycle === "h12"
      ? "h12"
      : "h23";
  } catch {
    return undefined;
  }
}

export function withTimeFormat(
  options: Intl.DateTimeFormatOptions = {},
  locale?: string,
): Intl.DateTimeFormatOptions {
  const format = getTimeFormat();
  const hourCycle =
    format === "12h"
      ? "h12"
      : format === "24h"
        ? "h23"
        : systemHourCycle(locale);
  if (!hourCycle) {
    return options;
  }
  // hour12 would outrank hourCycle.
  const { hour12: _hour12, ...rest } = options;
  return { ...rest, hourCycle };
}
