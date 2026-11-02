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

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTimeFormat(): TimeFormat {
  return useSyncExternalStore(subscribe, getTimeFormat);
}

/** Follow changes made in another tab. */
export function watchTimeFormat(win: Window): void {
  win.addEventListener("storage", (event) => {
    if (event.key === TIME_FORMAT_STORAGE_KEY || event.key === null) {
      notify();
    }
  });
}

type HourCycle = "h12" | "h23";

const language = (tag: string) => tag.split("-")[0].toLowerCase();

// The UI locale drops the region ("en" for en-GB), so when it shares the browser's language the
// browser's own clock wins; any other UI language keeps its convention (German stays 24-hour).
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

/** Intl options with the chosen clock, for formatting in `locale` (omit it for the browser's). */
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
  // hour12 outranks hourCycle, so drop it.
  const { hour12: _hour12, ...rest } = options;
  return { ...rest, hourCycle };
}
