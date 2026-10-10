// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, useSyncExternalStore } from "react";

// How often each update notification may appear (Settings -> General -> Notifications).
export const NOTIFICATION_FREQUENCIES = [
  "always",
  "daily",
  "weekly",
  "off",
] as const;
export type NotificationFrequency = (typeof NOTIFICATION_FREQUENCIES)[number];

export const NOTIFICATION_CHANNELS = [
  "llama",
  "whisper",
  "audio",
  "unsloth",
] as const;
export type NotificationChannel = (typeof NOTIFICATION_CHANNELS)[number];

const DAY_MS = 24 * 60 * 60 * 1000;
const PERIOD_MS: Partial<Record<NotificationFrequency, number>> = {
  daily: DAY_MS,
  weekly: 7 * DAY_MS,
};
// setTimeout overflows past 2^31 - 1 ms.
const MAX_TIMEOUT_MS = 2_147_483_647;

// These were on/off switches; only "false" was ever stored, and it means off.
const LEGACY_SWITCH_KEYS: Partial<Record<NotificationChannel, string>> = {
  llama: "unsloth_show_llama_update_banner",
  whisper: "unsloth_show_whisper_update_banner",
  audio: "unsloth_show_audio_cpp_update_banner",
};

const frequencyKey = (channel: NotificationChannel) =>
  `unsloth_${channel}_notification_frequency`;
const lastShownKey = (channel: NotificationChannel) =>
  `unsloth_${channel}_notification_last_shown`;

/** Every key this module owns, for "Reset all local preferences". */
export const NOTIFICATION_PREF_KEYS: readonly string[] = [
  ...NOTIFICATION_CHANNELS.flatMap((c) => [frequencyKey(c), lastShownKey(c)]),
  ...Object.values(LEGACY_SWITCH_KEYS),
];

const OWN_KEYS = new Set(NOTIFICATION_PREF_KEYS);
const listeners = new Set<() => void>();

function isFrequency(value: unknown): value is NotificationFrequency {
  return (NOTIFICATION_FREQUENCIES as readonly unknown[]).includes(value);
}

export function getNotificationFrequency(
  channel: NotificationChannel,
): NotificationFrequency {
  try {
    const stored = localStorage.getItem(frequencyKey(channel));
    if (isFrequency(stored)) return stored;
    const legacy = LEGACY_SWITCH_KEYS[channel];
    return legacy && localStorage.getItem(legacy) === "false"
      ? "off"
      : "always";
  } catch {
    return "always";
  }
}

export function setNotificationFrequency(
  channel: NotificationChannel,
  frequency: NotificationFrequency,
): void {
  try {
    localStorage.setItem(frequencyKey(channel), frequency);
    const legacy = LEGACY_SWITCH_KEYS[channel];
    if (legacy) localStorage.removeItem(legacy);
  } catch {
    // storage unavailable
  }
  for (const listener of listeners) listener();
}

function getLastShown(channel: NotificationChannel): number | null {
  try {
    const value = Number(localStorage.getItem(lastShownKey(channel)));
    return Number.isFinite(value) && value > 0 ? value : null;
  } catch {
    return null;
  }
}

export function markNotificationShown(
  channel: NotificationChannel,
  now: number = Date.now(),
): void {
  try {
    localStorage.setItem(lastShownKey(channel), String(now));
  } catch {
    // storage unavailable
  }
  for (const listener of listeners) listener();
}

/** Whether a notification last shown at `lastShown` may appear again at `now`. */
export function notificationDue(
  frequency: NotificationFrequency,
  lastShown: number | null,
  now: number,
): boolean {
  if (frequency === "off") return false;
  const period = PERIOD_MS[frequency];
  // A clock moved backwards reads as due rather than silencing it for good.
  return (
    period == null ||
    lastShown == null ||
    now - lastShown >= period ||
    now < lastShown
  );
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Sync changes made in another tab.
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || OWN_KEYS.has(event.key)) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function useNotificationFrequency(
  channel: NotificationChannel,
): NotificationFrequency {
  return useSyncExternalStore(subscribe, () =>
    getNotificationFrequency(channel),
  );
}

/** Whether `channel` may show now; re-renders when its quiet period ends. */
export function useNotificationDue(channel: NotificationChannel): boolean {
  const frequency = useNotificationFrequency(channel);
  const lastShown = useSyncExternalStore(subscribe, () =>
    getLastShown(channel),
  );
  const [, setTick] = useState(0);
  const due = notificationDue(frequency, lastShown, Date.now());
  useEffect(() => {
    const period = PERIOD_MS[frequency];
    if (due || period == null || lastShown == null) return;
    const wait = Math.min(
      Math.max(lastShown + period - Date.now(), 0) + 1000,
      MAX_TIMEOUT_MS,
    );
    const timer = setTimeout(() => setTick((tick) => tick + 1), wait);
    return () => clearTimeout(timer);
  }, [due, frequency, lastShown]);
  return due;
}

/**
 * Whether a notification that `wanted` to show may show. Opening it records the time,
 * and it stays open until `wanted` drops (dismiss, snooze, update) or the channel is off.
 */
export function useNotificationGate(
  channel: NotificationChannel,
  wanted: boolean,
): boolean {
  const due = useNotificationDue(channel);
  const frequency = useNotificationFrequency(channel);
  const [heldFor, setHeldFor] = useState<NotificationChannel | null>(null);
  const held = heldFor === channel;
  const open = wanted && frequency !== "off" && (held || due);
  useEffect(() => {
    if (open && !held) {
      markNotificationShown(channel);
      setHeldFor(channel);
    } else if (!wanted && heldFor !== null) {
      setHeldFor(null);
    }
  }, [open, held, wanted, heldFor, channel]);
  return open;
}
