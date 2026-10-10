// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
const {
  NOTIFICATION_PREF_KEYS,
  getNotificationFrequency,
  markNotificationShown,
  notificationDue,
  setNotificationFrequency,
} = await import("../src/hooks/use-notification-frequency.ts");

const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;

test("each frequency decides when a notification may show again", () => {
  const shown = 1_000_000_000_000;
  assert.equal(notificationDue("always", shown, shown + 1), true);
  assert.equal(notificationDue("off", null, shown), false);
  for (const [frequency, period] of [
    ["daily", DAY],
    ["weekly", 7 * DAY],
  ] as const) {
    assert.equal(
      notificationDue(frequency, null, shown),
      true,
      `${frequency} never shown`,
    );
    assert.equal(
      notificationDue(frequency, shown, shown + period - 1),
      false,
      frequency,
    );
    assert.equal(
      notificationDue(frequency, shown, shown + period),
      true,
      frequency,
    );
    // A clock set back must not silence it until the clock catches up.
    assert.equal(
      notificationDue(frequency, shown, shown - HOUR),
      true,
      frequency,
    );
  }
});

test("the old on/off switches carry over", () => {
  store.clear();
  assert.equal(getNotificationFrequency("llama"), "always");
  store.set("unsloth_show_llama_update_banner", "false");
  assert.equal(getNotificationFrequency("llama"), "off");
  // Playwright layout tests still write "true", which stays on.
  store.set("unsloth_show_whisper_update_banner", "true");
  assert.equal(getNotificationFrequency("whisper"), "always");
  assert.equal(getNotificationFrequency("unsloth"), "always");
});

test("choosing a frequency replaces the old switch", () => {
  store.clear();
  store.set("unsloth_show_audio_cpp_update_banner", "false");
  setNotificationFrequency("audio", "weekly");
  assert.equal(getNotificationFrequency("audio"), "weekly");
  assert.equal(store.has("unsloth_show_audio_cpp_update_banner"), false);
  store.set("unsloth_audio_notification_frequency", "hourly");
  assert.equal(
    getNotificationFrequency("audio"),
    "always",
    "an unknown value falls back",
  );
});

test("showing records the time and reset clears every key", () => {
  store.clear();
  setNotificationFrequency("unsloth", "daily");
  markNotificationShown("unsloth", 42);
  assert.equal(store.get("unsloth_unsloth_notification_last_shown"), "42");
  for (const key of store.keys()) {
    assert.ok(NOTIFICATION_PREF_KEYS.includes(key), key);
  }
  for (const legacy of [
    "unsloth_show_llama_update_banner",
    "unsloth_show_whisper_update_banner",
    "unsloth_show_audio_cpp_update_banner",
  ]) {
    assert.ok(NOTIFICATION_PREF_KEYS.includes(legacy), legacy);
  }
});
