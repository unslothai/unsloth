// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import {
  notifyNative,
  primeNativeNotificationPermission,
} from "@/lib/native-notifications";
import { useChatPreferencesStore } from "../stores/chat-preferences-store.ts";

const TITLE = "Reply finished";
const BODY = "Your reply in Unsloth Studio is ready.";

// A reply cut at Max Tokens resumes as a new run (auto-continue), and a queued prompt starts the
// next one; any run starting cancels this wait, so one notification fires once replies settle.
export const REPLY_SETTLE_MS = 3_000;

let sequence = 0;
let pending: ReturnType<typeof setTimeout> | null = null;

function browserNotificationApi(): typeof Notification | null {
  if (typeof window === "undefined" || !("Notification" in window)) {
    return null;
  }
  return window.Notification;
}

function windowIsInBackground(): boolean {
  if (typeof document === "undefined") {
    return false;
  }
  return (
    document.visibilityState !== "visible" ||
    (typeof document.hasFocus === "function" && !document.hasFocus())
  );
}

// Called when a run starts, never per token.
export function cancelReplyFinishedNotification(): void {
  if (pending !== null) {
    clearTimeout(pending);
    pending = null;
  }
}

// Runs once when a reply ends, never per token. Off by default, and silent while the user is
// looking at the window: the reply on screen already says it finished.
export function notifyReplyFinished(): void {
  if (!useChatPreferencesStore.getState().notifyWhenReplyFinishes) {
    return;
  }
  cancelReplyFinishedNotification();
  // A reply that finished in view was seen, even if the user looks away before the timer.
  if (!windowIsInBackground()) {
    return;
  }
  pending = setTimeout(() => {
    pending = null;
    sendReplyFinished();
  }, REPLY_SETTLE_MS);
}

function sendReplyFinished(): void {
  if (
    !useChatPreferencesStore.getState().notifyWhenReplyFinishes ||
    !windowIsInBackground()
  ) {
    return;
  }
  sequence += 1;
  if (isTauri) {
    notifyNative({
      key: `reply-finished:${sequence}`,
      title: TITLE,
      body: BODY,
      requestPermission: false,
    }).catch(() => undefined);
    return;
  }
  const api = browserNotificationApi();
  if (api?.permission !== "granted") {
    return;
  }
  try {
    // No tag: a same-tag replacement alerts only with renotify, so a later reply would arrive
    // silently while an earlier notification is still showing. The settle window already merges
    // bursts.
    const notification = new api(TITLE, { body: BODY });
    notification.onclick = () => {
      window.focus();
      notification.close();
    };
  } catch {
    // Some browsers only allow notifications from a service worker.
  }
}

// Called from the settings switch, a user gesture, so the browser or OS can show its prompt.
// Resolves false when notifications cannot be shown, so the switch does not claim otherwise.
export async function requestReplyFinishedPermission(): Promise<boolean> {
  if (isTauri) {
    return primeNativeNotificationPermission();
  }
  const api = browserNotificationApi();
  if (!api) {
    return false;
  }
  if (api.permission === "granted") {
    return true;
  }
  if (api.permission === "denied") {
    return false;
  }
  try {
    return (await api.requestPermission()) === "granted";
  } catch {
    return false;
  }
}
