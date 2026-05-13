// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

const MAX_BODY_LENGTH = 200;
const MAX_SENT_KEYS = 200;
const HF_TOKEN_PATTERN = /\bhf_[A-Za-z0-9]{20,}\b/g;
const GITHUB_TOKEN_PATTERN = /\bghp_[A-Za-z0-9_]{20,}\b/g;
const GITHUB_PAT_PATTERN = /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g;
const OPENAI_KEY_PATTERN = /\bsk-[A-Za-z0-9_-]{20,}\b/g;
const UNIX_HOME_PATH_PATTERN =
  /(^|[\s"'(=])(?:\/Users|\/home)\/[^\s"'<>),;]+/gi;
const UNIX_ABSOLUTE_PATH_PATTERN = /(^|[\s"'(=])\/[^\s"'<>),;]+/g;
const TILDE_PATH_PATTERN = /(^|[\s"'(=])~[\\/][^\s"'<>),;]+/g;
const RELATIVE_PATH_PATTERN = /(^|[\s"'(=])\.{1,2}[\\/][^\s"'<>),;]+/g;
const WINDOWS_DRIVE_PATH_PATTERN = /(^|[\s"'(=])[A-Za-z]:[\\/][^\s"'<>),;]+/g;
const WINDOWS_UNC_PATH_PATTERN = /(^|[\s"'(=])\\\\[^\s"'<>),;]+/g;
const WHITESPACE_PATTERN = /\s+/g;
const LOCAL_PATH_PATTERN = /^(?:\/|~[\\/]|\.{1,2}[\\/]|[A-Za-z]:[\\/]|\\\\)/;
const PATH_SEPARATOR_PATTERN = /[\\/]/;
const SENSITIVE_TOKEN_PATTERN =
  /\bhf_[A-Za-z0-9]{20,}\b|\bghp_[A-Za-z0-9_]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b/;

type PermissionState = "unknown" | "granted" | "denied";

let permissionState: PermissionState = "unknown";
let permissionPrimeInFlight: Promise<void> | null = null;
const inFlightKeys = new Set<string>();
const sentKeys = new Set<string>();

export interface NativeNotificationOptions {
  key: string;
  title: string;
  body?: string;
  requestPermission?: boolean;
}

function hasGrantedPermission(): boolean {
  return permissionState === "granted";
}

function getBrowserNotificationApi(): typeof Notification | null {
  if (typeof window === "undefined" || !("Notification" in window)) {
    return null;
  }

  return window.Notification;
}

function rememberSentKey(key: string): void {
  sentKeys.add(key);

  if (sentKeys.size <= MAX_SENT_KEYS) {
    return;
  }

  const oldestKey = sentKeys.values().next().value;
  if (oldestKey) {
    sentKeys.delete(oldestKey);
  }
}

async function checkNativePermission(allowRequest: boolean): Promise<boolean> {
  const notification = getBrowserNotificationApi();
  if (notification?.permission === "granted") {
    permissionState = "granted";
    return true;
  }

  if (notification?.permission === "denied") {
    permissionState = "denied";
    return false;
  }

  if (allowRequest && notification?.permission === "default") {
    const permission = await notification.requestPermission();
    if (permission === "granted") {
      permissionState = "granted";
      return true;
    }
    if (permission === "denied") {
      permissionState = "denied";
    }
    return false;
  }

  try {
    const { isPermissionGranted } = await import(
      "@tauri-apps/plugin-notification"
    );
    const granted = await isPermissionGranted();

    if (granted) {
      permissionState = "granted";
    }

    return granted;
  } catch {
    return false;
  }
}

async function ensurePermission(allowRequest: boolean): Promise<boolean> {
  if (!isTauri) {
    return false;
  }
  if (permissionState !== "unknown") {
    return hasGrantedPermission();
  }

  if (permissionPrimeInFlight) {
    await permissionPrimeInFlight;
    if (permissionState !== "unknown" || !allowRequest) {
      return hasGrantedPermission();
    }
  }

  return checkNativePermission(allowRequest);
}

export async function primeNativeNotificationPermission(): Promise<void> {
  if (!isTauri || permissionState !== "unknown") {
    return;
  }

  if (!permissionPrimeInFlight) {
    permissionPrimeInFlight = checkNativePermission(true)
      .then(() => undefined)
      .finally(() => {
        permissionPrimeInFlight = null;
      });
  }

  await permissionPrimeInFlight;
}

export async function notifyNative(
  options: NativeNotificationOptions,
): Promise<void> {
  if (!isTauri) {
    return;
  }
  if (sentKeys.has(options.key) || inFlightKeys.has(options.key)) {
    return;
  }
  inFlightKeys.add(options.key);

  try {
    const granted = await ensurePermission(options.requestPermission !== false);
    if (!granted) {
      return;
    }

    const body = options.body
      ? sanitizeNotificationBody(options.body, "")
      : undefined;
    const { sendNotification } = await import(
      "@tauri-apps/plugin-notification"
    );
    sendNotification({ title: options.title, body });
    rememberSentKey(options.key);
  } catch {
    // Native notifications are non-critical UI side effects.
  } finally {
    inFlightKeys.delete(options.key);
  }
}

export function sanitizeNotificationBody(
  input: string | null | undefined,
  fallback: string,
): string {
  const fallbackText = fallback.trim() || "Notification update.";
  let text = typeof input === "string" ? input.trim() : "";

  if (!text) {
    text = fallbackText;
  }

  text = text
    .replace(HF_TOKEN_PATTERN, "hf_[redacted]")
    .replace(GITHUB_TOKEN_PATTERN, "ghp_[redacted]")
    .replace(GITHUB_PAT_PATTERN, "github_pat_[redacted]")
    .replace(OPENAI_KEY_PATTERN, "sk-[redacted]")
    .replace(UNIX_HOME_PATH_PATTERN, "$1[path]")
    .replace(UNIX_ABSOLUTE_PATH_PATTERN, "$1[path]")
    .replace(TILDE_PATH_PATTERN, "$1[path]")
    .replace(RELATIVE_PATH_PATTERN, "$1[path]")
    .replace(WINDOWS_DRIVE_PATH_PATTERN, "$1[path]")
    .replace(WINDOWS_UNC_PATH_PATTERN, "$1[path]")
    .replace(WHITESPACE_PATTERN, " ")
    .trim();

  if (!text) {
    text = fallbackText;
  }

  if (text.length > MAX_BODY_LENGTH) {
    return `${text.slice(0, MAX_BODY_LENGTH - 1).trimEnd()}…`;
  }

  return text;
}

export function safeNotificationLabel(
  input: string | null | undefined,
  fallback: string,
): string {
  const fallbackText = fallback.trim() || "The item";
  const raw = typeof input === "string" ? input.trim() : "";
  if (!raw) {
    return fallbackText;
  }

  const collapsed = raw.replace(WHITESPACE_PATTERN, " ");
  const isLocalPath = LOCAL_PATH_PATTERN.test(collapsed);

  if (isLocalPath) {
    const basename =
      collapsed.split(PATH_SEPARATOR_PATTERN).filter(Boolean).at(-1) ?? "";
    if (
      basename &&
      basename !== "." &&
      basename !== ".." &&
      !SENSITIVE_TOKEN_PATTERN.test(basename)
    ) {
      return sanitizeNotificationBody(basename, fallbackText);
    }
    return fallbackText;
  }

  return sanitizeNotificationBody(collapsed, fallbackText);
}
