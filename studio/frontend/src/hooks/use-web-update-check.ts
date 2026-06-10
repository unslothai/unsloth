// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";
import { apiUrl, isTauri } from "@/lib/api-base";
import { useCallback, useEffect, useState } from "react";

const WEB_UPDATE_CHECK_DELAY_MS = 5000;
const DISMISS_PREFIX = "unsloth_web_update_dismissed";
const CAN_SHOW_KEY = "can_show_web_notification";
const UPDATE_AVAILABLE_KEY = "update_available";
const INSTALL_SOURCE_KEY = "install_source";
const LATEST_VERSION_KEY = "latest_version";
const CURRENT_VERSION_KEY = "current_version";
const CHECKED_AT_KEY = "checked_at";

type ApiObject = Record<string, unknown>;

export type WebUpdateInstallSource =
  | "pypi"
  | "editable"
  | "local_path"
  | "vcs"
  | "local_repo"
  | "unknown";

export interface WebUpdateStatus {
  currentVersion: string;
  latestVersion: string;
  installSource: "pypi";
  checkedAt: string;
}

interface UseWebUpdateCheckOptions {
  enabled?: boolean;
  delayMs?: number;
}

function stringField(value: ApiObject, key: string): string | null {
  const field = value[key];
  return typeof field === "string" ? field : null;
}

function toDisplayableUpdateStatus(value: unknown): WebUpdateStatus | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const status = value as ApiObject;
  const latestVersion = stringField(status, LATEST_VERSION_KEY);
  const currentVersion = stringField(status, CURRENT_VERSION_KEY);
  const checkedAt = stringField(status, CHECKED_AT_KEY);
  if (
    status[CAN_SHOW_KEY] !== true ||
    status[UPDATE_AVAILABLE_KEY] !== true ||
    status[INSTALL_SOURCE_KEY] !== "pypi" ||
    !latestVersion ||
    !currentVersion ||
    !checkedAt
  ) {
    return null;
  }

  return {
    currentVersion,
    latestVersion,
    installSource: "pypi",
    checkedAt,
  };
}

function dismissalKey(status: WebUpdateStatus): string {
  return `${DISMISS_PREFIX}:${status.installSource}:${status.latestVersion}`;
}

function isDismissed(status: WebUpdateStatus): boolean {
  if (typeof window === "undefined") {
    return true;
  }
  try {
    return window.localStorage.getItem(dismissalKey(status)) !== null;
  } catch {
    return false;
  }
}

function markDismissed(status: WebUpdateStatus): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(dismissalKey(status), String(Date.now()));
  } catch {
    // Ignore storage failures; the banner can still be dismissed in-memory.
  }
}

async function fetchDisplayableUpdateStatus(): Promise<WebUpdateStatus | null> {
  const token = getAuthToken();
  if (!token) {
    return null;
  }

  const headers = new Headers();
  headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(apiUrl("/api/studio/update-status"), { headers });
  if (!res.ok) {
    return null;
  }

  return toDisplayableUpdateStatus(await res.json());
}

export function useWebUpdateCheck({
  enabled = true,
  delayMs = WEB_UPDATE_CHECK_DELAY_MS,
}: UseWebUpdateCheckOptions = {}) {
  const [status, setStatus] = useState<WebUpdateStatus | null>(null);

  useEffect(() => {
    if (isTauri || !enabled || !getAuthToken()) {
      const clearTimer = window.setTimeout(() => setStatus(null), 0);
      return () => window.clearTimeout(clearTimer);
    }

    let canceled = false;
    const timer = window.setTimeout(() => {
      fetchDisplayableUpdateStatus()
        .then((nextStatus) => {
          if (canceled) {
            return;
          }
          setStatus(nextStatus && !isDismissed(nextStatus) ? nextStatus : null);
        })
        .catch(() => {
          if (!canceled) {
            setStatus(null);
          }
        });
    }, delayMs);

    return () => {
      canceled = true;
      window.clearTimeout(timer);
    };
  }, [delayMs, enabled]);

  const dismiss = useCallback(() => {
    setStatus((current) => {
      if (current) {
        markDismissed(current);
      }
      return null;
    });
  }, []);

  return { status: enabled && !isTauri ? status : null, dismiss };
}
