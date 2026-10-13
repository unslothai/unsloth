// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAccountOwner } from "@/features/auth";
import { useEffect, useState } from "react";

/** Folder new model downloads go to. Absent means the Models folder. */
export const DOWNLOAD_DIR_STORAGE_KEY = "unsloth.studio.downloadDir";
const CHANGE_EVENT = "unsloth:download-dir-change";

export function getDownloadDir(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(DOWNLOAD_DIR_STORAGE_KEY);
    return raw?.trim() ? raw : null;
  } catch {
    return null;
  }
}

/** What a model download sends: the folder pinned on the request, else the current choice. Only
 * the owner may pick a folder, and this browser may hold an owner's choice while another account
 * is signed in. */
export function downloadDirForRequest(
  pinned?: string | null,
): string | undefined {
  if (!isAccountOwner()) {
    return undefined;
  }
  return (pinned === undefined ? getDownloadDir() : pinned) ?? undefined;
}

export function setDownloadDir(path: string | null): void {
  try {
    if (path?.trim()) {
      window.localStorage.setItem(DOWNLOAD_DIR_STORAGE_KEY, path);
    } else {
      window.localStorage.removeItem(DOWNLOAD_DIR_STORAGE_KEY);
    }
  } catch {
    return;
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

export function useDownloadDir(): string | null {
  const [dir, setDir] = useState<string | null>(getDownloadDir);
  useEffect(() => {
    const sync = () => setDir(getDownloadDir());
    const handleStorage = (event: StorageEvent) => {
      if (event.key === null || event.key === DOWNLOAD_DIR_STORAGE_KEY) {
        sync();
      }
    };
    window.addEventListener(CHANGE_EVENT, sync);
    window.addEventListener("storage", handleStorage);
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync);
      window.removeEventListener("storage", handleStorage);
    };
  }, []);
  return dir;
}
