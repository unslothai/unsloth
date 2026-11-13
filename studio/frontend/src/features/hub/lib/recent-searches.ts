// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

const STORAGE_KEY = "unsloth.hub.recentSearches";
const CHANGE_EVENT = "unsloth:hub-recent-searches-change";

export const MAX_RECENT_SEARCHES = 8;
const MIN_QUERY_LENGTH = 2;

function readStored(): string[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((value): value is string => typeof value === "string")
      .slice(0, MAX_RECENT_SEARCHES);
  } catch {
    return [];
  }
}

function writeStored(list: string[]): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    return;
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

export function recordRecentSearch(query: string): void {
  const trimmed = query.trim();
  if (trimmed.length < MIN_QUERY_LENGTH) {
    return;
  }
  const withoutDuplicate = readStored().filter(
    (item) => item.toLowerCase() !== trimmed.toLowerCase(),
  );
  writeStored([trimmed, ...withoutDuplicate].slice(0, MAX_RECENT_SEARCHES));
}

export function removeRecentSearch(query: string): void {
  const existing = readStored();
  const next = existing.filter((item) => item !== query);
  if (next.length !== existing.length) {
    writeStored(next);
  }
}

export function clearRecentSearches(): void {
  if (readStored().length === 0) {
    return;
  }
  writeStored([]);
}

export function useRecentSearches(): string[] {
  const [list, setList] = useState<string[]>(readStored);

  useEffect(() => {
    const handleLocal = () => setList(readStored());
    const handleStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) {
        return;
      }
      if (event.key !== null && event.key !== STORAGE_KEY) {
        return;
      }
      setList(readStored());
    };
    window.addEventListener(CHANGE_EVENT, handleLocal);
    window.addEventListener("storage", handleStorage);
    return () => {
      window.removeEventListener(CHANGE_EVENT, handleLocal);
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  return list;
}
