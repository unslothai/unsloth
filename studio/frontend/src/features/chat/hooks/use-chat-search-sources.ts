// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import {
  KIND_ICONS,
  type LibraryItem,
  fileKind,
  isModelItem,
} from "@/features/library";
import type { IconSvgElement } from "@hugeicons/react";
import { useEffect, useState } from "react";

export interface LibrarySearchEntry {
  item: LibraryItem;
  icon: IconSvgElement;
}

export interface ChatSearchSources {
  ready: boolean;
  files: LibrarySearchEntry[];
  fineTunes: LibrarySearchEntry[];
}

const EMPTY: ChatSearchSources = { ready: false, files: [], fineTunes: [] };

export function useChatSearchSources(open: boolean): ChatSearchSources {
  const [sources, setSources] = useState<ChatSearchSources>(EMPTY);

  useEffect(() => {
    const clear = () => setSources(EMPTY);
    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, clear);
    return () => window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, clear);
  }, []);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    let unsubscribe: (() => void) | undefined;

    import("@/features/library/store")
      .then(({ useLibraryStore }) => {
        if (cancelled) return;
        const publish = (items: LibraryItem[]) => {
          const files: LibrarySearchEntry[] = [];
          const fineTunes: LibrarySearchEntry[] = [];
          for (const item of items) {
            if (item.archived) continue;
            const entry = { item, icon: KIND_ICONS[fileKind(item)] };
            (isModelItem(item) ? fineTunes : files).push(entry);
          }
          const newest = (a: LibrarySearchEntry, b: LibrarySearchEntry) =>
            libraryTime(b.item) - libraryTime(a.item);
          setSources((prev) => ({
            ready: prev.ready || isSettled(useLibraryStore.getState().status),
            files: files.sort(newest),
            fineTunes: fineTunes.sort(newest),
          }));
        };
        publish(useLibraryStore.getState().items);
        unsubscribe = useLibraryStore.subscribe((state, prev) => {
          if (state.items !== prev.items || state.status !== prev.status) {
            publish(state.items);
          }
        });
        void useLibraryStore.getState().refresh().catch(() => {});
      })
      .catch(() => {
        if (!cancelled) setSources((prev) => ({ ...prev, ready: true }));
      });

    return () => {
      cancelled = true;
      unsubscribe?.();
    };
  }, [open]);

  return sources;
}

const isSettled = (status: string) => status === "ready" || status === "error";

export function libraryTime(item: LibraryItem): number {
  return Math.max(item.openedAt ?? 0, item.updatedAt);
}
