// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Live state for the toolbar's Downloads button: running downloads and the latest finish.
// Session only; history-store.ts keeps the lasting record.

import { getLocale, translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { useBrowserHistoryStore } from "./history-store";

export type FinishedDownload = {
  key: string;
  name: string;
  size: number;
  contentType: string;
  url: string | null;
  /** Desktop app id for the saved file, to reveal or open it. */
  nativeId?: string;
  /** Its Download history row, when history is kept. */
  historyId?: string;
  failed: boolean;
};

type DownloadActivity = {
  active: Record<string, string>;
  /** The result a Downloads button is showing; null once dismissed or when none was on screen. */
  finished: FinishedDownload | null;
  /** Bumped per finish, so the same file downloaded twice shows again. */
  finishedSequence: number;
  /** Downloads buttons on screen. With none, a finish has nowhere to show but a toast. */
  buttons: number;
  dismissFinished: () => void;
};

// Saved from the panel this session, so Open can show the file again without the disk.
const MAX_KEPT_FILES = 8;
const keptFiles = new Map<string, { blob: Blob; name: string; contentType: string }>();

/** A result with no history row can only be reached from its notice: drop its bytes as that goes. */
function release(finished: FinishedDownload | null): void {
  if (finished && !finished.historyId) keptFiles.delete(finished.key);
}

export const useDownloadActivity = create<DownloadActivity>((set, get) => ({
  active: {},
  finished: null,
  finishedSequence: 0,
  buttons: 0,
  dismissFinished: () => {
    release(get().finished);
    set({ finished: null });
  },
}));

export function keptDownloadFile(id: string | undefined) {
  return id ? (keptFiles.get(id) ?? null) : null;
}

// Kept under a history row's id: they go when the row does (removed, cleared or pushed out).
const keptRows = new Set<string>();
let watchingRows = false;

/** Subscribed on first use, not at load: the history store sits in the chat import cycle. */
function watchRows(): void {
  if (watchingRows) return;
  watchingRows = true;
  useBrowserHistoryStore.subscribe((state, previous) => {
    if (state.downloads === previous.downloads || keptRows.size === 0) return;
    const rows = new Set(state.downloads.map((item) => item.id));
    for (const id of keptRows) {
      if (rows.has(id)) continue;
      keptRows.delete(id);
      keptFiles.delete(id);
    }
  });
}

function keepFile(id: string, file: { blob: Blob; name: string; contentType: string }, row: boolean): void {
  keptFiles.delete(id);
  keptFiles.set(id, file);
  if (row) {
    watchRows();
    keptRows.add(id);
  }
  for (const old of [...keptFiles.keys()].slice(0, Math.max(0, keptFiles.size - MAX_KEPT_FILES))) {
    keptFiles.delete(old);
    keptRows.delete(old);
  }
}

export function mountDownloadsButton(): () => void {
  useDownloadActivity.setState((state) => ({ buttons: state.buttons + 1 }));
  return () => {
    useDownloadActivity.setState((state) => ({ buttons: state.buttons - 1 }));
    // With the last button its notice and timer go too. A tab change swaps buttons in one commit,
    // so only once none came back.
    queueMicrotask(() => {
      const { buttons, dismissFinished } = useDownloadActivity.getState();
      if (buttons === 0) dismissFinished();
    });
  };
}

/** Marks a download as running under `key`; the button spins until it ends. */
export function beginDownload(key: string, name: string): void {
  useDownloadActivity.setState((state) => ({ active: { ...state.active, [key]: name } }));
}

/** Drops a download that never started (a save dialog cancelled), without a result to show. */
export function abandonDownload(key: string): void {
  useDownloadActivity.setState((state) => {
    if (!(key in state.active)) return state;
    const active = { ...state.active };
    delete active[key];
    return { active };
  });
}

/** Ends `key` and shows the result on the Downloads button. `historyId`: what recordDownload returned.
 *  False when no button is on screen. */
export function finishDownload(
  key: string,
  result: Omit<FinishedDownload, "key">,
  file?: { blob: Blob; name: string; contentType: string },
): boolean {
  const { finished: replaced, buttons } = useDownloadActivity.getState();
  // A replaced result the list can't show (failed or unrecorded) would vanish: toast it instead.
  if (replaced && !replaced.historyId) {
    const title = replaced.failed ? "browser.downloads.failed" : "browser.downloads.complete";
    toast[replaced.failed ? "error" : "success"](translate(title, {}, getLocale()), { description: replaced.name });
  }
  release(replaced);
  // Unrecorded, its native id was forgotten (history-store.ts), so it can't open or reveal.
  const nativeId = result.historyId ? result.nativeId : undefined;
  // Only a history row or a notice on screen can reach the copy; with neither it would only sit in memory.
  if (file && (result.historyId || buttons > 0)) keepFile(result.historyId ?? key, file, Boolean(result.historyId));
  const active = { ...useDownloadActivity.getState().active };
  delete active[key];
  // With no button on screen the caller toasts it, so nothing holds it here.
  useDownloadActivity.setState((state) => ({
    active,
    finished: buttons > 0 ? { ...result, nativeId, key } : null,
    finishedSequence: state.finishedSequence + 1,
  }));
  return buttons > 0;
}
