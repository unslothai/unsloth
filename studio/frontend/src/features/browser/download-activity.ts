// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Live state for the toolbar's Downloads button: running downloads and the latest finish.
// Session only; history-store.ts keeps the lasting record.

import { getLocale, translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { create } from "zustand";

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

/** Whether the current result went to a button rather than its caller's toast. */
let finishedOnButton = false;

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

function keepFile(id: string, file: { blob: Blob; name: string; contentType: string }): void {
  keptFiles.delete(id);
  keptFiles.set(id, file);
  for (const old of [...keptFiles.keys()].slice(0, Math.max(0, keptFiles.size - MAX_KEPT_FILES))) {
    keptFiles.delete(old);
  }
}

export function mountDownloadsButton(): () => void {
  useDownloadActivity.setState((state) => ({ buttons: state.buttons + 1 }));
  return () => useDownloadActivity.setState((state) => ({ buttons: state.buttons - 1 }));
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
  if (replaced && !replaced.historyId && finishedOnButton) {
    const title = replaced.failed ? "browser.downloads.failed" : "browser.downloads.complete";
    toast[replaced.failed ? "error" : "success"](translate(title, {}, getLocale()), { description: replaced.name });
  }
  release(replaced);
  // Unrecorded, its native id was forgotten (history-store.ts), so it can't open or reveal.
  const nativeId = result.historyId ? result.nativeId : undefined;
  if (file) keepFile(result.historyId ?? key, file);
  const active = { ...useDownloadActivity.getState().active };
  delete active[key];
  finishedOnButton = buttons > 0;
  useDownloadActivity.setState((state) => ({
    active,
    finished: { ...result, nativeId, key },
    finishedSequence: state.finishedSequence + 1,
  }));
  return finishedOnButton;
}
