// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface DesktopUpdateMetadata {
  currentVersion: string;
  version: string;
  date?: string;
  body?: string;
  rawJson: Record<string, unknown>;
}

export interface DesktopUpdateBundleStatus {
  version: string | null;
  downloaded: boolean;
  downloading: boolean;
}

interface DesktopUpdateDownloadEvent {
  version: string;
  downloaded: number;
  total: number | null;
}

const LEADING_V = /^v/;

export function sameUpdateVersion(left: string | null | undefined, right: string): boolean {
  if (!left) return false;
  return left.replace(LEADING_V, "") === right.replace(LEADING_V, "");
}

export function downloadPercent(downloaded: number, total: number | null): number {
  if (!total || total <= 0) return 0;
  return Math.min(100, Math.round((downloaded / total) * 100));
}

export async function checkDesktopUpdate(): Promise<DesktopUpdateMetadata | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<DesktopUpdateMetadata | null>("check_desktop_update");
}

export async function desktopUpdateBundleStatus(): Promise<DesktopUpdateBundleStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<DesktopUpdateBundleStatus>("desktop_update_bundle_status");
}

/**
 * Progress for the one bundle download in flight, whoever started it. Split out of
 * `downloadDesktopUpdate` because a webview reload leaves the native download running with
 * nothing listening, and the update that comes back has none of its own to report on.
 */
export async function listenDesktopUpdateDownload(
  expectedVersion: string,
  onProgress: (percent: number) => void,
): Promise<() => void> {
  const { listen } = await import("@tauri-apps/api/event");
  return listen<DesktopUpdateDownloadEvent>(
    "desktop-update-download",
    (event) => {
      if (!sameUpdateVersion(event.payload.version, expectedVersion)) return;
      onProgress(downloadPercent(event.payload.downloaded, event.payload.total));
    },
  );
}

export async function downloadDesktopUpdate(
  expectedVersion: string,
  onProgress: (percent: number) => void,
): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  const unlisten = await listenDesktopUpdateDownload(
    expectedVersion,
    onProgress,
  );
  try {
    await invoke("download_desktop_update");
    const status = await desktopUpdateBundleStatus();
    if (!status.downloaded || !sameUpdateVersion(status.version, expectedVersion)) {
      throw new Error(`Desktop update ${expectedVersion} was not downloaded.`);
    }
    onProgress(100);
  } finally {
    unlisten();
  }
}

export async function installDesktopUpdate(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("install_desktop_update");
}

// ── Background preparation ──

/** `prefetch.rs::PrefetchStatus`. */
export interface PrefetchStatus {
  state: "none" | "ready" | "noop" | "partial" | "stale";
  backendVersion: string | null;
  shellVersion: string | null;
  cacheDir: string | null;
  createdAt: number | null;
  running: boolean;
  runningShellVersion: string | null;
}

/** The installed backend has no `prefetch-update` command. Nothing to report. */
export const PREFETCH_UNSUPPORTED = "prefetch-unsupported";
/** Another prefetch owns the work; whatever it produces is what gets adopted. */
export const PREFETCH_BUSY = "prefetch-busy";

/**
 * Why a prefetch stopped. Only `failed` is worth a word to the user, and even
 * that one is not fatal: the restart falls back to downloading, exactly as it
 * did before there was a prefetch at all.
 */
export type PrefetchOutcome = "ready" | "unsupported" | "busy" | "failed";

const PREFETCH_POLL_MS = 1000;

export async function prefetchStatus(): Promise<PrefetchStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<PrefetchStatus>("prefetch_status");
}

export async function startPrefetch(
  shellVersion: string,
  onLog: (line: string) => void,
): Promise<PrefetchOutcome> {
  const [{ invoke }, { listen }] = await Promise.all([
    import("@tauri-apps/api/core"),
    import("@tauri-apps/api/event"),
  ]);
  const unlisten = await listen<string>("prefetch-progress", (event) => {
    onLog(event.payload);
  });
  try {
    await invoke("start_prefetch_update", { shellVersion });
    return "ready";
  } catch (e) {
    const reason = String(e);
    // A backend that predates this feature exits with click's usage error, which
    // the shell maps to this token. It is the expected answer on the release that
    // introduces the prefetch, not a fault.
    if (reason.includes(PREFETCH_UNSUPPORTED)) return "unsupported";
    if (reason.includes(PREFETCH_BUSY)) return "busy";
    console.warn("Background update preparation failed:", e);
    return "failed";
  } finally {
    unlisten();
  }
}

/**
 * Join a prefetch this renderer did not start.
 *
 * A webview reload leaves the native child running with no listener attached and
 * the shell refuses a second one, so the only thing to do is wait it out.
 */
export async function adoptPrefetch(
  cancelled: () => boolean,
): Promise<PrefetchStatus> {
  for (;;) {
    const status = await prefetchStatus();
    if (!status.running || cancelled()) return status;
    await new Promise((resolve) => setTimeout(resolve, PREFETCH_POLL_MS));
  }
}

export async function cancelPrefetch(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("cancel_prefetch_update");
}

export async function discardPrefetch(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("discard_prefetch");
}
