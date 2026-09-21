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
