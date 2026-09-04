// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  downloadPercent,
  sameUpdateVersion,
  type DesktopUpdateBundleStatus,
  type StagedUpdateStatus,
} from "@/lib/update-preparation";

export interface DesktopUpdateMetadata {
  currentVersion: string;
  version: string;
  date?: string;
  body?: string;
  rawJson: Record<string, unknown>;
}

interface DesktopUpdateDownloadEvent {
  version: string;
  downloaded: number;
  total: number | null;
}

export async function checkDesktopUpdate(): Promise<DesktopUpdateMetadata | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<DesktopUpdateMetadata | null>("check_desktop_update");
}

export async function desktopUpdateBundleStatus(): Promise<DesktopUpdateBundleStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<DesktopUpdateBundleStatus>("desktop_update_bundle_status");
}

export async function downloadDesktopUpdate(
  expectedVersion: string,
  onProgress: (percent: number) => void,
): Promise<void> {
  const [{ invoke }, { listen }] = await Promise.all([
    import("@tauri-apps/api/core"),
    import("@tauri-apps/api/event"),
  ]);
  const unlisten = await listen<DesktopUpdateDownloadEvent>(
    "desktop-update-download",
    (event) => {
      if (!sameUpdateVersion(event.payload.version, expectedVersion)) return;
      onProgress(downloadPercent(event.payload.downloaded, event.payload.total));
    },
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

export async function waitForDesktopUpdateDownload(
  expectedVersion: string,
  onProgress: (percent: number) => void,
  isCancelled: () => boolean,
  pollMs = 500,
): Promise<void> {
  const { listen } = await import("@tauri-apps/api/event");
  const unlisten = await listen<DesktopUpdateDownloadEvent>(
    "desktop-update-download",
    (event) => {
      if (!sameUpdateVersion(event.payload.version, expectedVersion)) return;
      onProgress(downloadPercent(event.payload.downloaded, event.payload.total));
    },
  );
  try {
    while (!isCancelled()) {
      const status = await desktopUpdateBundleStatus();
      if (!status.downloading) return;
      await new Promise((resolve) => setTimeout(resolve, pollMs));
    }
  } finally {
    unlisten();
  }
}

export async function installDesktopUpdate(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("install_desktop_update");
}

export async function stagedUpdateStatus(): Promise<StagedUpdateStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<StagedUpdateStatus>("staged_update_status");
}

export async function startStagedUpdate(onLine: (line: string) => void): Promise<void> {
  const [{ invoke }, { listen }] = await Promise.all([
    import("@tauri-apps/api/core"),
    import("@tauri-apps/api/event"),
  ]);
  const unlisten = await listen<string>("stage-progress", (event) => onLine(event.payload));
  try {
    await invoke("start_staged_update");
  } finally {
    unlisten();
  }
}

/// Follow a staged run this webview did not start. Polls rather than waiting on
/// stage-complete, because the run can finish between the status read that chose
/// this path and any listener registered after it.
export async function adoptStagedUpdate(
  onLine: (line: string) => void,
  isCancelled: () => boolean,
  pollMs = 2_000,
): Promise<StagedUpdateStatus> {
  const { listen } = await import("@tauri-apps/api/event");
  const unlisten = await listen<string>("stage-progress", (event) => onLine(event.payload));
  try {
    for (;;) {
      const status = await stagedUpdateStatus();
      if (!status.staging || isCancelled()) return status;
      await new Promise((resolve) => setTimeout(resolve, pollMs));
    }
  } finally {
    unlisten();
  }
}

export async function cancelStagedUpdate(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("cancel_staged_update");
}

export async function discardStagedUpdate(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("discard_staged_update");
}
