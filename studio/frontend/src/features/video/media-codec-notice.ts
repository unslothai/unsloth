// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// WebKitGTK plays mp4s via system GStreamer (needs gst-libav + gst-plugins-good); show the apt fix once.

export type MissingPackagesProbe = () => Promise<readonly string[]>;

let checked = false;

export function mediaCodecInstallCommand(missing: readonly string[]): string | null {
  const packages = missing.filter((name) => /^[a-z0-9][a-z0-9.+-]*$/.test(name));
  return packages.length ? `sudo apt install ${packages.join(" ")}` : null;
}

/** Once per session; a failed probe (older shell, browser) shows nothing. */
export async function checkMediaCodecsOnce(
  probe: MissingPackagesProbe,
  notify: (command: string) => void,
): Promise<void> {
  if (checked) return;
  checked = true;
  let missing: readonly string[];
  try {
    missing = await probe();
  } catch {
    return;
  }
  const command = mediaCodecInstallCommand(Array.isArray(missing) ? missing : []);
  if (command) notify(command);
}

export async function desktopMissingMediaCodecPackages(): Promise<readonly string[]> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<string[]>("missing_media_codec_packages");
}

export function resetMediaCodecCheckForTests(): void {
  checked = false;
}
