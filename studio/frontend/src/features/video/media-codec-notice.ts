// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Linux desktop (deb or extracted install): WebKitGTK plays generated mp4s through the system
// GStreamer, which needs gst-libav (avdec_h264, avdec_aac) and gst-plugins-good (qtdemux).
// The desktop shell reports which Debian packages are missing; this shows the install command once.

export type MissingPackagesProbe = () => Promise<readonly string[]>;

let checked = false;

/** The apt command for the missing packages, or null when nothing is missing. */
export function mediaCodecInstallCommand(missing: readonly string[]): string | null {
  const packages = missing.filter((name) => /^[a-z0-9][a-z0-9.+-]*$/.test(name));
  return packages.length ? `sudo apt install ${packages.join(" ")}` : null;
}

/** Runs the probe once per app session; a failed probe (older shell, browser) shows nothing. */
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

/** The desktop shell's probe. */
export async function desktopMissingMediaCodecPackages(): Promise<readonly string[]> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<string[]>("missing_media_codec_packages");
}

export function resetMediaCodecCheckForTests(): void {
  checked = false;
}
