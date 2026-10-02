// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

// A Tauri quit never fires beforeunload, so the desktop close button asks Rust whether any
// backend download is in flight. Each source reports here and Rust gets one combined answer:
// two sources writing the flag directly would overwrite each other.
type DownloadSource = "hub" | "npu";

const activeSources = new Set<DownloadSource>();
let published: boolean | null = null;

export function reportDownloadsActive(
  source: DownloadSource,
  active: boolean,
): void {
  if (active) {
    activeSources.add(source);
  } else {
    activeSources.delete(source);
  }
  const anyActive = activeSources.size > 0;
  if (anyActive === published) return;
  published = anyActive;
  if (!isTauri) return;
  void import("@tauri-apps/api/core")
    .then(({ invoke }) =>
      invoke("set_renderer_activity", { kind: "downloads", active: anyActive }),
    )
    .catch(() => {});
}
