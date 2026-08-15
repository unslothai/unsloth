// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export async function checkDesktopUpdate() {
  const [{ invoke }, { Update }] = await Promise.all([
    import("@tauri-apps/api/core"),
    import("@tauri-apps/plugin-updater"),
  ]);
  const metadata = await invoke<ConstructorParameters<typeof Update>[0] | null>(
    "check_desktop_update",
  );
  return metadata ? new Update(metadata) : null;
}
