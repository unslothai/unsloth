// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export async function loadKeepAwake(): Promise<boolean | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean | null>("get_keep_awake");
}

export async function updateKeepAwake(enabled: boolean): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("set_keep_awake", { enabled });
}
