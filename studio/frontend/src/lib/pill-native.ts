// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

export type PillNativeStatus = {
  supported: boolean;
  enabled: boolean;
  hotkey: string;
  excludedApps: string[];
};

export type PillNativeConfig = {
  enabled: boolean;
  hotkey: string;
  excludedApps: string[];
};

async function invokeNative<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri) {
    throw new Error("Native desktop features are only available in the Tauri app.");
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export const isMacPlatform = (): boolean =>
  typeof navigator !== "undefined" && /Mac/.test(navigator.userAgent);

export async function pillStatus(): Promise<PillNativeStatus> {
  return invokeNative<PillNativeStatus>("pill_status");
}

export async function pillSetConfig(config: PillNativeConfig): Promise<PillNativeStatus> {
  return invokeNative<PillNativeStatus>("pill_set_config", { config });
}

export async function pillServerPort(): Promise<number | null> {
  return invokeNative<number | null>("pill_server_port");
}

export async function askHide(): Promise<void> {
  return invokeNative<void>("ask_hide");
}

export async function askResize(width: number, height: number): Promise<void> {
  return invokeNative<void>("ask_resize", { width, height });
}
