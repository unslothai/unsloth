// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

export type GpuRenderingSetting = "auto" | "on" | "off";

export interface GpuRenderingStatus {
  setting: GpuRenderingSetting;
  /** False on macOS, where WebKit always renders with the GPU. */
  configurable: boolean;
  restartRequired: boolean;
  /** GPU rendering crashed at launch, so this launch fell back to CPU rendering. */
  fallbackActive: boolean;
  /** Name of the environment variable that overrides the setting. */
  envOverride: string | null;
  /** Runtime WebKitGTK version; set on Linux Wayland below 2.54.2 only. */
  oldWebkitWarning: string | null;
  /** NVIDIA driver branch; set on Linux X11 below 610, where On keeps compositing off. */
  oldDriverWarning: string | null;
}

export async function loadGpuRendering(): Promise<GpuRenderingStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<GpuRenderingStatus>("get_gpu_rendering");
}

export async function updateGpuRendering(
  value: GpuRenderingSetting,
): Promise<GpuRenderingStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<GpuRenderingStatus>("set_gpu_rendering", { value });
}

/** False when a quit is already pending; otherwise the app quits through its usual confirmations. */
export async function relaunchToApplyGpuRendering(): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("relaunch_to_apply_gpu_rendering");
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });
}

/**
 * Tells the desktop shell the window has painted, which ends the GPU startup
 * watch. Waits two frames so the first frame after show() has been drawn.
 */
export async function notifyRendererReady(): Promise<void> {
  if (!isTauri) return;
  try {
    await nextFrame();
    await nextFrame();
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("renderer_ready");
  } catch {
    // Older shells lack the command, and the shell has its own ready path.
  }
}
