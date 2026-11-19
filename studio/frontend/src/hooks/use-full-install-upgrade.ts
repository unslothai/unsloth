// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import {
  type TauriRepairController,
  canUpgradeForVideo,
  canUpgradeToFullInstall,
  useTauriRepairController,
} from "./tauri-repair-context";

export function useFullInstallUpgrade(): TauriRepairController | null {
  const repair = useTauriRepairController();
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const chatOnlyHint = usePlatformStore((s) => s.chatOnlyHint);
  const verdictMeasured = usePlatformStore(
    (s) => s.isChatOnly() && !s.capabilitiesUnknown(),
  );
  return canUpgradeToFullInstall(chatOnlyReason, chatOnlyHint, verdictMeasured, repair)
    ? repair
    : null;
}

/** The full install for the Video page; its own eligibility (``canUpgradeForVideo``), so a training-only blocker does
 * not hide it. Kept here so the Video page never reads the training verdict itself. */
export function useFullInstallBringsVideo(): TauriRepairController | null {
  const repair = useTauriRepairController();
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const chatOnlyHint = usePlatformStore((s) => s.chatOnlyHint);
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cpuTraining = usePlatformStore((s) => s.cpuTraining);
  const verdictMeasured = usePlatformStore(
    (s) => s.isChatOnly() && !s.capabilitiesUnknown(),
  );
  return canUpgradeForVideo(chatOnlyReason, chatOnlyHint, verdictMeasured, repair, {
    deviceType,
    cpuTraining,
  })
    ? repair
    : null;
}
