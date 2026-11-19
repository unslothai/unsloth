// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import {
  type TauriRepairController,
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

/** The full install brings video only where it brings a video device: not on a GPU-less host, where it trains on the
 * CPU. Kept here so the Video page never reads the training verdict itself. */
export function useFullInstallBringsVideo(): TauriRepairController | null {
  const repair = useFullInstallUpgrade();
  const cpuTraining = usePlatformStore((s) => s.cpuTraining);
  return cpuTraining ? null : repair;
}
