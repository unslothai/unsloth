// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { FloatingMonitor } from "@/components/floating-monitor";
import { SettingsDialog } from "./settings-dialog";

export function SettingsDialogMount({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <>
      <SettingsDialog />
      <FloatingMonitor />
    </>
  );
}
