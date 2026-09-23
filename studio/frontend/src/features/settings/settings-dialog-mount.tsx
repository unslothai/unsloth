// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense, useEffect, useState } from "react";

import { useMonitorOverlayStore } from "./stores/monitor-overlay-store";
import { SettingsDialog } from "./settings-dialog";

const FloatingMonitor = lazy(() =>
  import("@/components/floating-monitor").then((module) => ({
    default: module.FloatingMonitor,
  })),
);

export function SettingsDialogMount({ active }: { active: boolean }) {
  const isMonitorOpen = useMonitorOverlayStore((state) => state.isOpen);
  const [monitorWasMounted, setMonitorWasMounted] = useState(false);

  useEffect(() => {
    if (active && isMonitorOpen) setMonitorWasMounted(true);
  }, [active, isMonitorOpen]);

  if (!active) return null;
  return (
    <>
      <SettingsDialog />
      {(isMonitorOpen || monitorWasMounted) && (
        <Suspense fallback={null}>
          <FloatingMonitor />
        </Suspense>
      )}
    </>
  );
}
