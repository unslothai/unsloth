// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import {
  type HostClass,
  classifyHost,
} from "@/features/model-picker/components/model-selector/host-artifact-policy";
import { useMemo } from "react";
import { useGpuInfo } from "./use-gpu-info";

/** What this host can run, for the media pickers.
 *
 * The two inputs live in different places -- the OS in the platform store, the resolved torch
 * backend in system info -- so every caller reads them the same way from here. */
export function useHostClass(): HostClass {
  const gpu = useGpuInfo();
  const deviceType = usePlatformStore((s) => s.deviceType);
  return useMemo(
    () =>
      classifyHost({
        deviceType,
        deviceBackend: gpu.backend,
        budgetKnown: gpu.budgetKnown,
      }),
    [deviceType, gpu.backend, gpu.budgetKnown],
  );
}
