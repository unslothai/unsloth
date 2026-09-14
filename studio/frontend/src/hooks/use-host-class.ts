// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import {
  type HostClass,
  classifyHost,
} from "@/features/model-picker/components/model-selector/host-artifact-policy";
import { useMemo } from "react";
import { useGpuInfo } from "./use-gpu-info";

/** Combine platform and backend state into a media-picker host class. */
export function useHostClass(): HostClass {
  const gpu = useGpuInfo();
  const deviceType = usePlatformStore((s) => s.deviceType);
  return useMemo(
    () =>
      classifyHost({
        deviceType,
        deviceBackend: gpu.backend,
        budgetKnown: gpu.budgetKnown,
        denseQuantSupported: gpu.denseQuantSupported,
      }),
    [deviceType, gpu.backend, gpu.budgetKnown, gpu.denseQuantSupported],
  );
}

/** The dense quant schemes this host can run, best first. Separate from `useHostClass` because
 *  the class answers "can it", and this answers "with what": a row that names the precision it
 *  will run needs the scheme, and only the backend knows whether that is fp8 or int8. */
export function useDenseQuantSchemes(): readonly string[] {
  return useGpuInfo().denseQuantSchemes;
}
