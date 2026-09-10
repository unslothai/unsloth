// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import {
  type HostClass,
  classifyHost,
} from "@/features/model-picker/components/model-selector/host-artifact-policy";
import { useMemo } from "react";
import { useGpuInfo } from "./use-gpu-info";

/** The explicit quant schemes this host can run, for the picker's row labels. */
export function useDenseQuantSchemes(): string[] | undefined {
  return useGpuInfo().denseQuantSchemes;
}

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
