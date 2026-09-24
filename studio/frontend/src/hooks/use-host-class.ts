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

/** Whether the backend accepts NVFP4 for image/video generation (its UNSLOTH_NVFP4_DIFFUSION switch). */
export function useNvfp4Diffusion(): boolean {
  return useGpuInfo().nvfp4Diffusion;
}

/** Whether `/api/system` has answered yet, i.e. whether a false `useNvfp4Diffusion()` is the
 *  backend's word rather than the not-yet-loaded default. */
export function useNvfp4DiffusionKnown(): boolean {
  return useGpuInfo().budgetKnown;
}

/** The dense quant schemes this host can run, best first. */
export function useDenseQuantSchemes(): readonly string[] {
  return useGpuInfo().denseQuantSchemes;
}
