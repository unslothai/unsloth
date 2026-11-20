// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { authFetch } from "@/features/auth";
import {
  type HardwareInfo,
  refreshHardwareInfo,
} from "@/hooks/use-hardware-info";
import { useEffect, useState } from "react";

export type DeviceSummary =
  | { status: "checking" }
  | { status: "error" }
  | {
      status: "ready";
      gpuName: string | null;
      gpuCount: number;
      vramGb: number | null;
      unifiedMemory: boolean;
      appleSilicon: boolean;
      ramGb: number | null;
      train: boolean;
      media: boolean;
    };

async function readRamGb(): Promise<number | null> {
  try {
    const res = await authFetch("/api/system");
    if (!res.ok) return null;
    const data = (await res.json()) as { memory?: Record<string, unknown> };
    const total = data.memory?.total_gb;
    return typeof total === "number" ? total : null;
  } catch {
    return null;
  }
}

function toSummary(
  hardware: HardwareInfo,
  ramGb: number | null,
): DeviceSummary {
  const platform = usePlatformStore.getState();
  const primary = hardware.gpus[0];
  return {
    status: "ready",
    gpuName: hardware.gpuName ?? primary?.name ?? null,
    gpuCount: hardware.gpus.length,
    vramGb: hardware.vramTotalGb ?? primary?.vramTotalGb ?? null,
    unifiedMemory: platform.appleSilicon || primary?.unifiedMemory === true,
    appleSilicon: platform.appleSilicon,
    ramGb,
    train: !platform.isTrainUnavailable(),
    media: !platform.isChatOnly(),
  };
}

// Detection is deferred until a page asks for it; this step is that ask.
export function useDeviceSummary(): DeviceSummary {
  const [summary, setSummary] = useState<DeviceSummary>({ status: "checking" });
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await fetchDeviceType({ detectHardware: true });
        const [hardware, ramGb] = await Promise.all([
          refreshHardwareInfo(),
          readRamGb(),
        ]);
        if (cancelled) return;
        if (!hardware.loaded) {
          setSummary({ status: "error" });
          return;
        }
        setSummary(toSummary(hardware, ramGb));
      } catch {
        if (!cancelled) setSummary({ status: "error" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  return summary;
}
